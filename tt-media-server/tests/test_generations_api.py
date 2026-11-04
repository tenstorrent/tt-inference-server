# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the generations API: the models' request schemas, checked
against docs/api/generation, and submit → job → content → delete
through the router with a fake device. Runners and tt_dit run on hardware, not
here."""

import asyncio
import base64
import importlib.util
import json
import pathlib
import threading
import wave
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import yaml
from config.constants import JobTypes
from domain.flux2_klein_4b_request import Flux2klein4bRequest
from domain.flux3_image_klein_request import Flux3ImageKleinRequest
from domain.generation_request import MEDIA_MAX_BYTES
from domain.ideogram4_request import Ideogram4Request
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.generation_models import GENERATION_MODELS
from domain.minimax_h3_request import MiniMaxH3FL2VARequest, MiniMaxH3Ref2VARequest
from domain.qwen_image_edit_2511_request import QwenImageEdit2511Request
from domain.wan22_i2v_a14b_request import WAN_NEGATIVE_PROMPT, Wan22i2vA14bRequest
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError
from utils.media_probe import ReferenceMediaProbe, media_format_error
from utils.video_manager import VideoManager

_SPEC_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "api" / "generation"

_CAPTION = {
    "compositional_deconstruction": {
        "background": "a brick wall",
        "elements": [{"type": "text", "text": "OPEN", "bbox": [10, 20, 30, 40]}],
    }
}


def _data_uri(width: int = 64, height: int = 64, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format=image_format)
    return f"data:image/{image_format.lower()};base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _with_images(request, **images):
    request._images = {
        name: [Image.new("RGB", size) for size in sizes]
        for name, sizes in images.items()
    }
    return request


# --- Request schemas against the spec ----------------------------------------------


def _resolved(schema: dict, defs: dict) -> dict:
    """A property's own schema: Optional's non-null branch, $refs followed."""
    if "anyOf" in schema:
        (schema,) = [branch for branch in schema["anyOf"] if branch != {"type": "null"}]
    if "$ref" in schema and schema["$ref"].startswith("#/$defs/"):
        schema = defs[schema["$ref"].removeprefix("#/$defs/")]
    return schema


@pytest.mark.parametrize(
    "model", GENERATION_MODELS.values(), ids=lambda model: model.id
)
def test_request_schema_matches_the_spec(model):
    request_model = model.request
    path = _SPEC_DIR / "models" / f"{model.id}.yaml"
    spec = yaml.safe_load(path.read_text())
    submit = spec["paths"][f"/v1/generations/{model.id}"]["post"]
    # A variant's file takes its base model's schema by reference.
    file, _, pointer = submit["requestBody"]["content"]["application/json"]["schema"][
        "$ref"
    ].partition("#/components/schemas/")
    if file:
        spec = yaml.safe_load((path.parent / file).read_text())
    expected = spec["components"]["schemas"][pointer]
    actual = request_model.model_json_schema()
    defs = actual.get("$defs", {})

    assert actual["additionalProperties"] is False
    assert set(actual["properties"]) == set(expected["properties"])
    assert set(actual.get("required", [])) == set(expected["required"])
    for name, want in expected["properties"].items():
        got = _resolved(actual["properties"][name], defs)
        default = actual["properties"][name].get("default")
        if "enum" in want:
            assert got.get("enum") == want["enum"], name
        if "default" in want:
            assert default == want["default"], name
        for bound in ("minimum", "maximum", "minItems", "maxItems", "minLength"):
            if bound in want:
                assert got.get(bound) == want[bound], (name, bound)


def test_ideogram_caption_matches_the_spec():
    spec = yaml.safe_load(
        (_SPEC_DIR / "models" / "ideogram-ai" / "ideogram-4.yaml").read_text()
    )["components"]["schemas"]
    defs = Ideogram4Request.model_json_schema()["$defs"]
    for name in (
        "Ideogram4Caption",
        "Ideogram4ObjectElement",
        "Ideogram4TextElement",
    ):
        assert list(defs[name]["properties"]) == list(spec[name]["properties"]), name
        assert set(defs[name].get("required", [])) == set(spec[name]["required"])


def test_unknown_field_rejected():
    with pytest.raises(ValidationError, match="extra"):
        Flux2klein4bRequest(prompt="a cat", steps=4)


@pytest.mark.parametrize(
    "media",
    ["http://example.com/a.png", "iVBORw0KGgo=", "data:text/plain;base64,aGk="],
)
def test_media_must_be_https_or_image_data_uri(media):
    with pytest.raises(ValidationError, match="https:// URL or a base64 data URI"):
        Wan22i2vA14bRequest(
            prompt="a boat", first_frame=media, aspect_ratio="16:9", size="480p"
        )


def test_wan_fields():
    request = Wan22i2vA14bRequest(
        prompt="a boat", first_frame="https://x/a.png", aspect_ratio="9:16", size="720p"
    )
    assert (request.frames, request.fps) == (81, 16)
    assert request.negative_prompt == WAN_NEGATIVE_PROMPT
    assert request.output_size() == (720, 1280)
    for fields in ({"aspect_ratio": "1:1"}, {"frames": 97}, {"aspect_ratio": None}):
        with pytest.raises(ValidationError):
            Wan22i2vA14bRequest(
                prompt="a boat",
                first_frame="https://x/a.png",
                **{"aspect_ratio": "16:9", "size": "480p", **fields},
            )
    assert len(set(Wan22i2vA14bRequest.warmup_sizes())) == 4


@pytest.mark.parametrize(
    "fields",
    [{}, {"duration": 6, "fps": 25, "aspect_ratio": "16:9", "size": "1080p"}],
)
def test_ltx_serves_one_shape(fields):
    request = Ltx23DistilledRequest(prompt="a violinist", **fields)
    assert (request.frames, request.fps, request.output_size()) == (
        153,
        25,
        (1920, 1088),
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"duration": 5}, "Input should be 6"),
        ({"fps": 24}, "Input should be 25"),
        ({"aspect_ratio": "9:16"}, "Input should be '16:9'"),
        ({"size": "720p"}, "Input should be '1080p'"),
        ({"height": 1088}, "extra"),
        ({"num_inference_steps": 11}, "extra"),
    ],
)
def test_ltx_rejects(fields, message):
    with pytest.raises(ValidationError, match=message):
        Ltx23DistilledRequest(prompt="a violinist", **fields)


def test_qwen_size_follows_last_image():
    request = QwenImageEdit2511Request(
        prompt="add a hat", images=["https://x/a.png", "https://x/b.png"]
    )
    assert request.negative_prompt == " "
    _with_images(request, images=[(1000, 1000), (1600, 900)])
    assert request.output_size() == (1376, 768)
    request.aspect_ratio = "3:4"
    assert request.output_size() == (896, 1184)


@pytest.mark.parametrize(
    "fields", [{"images": ["https://x/a.png"] * 4}, {"count": 2}, {"size": "1K"}]
)
def test_qwen_rejects(fields):
    with pytest.raises(ValidationError):
        QwenImageEdit2511Request(
            prompt="x", **{"images": ["https://x/a.png"], **fields}
        )


@pytest.mark.parametrize(
    ("fields", "images", "expected"),
    [
        ({}, [], (1024, 1024)),
        ({"size": "2K"}, [], (2048, 2048)),
        ({"aspect_ratio": "16:9", "size": "2K"}, [], (2752, 1536)),
        ({}, [(1600, 900), (100, 100)], (1376, 768)),
        ({"size": "2K"}, [(900, 1250)], (1792, 2368)),
    ],
)
def test_klein_output_size(fields, images, expected):
    if images:
        fields["images"] = ["https://x/a.png"] * len(images)
        request = _with_images(
            Flux2klein4bRequest(prompt="a cat", **fields), images=images
        )
    else:
        request = Flux2klein4bRequest(prompt="a cat", **fields)
    assert request.output_size() == expected


def test_klein_has_14_sizes():
    sizes = set(Flux2klein4bRequest.warmup_sizes())
    assert len(sizes) == 14
    assert all(width % 16 == 0 and height % 16 == 0 for width, height in sizes)



@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, (2048, 2048)),
        ({"size": "1K"}, (1024, 1024)),
        ({"size": "4K"}, (4096, 4096)),
    ],
)
def test_flux3_output_size(fields, expected):
    assert Flux3ImageKleinRequest(prompt="a cat", **fields).output_size() == expected


@pytest.mark.parametrize(
    "fields", [{"aspect_ratio": "16:9"}, {"images": ["https://x/a.png"] * 2}]
)
def test_flux3_serves_one_square_shape_per_size(fields):
    with pytest.raises(ValidationError):
        Flux3ImageKleinRequest(prompt="a cat", **fields)


def test_output_reports_the_runners_prompt(tmp_path):
    from model_services.generation_service import GenerationService

    request = Flux3ImageKleinRequest(prompt="a cat")
    request._prompt, request._seed = "a cat", 5
    image = Image.new("RGB", (8, 8))
    image.info["prompt"] = "a tabby cat on a windowsill"
    output = GenerationService._write_output(tmp_path, 0, request, image)
    assert output["prompt"] == "a tabby cat on a windowsill"


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, (2048, 2048)),
        ({"size": "1K"}, (1024, 1024)),
        ({"aspect_ratio": "16:9", "size": "1K"}, (1280, 720)),
        ({"aspect_ratio": "3:2"}, (2048, 1360)),
    ],
)
def test_ideogram_output_size(fields, expected):
    request = Ideogram4Request(structured_prompt=_CAPTION, **fields)
    assert request.output_size() == expected
    assert len(set(Ideogram4Request.warmup_sizes())) == 14


def test_ideogram_caption_is_written_in_the_model_key_order():
    request = Ideogram4Request(
        structured_prompt={
            "compositional_deconstruction": {
                "elements": [{"desc": "d", "bbox": [0, 0, 1, 1], "type": "obj"}],
                "background": "b",
            },
            "style_description": {
                "medium": "m",
                "photo": "p",
                "lighting": "l",
                "aesthetics": "a",
            },
            "high_level_description": "Ein Café",
        }
    )
    assert request.model_prompt() == (
        '{"high_level_description":"Ein Café",'
        '"style_description":{"aesthetics":"a","lighting":"l","photo":"p","medium":"m"},'
        '"compositional_deconstruction":{"background":"b",'
        '"elements":[{"type":"obj","bbox":[0,0,1,1],"desc":"d"}]}}'
    )


@pytest.mark.parametrize(
    ("caption", "message"),
    [
        (
            {"compositional_deconstruction": {"background": "b", "elements": [
                {"type": "obj", "desc": "d", "bbox": [30, 0, 10, 10]}
            ]}},
            "ymin <= ymax",
        ),
        (
            {
                "style_description": {
                    "aesthetics": "a",
                    "lighting": "l",
                    "medium": "m",
                    "photo": "p",
                    "art_style": "s",
                },
                **_CAPTION,
            },
            "exactly one of photo and art_style",
        ),
        (
            {"compositional_deconstruction": {"background": "b", "elements": [
                {"type": "obj", "desc": "d", "color_palette": ["#ff0000"]}
            ]}},
            "pattern",
        ),
        ({"caption": "plain"}, "extra"),
    ],
)  # fmt: skip
def test_ideogram_rejects_invalid_captions(caption, message):
    with pytest.raises(ValidationError, match=message):
        Ideogram4Request(structured_prompt=caption)


def test_ideogram_has_no_plain_prompt():
    with pytest.raises(ValidationError, match="extra"):
        Ideogram4Request(prompt="a poster", structured_prompt=_CAPTION)


# --- Router and jobs -------------------------------------------------------------

_RETENTION_SECONDS = 3600


def _router_module(monkeypatch, runner: str):
    from config.settings import settings

    monkeypatch.setattr(settings, "model_runner", runner)
    monkeypatch.setattr(settings, "job_retention_seconds", _RETENTION_SECONDS)
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "generations.py"
    )
    spec = importlib.util.spec_from_file_location("generations_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_the_served_model_is_mounted(monkeypatch):
    module = _router_module(monkeypatch, "tt-ideogram-4")
    app = FastAPI()
    app.include_router(module.router)
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/v1/generations/ideogram-ai/ideogram-4",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/content/{index}",
        "/v1/jobs/{job_id}/cancel",
    }


@pytest.mark.parametrize("job_type", ["training", "video"])
def test_other_services_jobs_are_404(monkeypatch, job_type):
    # E.g. restored from a job database shared with a training deployment.
    module = _router_module(monkeypatch, "tt-ideogram-4")
    service = SimpleNamespace(
        get_job_metadata=lambda job_id: {"id": job_id, "job_type": job_type}
    )
    with pytest.raises(module.HTTPException) as error:
        module._job_or_404(service, "j")
    assert error.value.status_code == 404


class _FakeDevice:
    """Stands in for the scheduler and device: each request yields an image,
    or a video for Wan, after ``delay`` seconds. ``video``, if set, makes a
    video's result instead."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.requests = []
        self.error = None
        self.video = None

    def check_is_model_ready(self):
        return True

    async def process(self, request):
        self.requests.append(request)
        if request._start_event is not None:
            request._start_event.set()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if request.OUTPUT_TYPE == "video" and self.video is not None:
            return self.video(request)
        if request.OUTPUT_TYPE == "video":
            return np.zeros(
                (request.frames, request._height, request._width, 3), np.uint8
            )
        return Image.new("RGB", (request._width, request._height), "blue")


@pytest.fixture
def generations_app(monkeypatch, tmp_path):
    """(build(runner, delay) -> (client, device)), sharing one job manager."""
    from model_services.generation_service import GenerationService
    from resolver.service_resolver import service_resolver
    from security.api_key_checker import get_api_key
    from utils.job_manager import JobManager

    monkeypatch.setenv("TT_GENERATIONS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "utils.video_manager._VIDEO_OUTPUT_DIR", tmp_path.parent / "encoded"
    )
    monkeypatch.setattr(
        "utils.job_manager.get_settings",
        lambda: SimpleNamespace(
            enable_job_persistence=False,
            max_queue_size=10,
            max_jobs=10,
            job_cleanup_interval_seconds=3600,
            job_retention_seconds=_RETENTION_SECONDS,
            job_max_stuck_time_seconds=3600,
        ),
    )
    clients = []

    def build(runner: str, delay: float = 0.0):
        device = _FakeDevice(delay)
        service = object.__new__(GenerationService)
        service.scheduler = device
        service.logger = SimpleNamespace(
            error=lambda *_: None, info=lambda *_: None, warning=lambda *_: None
        )
        service._job_manager = JobManager()
        service._processManager = SimpleNamespace(
            Event=threading.Event, Namespace=SimpleNamespace
        )
        service._createStartEvent = lambda: asyncio.Event()
        service.process = device.process

        app = FastAPI()
        app.include_router(_router_module(monkeypatch, runner).router)
        app.dependency_overrides[service_resolver] = lambda: service
        app.dependency_overrides[get_api_key] = lambda: None
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        clients.append((client, service))
        return client, device

    yield build
    for client, service in clients:
        service._job_manager._cleanup_task.cancel()


async def _wait_final(client, job_id: str) -> dict:
    for _ in range(200):
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {job}")


async def test_image_job_lifecycle(generations_app, tmp_path):
    client, device = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={
            "prompt": "add a hat",
            "images": [_data_uri(1600, 900)],
            "seed": 7,
            "format": "jpeg",
        },
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert set(job) == {"id", "model", "status", "created_at"}
    assert job["model"] == "qwen/qwen-image-edit-2511"

    job = await _wait_final(client, job["id"])
    assert job["status"] == "completed"
    assert job["expires_at"] == job["completed_at"] + _RETENTION_SECONDS
    (request,) = device.requests
    assert (request._seed, request._prompt) == (7, "add a hat")
    assert request.negative_prompt == " "

    (output,) = job["outputs"]
    assert output == {
        "type": "image",
        "url": f"/v1/jobs/{job['id']}/content/0",
        "content_type": "image/jpeg",
        "seed": 7,
        "prompt": "add a hat",
        "width": 1376,
        "height": 768,
    }
    content = await client.get(output["url"])
    assert content.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(content.content)).size == (1376, 768)
    assert (await client.get(f"/v1/jobs/{job['id']}/content/1")).status_code == 404

    assert (await client.get("/v1/jobs")).json() == [job]
    assert (await client.delete(f"/v1/jobs/{job['id']}")).json() == {
        "id": job["id"],
        "deleted": True,
    }
    assert not (tmp_path / job["id"]).exists()
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404


async def test_video_job(generations_app):
    client, device = generations_app("tt-wan2.2-i2v")
    response = await client.post(
        "/v1/generations/wan-ai/wan2.2-i2v-a14b",
        json={
            "prompt": "a boat",
            "first_frame": _data_uri(900, 1600),
            "aspect_ratio": "9:16",
            "size": "480p",
        },
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert device.requests[0].negative_prompt == WAN_NEGATIVE_PROMPT
    (output,) = job["outputs"]
    assert output["type"] == "video" and output["content_type"] == "video/mp4"
    assert (output["width"], output["height"], output["fps"]) == (480, 832, 16)
    assert output["num_frames"] == 81 and output["duration"] == 81 / 16
    assert output["audio"] is None
    content = await client.get(output["url"])
    assert content.headers["content-type"] == "video/mp4" and content.content


def _sound_video(frames: int, sampling_rate: int, fps: int) -> str:
    """An MP4 with sound, not faststart, as the LTX and H3 runners write."""
    return VideoManager().export_to_mp4_with_audio(
        np.zeros((frames, 64, 64, 3), np.uint8),
        np.zeros((2, sampling_rate * frames // fps), np.float32),
        sampling_rate,
        fps=fps,
    )


async def test_video_with_sound_job(generations_app):
    import av

    client, device = generations_app("tt-ltx-2.3-distilled")
    device.video = lambda request: _sound_video(9, 16000, 25)
    response = await client.post(
        "/v1/generations/lightricks/ltx-2.3-22b-distilled",
        json={"prompt": "a violinist", "duration": 6},
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    (output,) = job["outputs"]
    assert output["audio"] == {"codec": "aac", "sample_rate": 16000, "channels": 2}
    assert output["fps"] == 25 and output["num_frames"] == 9
    content = await client.get(output["url"])
    with av.open(BytesIO(content.content)) as container:
        assert container.streams.video and container.streams.audio
    # faststart: the index comes before the media data.
    assert content.content.index(b"moov") < content.content.index(b"mdat")


async def test_invalid_media_is_422_before_queueing(generations_app):
    client, device = generations_app("tt-qwen-image-edit-2511")
    not_an_image = "data:image/png;base64," + base64.b64encode(b"nope").decode()
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri(), not_an_image]},
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "images", 1]
    assert device.requests == [] and (await client.get("/v1/jobs")).json() == []


async def test_disallowed_url_is_400_and_large_image_413(generations_app, monkeypatch):
    from utils.media_downloader import MediaDownloadPolicyError

    async def refuse(url, **kwargs):
        raise MediaDownloadPolicyError("host is not allowed")

    monkeypatch.setattr("model_services.generation_service.download_media_url", refuse)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    disallowed = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": ["https://example.com/a.png"]},
    )
    assert disallowed.status_code == 400
    assert disallowed.json()["detail"] == "host is not allowed"

    monkeypatch.setitem(MEDIA_MAX_BYTES, "image", 10)
    large = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri()]},
    )
    assert large.status_code == 413
    assert (await client.get("/v1/jobs")).json() == []


async def test_full_queue_is_429_before_media_is_fetched(generations_app, monkeypatch):
    async def fetch(url, **kwargs):
        raise AssertionError("fetched")

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    import utils.job_manager

    settings = utils.job_manager.get_settings()
    settings.max_queue_size = 0
    monkeypatch.setattr("utils.job_manager.get_settings", lambda: settings)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": ["https://example.com/a.png"]},
    )
    assert response.status_code == 429


async def test_inputs_are_fetched_concurrently(generations_app, monkeypatch):
    png = base64.b64decode(_data_uri().split(",")[1])
    fetching = 0
    both = asyncio.Event()

    async def fetch(url, **kwargs):
        nonlocal fetching
        fetching += 1
        if fetching == 2:
            both.set()
        await asyncio.wait_for(both.wait(), timeout=5)
        return png

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    client, device = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={
            "prompt": "x",
            "images": ["https://example.com/a.png", "https://example.com/b.png"],
        },
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])
    (request,) = device.requests
    # The runner reads the decoded images, not the files.
    assert len(request._images["images"]) == 2 and request._files == {}


async def test_a_refused_input_stops_the_other_fetches(generations_app, monkeypatch):
    from utils.media_downloader import MediaDownloadPolicyError

    stopped = asyncio.Event()

    async def fetch(url, **kwargs):
        if url.endswith("a.png"):
            raise MediaDownloadPolicyError("host is not allowed")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    response = await asyncio.wait_for(
        client.post(
            "/v1/generations/qwen/qwen-image-edit-2511",
            json={
                "prompt": "x",
                "images": ["https://example.com/a.png", "https://example.com/b.png"],
            },
        ),
        timeout=5,
    )
    assert response.status_code == 400
    assert stopped.is_set()


async def test_schema_errors_are_422(generations_app):
    client, _ = generations_app("tt-ideogram-4")
    response = await client.post(
        "/v1/generations/ideogram-ai/ideogram-4",
        json={"structured_prompt": _CAPTION, "size": "8K", "colour": "red"},
    )
    assert response.status_code == 422
    locs = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert {("body", "colour"), ("body", "size")} <= locs
    other = await client.post("/v1/generations/qwen/qwen-image-edit-2511", json={})
    assert other.status_code == 404


async def test_cancel_reaches_the_device_worker(generations_app):
    client, device = generations_app("tt-flux.2-klein-4b", delay=0.5)
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()
    while not device.requests:
        await asyncio.sleep(0.01)
    (request,) = device.requests
    assert not request._cancel_event.is_set()
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    # A worker skips a request whose cancel event is set, or stops early.
    assert request._cancel_event.is_set()
    assert (await _wait_final(client, job["id"]))["status"] == "cancelled"


async def test_cancel_and_delete_rules(generations_app):
    client, _ = generations_app("tt-flux.2-klein-4b", delay=0.5)
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()

    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 409
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    cancelled = await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in ("cancelling", "cancelled")
    job = await _wait_final(client, job["id"])
    assert job["status"] == "cancelled" and "outputs" not in job
    again = await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert again.status_code == 409
    assert (await client.post("/v1/jobs/nope/cancel")).status_code == 404
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200


async def test_ideogram_gets_the_serialized_caption(generations_app):
    client, device = generations_app("tt-ideogram-4")
    response = await client.post(
        "/v1/generations/ideogram-ai/ideogram-4",
        json={"structured_prompt": _CAPTION, "aspect_ratio": "16:9", "size": "1K"},
    )
    job = await _wait_final(client, response.json()["id"])
    caption = (
        '{"compositional_deconstruction":{"background":"a brick wall",'
        '"elements":[{"type":"text","bbox":[10,20,30,40],"text":"OPEN"}]}}'
    )
    assert device.requests[0]._prompt == caption
    (output,) = job["outputs"]
    assert output["prompt"] == caption
    assert (output["width"], output["height"]) == (1280, 720)


async def test_failed_job_reports_error(generations_app, tmp_path):
    client, device = generations_app("tt-flux.2-klein-4b")
    device.error = RuntimeError("device fell over")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "failed" and "outputs" not in job
    assert job["error"] == {"code": "processing_error", "message": "device fell over"}
    assert job["expires_at"] > job["completed_at"]
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    assert list(tmp_path.iterdir()) == []


async def test_a_generations_dir_in_any_spelling_works(
    generations_app, monkeypatch, tmp_path
):
    monkeypatch.setenv("TT_GENERATIONS_DIR", f"{tmp_path}//./")
    client, _ = generations_app("tt-flux.2-klein-4b")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    assert (tmp_path / job["id"] / "outputs.json").exists()


async def test_what_a_runner_leaves_in_a_failed_jobs_directory_goes_with_it(
    generations_app, tmp_path
):
    # As LTX's runner writes its video into the job's directory, here before
    # the job fails.
    client, device = generations_app("tt-ltx-2.3-distilled")

    def video(request):
        (tmp_path / request._task_id).mkdir()
        (tmp_path / request._task_id / "ltx.mp4").write_bytes(b"mp4")
        raise RuntimeError("device fell over")

    device.video = video
    response = await client.post(
        "/v1/generations/lightricks/ltx-2.3-22b-distilled",
        json={"prompt": "a violinist"},
    )
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "failed"
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(("size", "status"), [(12, 422), (20, 422), (9, 202)])
async def test_oversized_images_are_422(generations_app, monkeypatch, size, status):
    # Over the limit (12x12) and over twice it, where PIL itself refuses (20x20).
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    client, _ = generations_app("tt-qwen-image-edit-2511")
    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "x", "images": [_data_uri(size, size)]},
    )
    assert response.status_code == status, response.text
    if status == 422:
        assert "more than 100 pixels" in response.json()["detail"][0]["msg"]


def test_generation_job_type_and_manifest_roundtrip(tmp_path):
    from model_services.generation_service import MANIFEST, read_manifest

    (tmp_path / MANIFEST).write_text(json.dumps([{"file": "0.png"}]))
    assert read_manifest(str(tmp_path)) == [{"file": "0.png"}]
    assert JobTypes.GENERATION.value == "generation"


async def test_an_earlier_output_is_not_an_input(generations_app):
    client, _ = generations_app("tt-qwen-image-edit-2511")
    first = (
        await client.post(
            "/v1/generations/qwen/qwen-image-edit-2511",
            json={"prompt": "a hat", "images": [_data_uri(1600, 900)]},
        )
    ).json()
    (output,) = (await _wait_final(client, first["id"]))["outputs"]

    response = await client.post(
        "/v1/generations/qwen/qwen-image-edit-2511",
        json={"prompt": "a scarf", "images": [output["url"]]},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "images", 0]


def test_stored_request_digests_data_uris():
    from model_services.generation_service import _request_parameters

    image = _data_uri()
    request = QwenImageEdit2511Request(
        prompt="x", images=[image, "DATA" + image[4:], "https://x/a.png"]
    )
    digest, upper_digest, url = _request_parameters(request)["images"]
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert upper_digest.startswith("sha256:")
    assert url == "https://x/a.png"


def test_images_are_turned_upright():
    from utils.media_probe import decode_image

    # Stored landscape, shown portrait: EXIF orientation 6 is "rotate 90° CW".
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (30, 40)


async def test_lost_outputs_are_404(generations_app, tmp_path):
    client, _ = generations_app("tt-flux.2-klein-4b")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux.2-klein-4b", json={"prompt": "a cat"}
    )
    job = await _wait_final(client, response.json()["id"])
    for path in (tmp_path / job["id"]).iterdir():
        path.unlink()
    (tmp_path / job["id"]).rmdir()
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 404
    assert (await client.get("/v1/jobs")).json() == []


async def test_outputs_of_a_job_cancelled_while_writing_go_with_the_job(
    generations_app, monkeypatch, tmp_path
):
    # A running job is cancelled cooperatively: it finishes, becomes
    # `cancelled` without outputs, and its files are deleted with the job.
    from model_services.generation_service import GenerationService

    writing, release = threading.Event(), threading.Event()
    write_outputs = GenerationService._write_outputs

    def slow_write(self, request, results):
        writing.set()
        release.wait(5)
        return write_outputs(self, request, results)

    monkeypatch.setattr(GenerationService, "_write_outputs", slow_write)
    client, _ = generations_app("tt-flux.2-klein-4b")
    job = (
        await client.post(
            "/v1/generations/black-forest-labs/flux.2-klein-4b",
            json={"prompt": "a cat"},
        )
    ).json()
    await asyncio.to_thread(writing.wait, 5)
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    release.set()
    job = await _wait_final(client, job["id"])
    assert job["status"] == "cancelled" and "outputs" not in job
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 200
    assert not (tmp_path / job["id"]).exists()


# --- MiniMax-H3 ------------------------------------------------------------------


def _clip_data_uri(seconds: float, video: bool = True, channels: int = 2) -> str:
    """A data URI of an MP4 with sound (``video``), or of a WAV."""
    if video:
        frames = round(seconds * 25)
        path = VideoManager().export_to_mp4_with_audio(
            np.zeros((frames, 64, 64, 3), np.uint8),
            np.zeros((channels, 16000 * frames // 25), np.float32),
            16000,
            fps=25,
        )
        data = pathlib.Path(path).read_bytes()
        pathlib.Path(path).unlink()
        return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * channels * round(16000 * seconds))
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode()


def _probe(duration_s=5.0, size=(64, 64), fps=25.0, audio_channels=2):
    return ReferenceMediaProbe(
        duration_s=duration_s, size=size, fps=fps, audio_channels=audio_channels
    )


@pytest.mark.parametrize(
    ("duration", "frames"), [(4, 107), (5, 124), (6, 158), (15, 362)]
)
def test_h3_frames_are_the_next_17n_plus_5(duration, frames):
    assert MiniMaxH3FL2VARequest(prompt="x", duration=duration).frames == frames


@pytest.mark.parametrize(
    ("fields", "images", "expected"),
    [
        ({}, {}, (1344, 768)),
        ({"aspect_ratio": "21:9"}, {}, (1536, 672)),
        ({"aspect_ratio": "9:16"}, {}, (768, 1344)),
        ({"aspect_ratio": "1:1"}, {}, (768, 768)),
        # A keyframe sets the shape, the first frame's if there are two.
        ({"last_frame": "https://x/b.png"}, {"last_frame": [(600, 800)]}, (768, 1024)),
        (
            {"first_frame": "https://x/a.png", "last_frame": "https://x/b.png"},
            {"first_frame": [(600, 800)], "last_frame": [(800, 600)]},
            (768, 1024),
        ),
    ],
)
def test_h3_output_size(fields, images, expected):
    request = _with_images(MiniMaxH3FL2VARequest(prompt="x", **fields), **images)
    assert request.output_size() == expected


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"aspect_ratio": "adaptive"}, "needs a first_frame or last_frame"),
        (
            {"first_frame": "https://x/a.png", "aspect_ratio": "16:9"},
            "must be adaptive",
        ),
        ({"size": "720p"}, "'768p'"),
        ({"width": 832, "height": 480}, "extra"),
        ({"duration": 3}, "greater than or equal to 4"),
        ({"duration": 5.5}, "integer"),
        ({"aspect_ratio": "2:1"}, "'21:9'"),
        ({"aspect_ratio": "16:9:1"}, "'21:9'"),
        ({"negative_prompt": "blur"}, "extra"),
    ],
)
def test_h3_rejects(fields, message):
    with pytest.raises(ValidationError, match=message):
        MiniMaxH3FL2VARequest(prompt="x", **fields)


def test_h3_aspect_ratio_defaults_to_adaptive_with_a_frame():
    assert MiniMaxH3FL2VARequest(prompt="x").aspect_ratio == "16:9"
    with_frame = MiniMaxH3FL2VARequest(prompt="x", last_frame="https://x/a.png")
    assert with_frame.aspect_ratio == "adaptive"


@pytest.mark.parametrize("ratio", ["16x9", "16/9", " 16 : 9 ", "016:9"])
def test_h3_ratio_spellings_of_the_earlier_api(ratio):
    assert MiniMaxH3FL2VARequest(prompt="x", aspect_ratio=ratio).aspect_ratio == "16:9"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_h3_prompt_must_not_be_blank(prompt):
    with pytest.raises(ValidationError):
        MiniMaxH3FL2VARequest(prompt=prompt)


@pytest.mark.parametrize(
    ("images", "errors"),
    [
        ({"first_frame": [(256, 1024)], "last_frame": [(5760, 1440)]}, []),
        ({"first_frame": [(255, 300)]}, [("first_frame", "256 to 5760")]),
        ({"last_frame": [(5761, 2000)]}, [("last_frame", "256 to 5760")]),
        ({"first_frame": [(300, 1500)]}, [("first_frame", "1:4 to 4:1")]),
    ],
)
def test_h3_keyframe_limits(images, errors):
    request = _with_images(MiniMaxH3FL2VARequest(prompt="x"), **images)
    got = request.media_errors()
    assert [(name, index) for name, index, _ in got] == [(n, 0) for n, _ in errors]
    for (_, _, message), (_, expected) in zip(got, errors):
        assert expected in message


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"audios": ["https://x/a.wav"]}, "at least one reference image or video"),
        ({}, "at least one reference image or video"),
        (
            {
                "images": ["https://x/a.png"] * 9,
                "videos": ["https://x/a.mp4"] * 3,
                "audios": ["https://x/a.wav"],
            },
            "at most 12",
        ),
        ({"images": ["https://x/a.png"] * 10}, "at most 9"),
        ({"images": ["https://x/a.png"], "videos": ["data:image/png;base64,"]}, "video"),
        ({"images": ["https://x/a.png"], "audios": ["/v1/jobs/j/content/0"]}, "audio"),
    ],
)
def test_h3_ref2va_rejects(fields, message):
    with pytest.raises(ValidationError, match=message):
        MiniMaxH3Ref2VARequest(prompt="x", **fields)


@pytest.mark.parametrize(
    ("probes", "errors"),
    [
        (
            {
                "videos": [_probe(), _probe(10.0, audio_channels=None)],
                "audios": [_probe(10.0, None, None, 1)],
            },
            [],
        ),
        ({"videos": [_probe(1.5)]}, [("videos", 0, "2 to 15 s")]),
        (
            {"audios": [_probe(16.0, None, None)]},
            [("audios", 0, "2 to 15 s"), ("audios", 0, "16 s of audio together")],
        ),
        # A video's soundtrack counts toward the audio limits.
        (
            {"videos": [_probe(5.0)], "audios": [_probe(11.0, None, None)]},
            [("audios", 0, "16 s of audio together")],
        ),
        (
            {"videos": [_probe(3.0)] * 3, "audios": [_probe(3.0, None, None)]},
            [("audios", 0, "4 audio references")],
        ),
        ({"videos": [_probe(8.0), _probe(8.0)]}, [("videos", 1, "16 s together")]),
        ({"videos": [_probe(size=None)]}, [("videos", 0, "no video stream")]),
        ({"videos": [_probe(size=(64, 300))]}, [("videos", 0, "1:4 to 4:1")]),
        ({"videos": [_probe(fps=None)]}, [("videos", 0, "no frame rate")]),
        ({"audios": [_probe(audio_channels=None)]}, [("audios", 0, "no audio stream")]),
        ({"audios": [_probe(audio_channels=6)]}, [("audios", 0, "mono or stereo")]),
    ],
)
def test_h3_reference_clip_limits(probes, errors):
    request = MiniMaxH3Ref2VARequest(prompt="x", images=["https://x/a.png"])
    request._images = {"images": [Image.new("RGB", (64, 64))]}
    request._probes = probes
    got = request.media_errors()
    assert [(name, index) for name, index, _ in got] == [(n, i) for n, i, _ in errors]
    for (_, _, message), (_, _, expected) in zip(got, errors):
        assert expected in message


_MP4 = "mov,mp4,m4a,3gp,3g2,mj2"


@pytest.mark.parametrize(
    ("kind", "probe", "error"),
    [
        ("video", (_MP4, "h264", "aac"), None),
        ("video", (_MP4, "hevc", "mp3"), None),
        ("video", (_MP4, "h264", None), None),
        ("video", ("matroska,webm", "vp9", "opus"), "must be MP4 or MOV"),
        ("video", (_MP4, "mpeg4", None), "must be H.264 or H.265"),
        ("video", (_MP4, "h264", "opus"), "must be AAC or MP3"),
        ("audio", ("wav", None, "pcm_s16le"), None),
        ("audio", ("mp3", None, "mp3"), None),
        ("audio", ("ogg", None, "vorbis"), "must be WAV or MP3"),
        ("audio", (_MP4, None, "aac"), "must be WAV or MP3"),
    ],
)
def test_clip_formats(kind, probe, error):
    container, video_codec, audio_codec = probe
    message = media_format_error(
        kind,
        ReferenceMediaProbe(5.0, None, None, None, container, video_codec, audio_codec),
    )
    assert message is None if error is None else error in message


async def test_a_clip_in_another_format_is_422(generations_app):
    client, device = generations_app("tt-minimax-h3-ref2va")
    audio = _clip_data_uri(3).replace("data:video/", "data:audio/", 1)  # an MP4
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={"prompt": "x", "images": [_data_uri()], "audios": [audio]},
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "audios", 0]
    assert "must be WAV or MP3" in error["msg"]
    assert device.requests == []


async def test_h3_keyframes_job(generations_app):
    client, device = generations_app("tt-minimax-h3-fl2va")
    device.video = lambda request: _sound_video(request.frames, 48000, 24)
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-fl2va",
        json={"prompt": "a lighthouse", "first_frame": _data_uri(600, 800)},
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    (request,) = device.requests
    assert (request._width, request._height) == (768, 1024)
    # The runner decodes the keyframe: only its file goes to the device.
    assert request._images == {}
    assert Image.open(BytesIO(request._files["first_frame"][0])).size == (600, 800)
    (output,) = job["outputs"]
    assert output["audio"] == {"codec": "aac", "sample_rate": 48000, "channels": 2}
    assert output["fps"] == 24 and output["num_frames"] == 124
    assert output["duration"] == 124 / 24


async def test_h3_keyframe_out_of_limits_is_422(generations_app):
    client, device = generations_app("tt-minimax-h3-fl2va-lightx2v")
    response = await client.post(
        "/v1/generations/tenstorrent/minimax-h3-fl2va-lightx2v",
        json={"prompt": "x", "last_frame": _data_uri(200, 300)},
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "last_frame"]
    assert "256 to 5760" in error["msg"]
    assert device.requests == []


async def test_h3_references_are_loaded_and_checked(generations_app):
    client, device = generations_app("tt-minimax-h3-ref2va")
    video, audio = _clip_data_uri(3), _clip_data_uri(2.5, video=False, channels=1)
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={
            "prompt": "she sings",
            "images": [_data_uri(64, 96)],
            "videos": [video],
            "audios": [audio],
        },
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    (request,) = device.requests
    assert request._images == {}
    assert Image.open(BytesIO(request._files["images"][0])).size == (64, 96)
    assert request._files["videos"] == [base64.b64decode(video.split(",")[1])]
    assert request._files["audios"] == [base64.b64decode(audio.split(",")[1])]
    (video_probe,) = request._probes["videos"]
    assert video_probe.size == (64, 64) and video_probe.fps == 25
    assert request._probes["audios"][0].audio_channels == 1

    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={
            "prompt": "x",
            "images": [_data_uri()],
            "videos": [_clip_data_uri(1), "data:video/mp4;base64,bm9wZQ=="],
        },
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert [error["loc"] for error in errors] == [["body", "videos", 1]]
    assert "not a readable video or audio file" in errors[0]["msg"]


async def test_a_too_large_clip_is_413_naming_its_kind(generations_app, monkeypatch):
    monkeypatch.setitem(MEDIA_MAX_BYTES, "video", 10)
    client, _ = generations_app("tt-minimax-h3-ref2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={"prompt": "x", "images": [_data_uri()], "videos": [_clip_data_uri(3)]},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "A video input is over 10 bytes"


def test_the_service_downloads_no_weights(monkeypatch):
    # The pipelines load their own weights.
    from model_services import base_service
    from model_services.generation_service import GenerationService

    downloads = []
    monkeypatch.setattr(base_service, "get_scheduler", lambda: None)
    monkeypatch.setattr(base_service.settings, "download_weights_from_service", True)
    monkeypatch.setattr(
        base_service,
        "HuggingFaceUtils",
        lambda: SimpleNamespace(download_weights=lambda: downloads.append(1)),
    )
    base_service.BaseService.__init__(object.__new__(GenerationService))
    assert downloads == []
