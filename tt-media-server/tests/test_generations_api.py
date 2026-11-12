# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the generations API: each deployment's OpenAPI document,
checked against docs/api/openapi, and submit → job → content → delete
through the router with a fake device. Runners and tt_dit run on hardware, not
here."""

import asyncio
import base64
import importlib
import importlib.util
import json
import pathlib
import threading
import time
import wave
from io import BytesIO
from types import SimpleNamespace

import httpx
import numpy as np
import domain.wan22_i2v_a14b_request as wan_request
import pytest
import yaml
from config.constants import JobTypes
from domain.flux3_image_klein_request import Flux3ImageKleinRequest
from domain.generation_request import MEDIA_MAX_BYTES
from domain.ltx23_distilled_request import Ltx23DistilledRequest
from domain.generation_models import GENERATION_MODELS
from domain.minimax_h3_request import MiniMaxH3FL2VARequest, MiniMaxH3Ref2VARequest
from domain.wan22_i2v_a14b_request import WAN_NEGATIVE_PROMPT, Wan22i2vA14bRequest
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError
from utils.media_probe import ReferenceMediaProbe, media_format_error
from utils.video_manager import VideoManager

from tests.fake_tt_metal_policy import H3_POLICY, WAN_POLICY

requires_tt_metal_policy = pytest.mark.requires_tt_metal_policy


def _data_uri(width: int = 64, height: int = 64, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format=image_format)
    return f"data:image/{image_format.lower()};base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _with_image_sizes(request, **sizes):
    request._image_sizes = sizes
    return request


# --- The spec ----------------------------------------------------------------------


def _load_dump_script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "dump_generations_openapi", path / "dump_generations_openapi.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dump_script = _load_dump_script()


@pytest.fixture(scope="module")
def served_generations_api():
    """Each deployment's generations API, built in parallel."""
    return _dump_script.served_generations_api_by_runner(list(GENERATION_MODELS))


def _spec_param(runner, model):
    # Their documents quote tt-metal's policy values.
    quotes_tt_metal = issubclass(
        model.request,
        (Wan22i2vA14bRequest, MiniMaxH3FL2VARequest, MiniMaxH3Ref2VARequest),
    )
    marks = [requires_tt_metal_policy] if quotes_tt_metal else []
    return pytest.param(runner, model, id=model.id, marks=marks)


# After an intended API change: python scripts/dump_generations_openapi.py
@pytest.mark.parametrize(
    ("runner", "model"),
    [_spec_param(runner, model) for runner, model in GENERATION_MODELS.items()],
)
def test_openapi_is_the_spec(served_generations_api, runner, model):
    expected = yaml.safe_load(_dump_script.spec_path(model.id).read_text())
    assert served_generations_api[runner] == expected


def test_every_spec_is_served():
    expected = {
        _dump_script.spec_path(model.id) for model in GENERATION_MODELS.values()
    }
    assert _dump_script.spec_files() == expected


def test_unknown_field_rejected():
    with pytest.raises(ValidationError, match="extra"):
        Ltx23DistilledRequest(prompt="a cat", steps=4)


@pytest.mark.parametrize(
    "media",
    ["ftp://example.com/a.png", "iVBORw0KGgo=", "data:text/plain;base64,aGk="],
)
def test_media_must_be_a_url_or_image_data_uri(media):
    with pytest.raises(ValidationError, match="https:// URL or a base64 data URI"):
        Wan22i2vA14bRequest(
            prompt="a boat", first_frame=media, aspect_ratio="16:9", size="480p"
        )


@requires_tt_metal_policy
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
    module = _router_module(monkeypatch, "tt-wan2.2-i2v")
    app = FastAPI()
    app.include_router(module.router)
    paths = set(app.openapi()["paths"])
    assert paths == {
        "/v1/generations/wan-ai/wan2.2-i2v-a14b",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/content/{index}",
        "/v1/jobs/{job_id}/cancel",
    }


@pytest.mark.parametrize("job_type", ["training", "video"])
def test_other_services_jobs_are_404(monkeypatch, job_type):
    # E.g. restored from a job database shared with a training deployment.
    module = _router_module(monkeypatch, "tt-wan2.2-i2v")
    service = SimpleNamespace(
        get_job_metadata=lambda job_id: {"id": job_id, "job_type": job_type}
    )
    with pytest.raises(module.HTTPException) as error:
        module._job_or_404(service, "j")
    assert error.value.status_code == 404


class _FakeDevice:
    """Stands in for the scheduler and device: each request yields an image,
    or a video for a video model, after ``delay`` seconds. ``video``, if set,
    makes a video's result instead. A job stays queued for ``queued`` seconds."""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.queued = 0.0
        self.requests = []
        self.error = None
        self.video = None

    def check_is_model_ready(self):
        return True

    async def process(self, request):
        self.requests.append(request)
        await asyncio.sleep(self.queued)
        if request._start_event is not None:
            request._start_event.set()
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if request.OUTPUT_TYPE == "video" and self.video is not None:
            return self.video(request)
        if request.OUTPUT_TYPE == "video":
            # The runner encodes its MP4 and returns the path.
            return VideoManager().export_to_mp4(
                np.zeros(
                    (request.frames, request._height, request._width, 3), np.uint8
                ),
                fps=request.fps,
            )
        return Image.new("RGB", (request._width, request._height), "blue")


@pytest.fixture
async def generations_app(monkeypatch, tmp_path):
    """(build(runner, delay, auth) -> (client, device)), sharing one job
    manager. The API key is checked only with ``auth``."""
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

    def build(runner: str, delay: float = 0.0, auth: bool = False):
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
        if not auth:
            app.dependency_overrides[get_api_key] = lambda: None
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        clients.append((client, service))
        return client, device

    yield build
    # On the test's event loop, which older pytest-asyncio closes before the
    # teardown of a sync fixture.
    for client, service in clients:
        await service._job_manager.shutdown()
        await client.aclose()


async def _wait_status(client, job_id: str, *statuses: str) -> dict:
    """The job, once its status is one of ``statuses``; fails after 10 s."""
    deadline = time.monotonic() + 10
    while True:
        job = (await client.get(f"/v1/jobs/{job_id}")).json()
        if job["status"] in statuses:
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"job {job_id} never became {statuses}: {job}")
        await asyncio.sleep(0.01)


async def _wait_final(client, job_id: str) -> dict:
    return await _wait_status(client, job_id, "completed", "failed", "cancelled")


def _sound_video(frames: int, sampling_rate: int, fps: int) -> str:
    """An MP4 with sound, as the LTX and H3 runners return."""
    return VideoManager().export_to_mp4_with_audio(
        np.zeros((frames, 64, 64, 3), np.uint8),
        np.zeros((2, sampling_rate * frames // fps), np.float32),
        sampling_rate,
        fps=fps,
    )


# A video model's endpoint and a request it takes without media.
_LTX = ("tt-ltx-2.3-distilled", "/v1/generations/lightricks/ltx-2.3-22b-distilled")


def _ltx_app(generations_app, delay: float = 0.0):
    """LTX, whose fake device returns a short clip instead of the full shape."""
    client, device = generations_app(_LTX[0], delay=delay)
    device.video = lambda request: _sound_video(9, 16000, 25)
    return client, device


def _wan_body(**fields) -> dict:
    return {"prompt": "x", "aspect_ratio": "16:9", "size": "480p", **fields}


def _wan_app(generations_app):
    client, device = generations_app("tt-wan2.2-i2v")
    device.video = lambda request: _sound_video(9, 16000, 25)
    return client, device


_WAN = "/v1/generations/wan-ai/wan2.2-i2v-a14b"


async def test_job_lifecycle(generations_app, tmp_path):
    client, device = _ltx_app(generations_app)
    response = await client.post(_LTX[1], json={"prompt": "a violinist", "seed": 7})
    assert response.status_code == 202, response.text
    job = response.json()
    assert set(job) == {"id", "model", "status", "created_at"}
    assert job["model"] == "lightricks/ltx-2.3-22b-distilled"

    job = await _wait_final(client, job["id"])
    assert job["status"] == "completed"
    assert job["expires_at"] == job["completed_at"] + _RETENTION_SECONDS
    (request,) = device.requests
    assert (request._seed, request._prompt) == (7, "a violinist")

    (output,) = job["outputs"]
    # The fake device's clip: 9 frames of 64x64 at 25 fps, stereo at 16 kHz.
    assert output == {
        "type": "video",
        "url": f"v1/jobs/{job['id']}/content/0",
        "content_type": "video/mp4",
        "seed": 7,
        "prompt": "a violinist",
        "width": 64,
        "height": 64,
        "fps": 25,
        "num_frames": 9,
        "duration": 9 / 25,
        "audio": {"codec": "aac", "sample_rate": 16000, "channels": 2},
    }
    content = await client.get(output["url"])
    assert content.headers["content-type"] == "video/mp4" and content.content
    assert (await client.get(f"/v1/jobs/{job['id']}/content/1")).status_code == 404

    assert (await client.get("/v1/jobs")).json() == [job]
    assert (await client.delete(f"/v1/jobs/{job['id']}")).json() == {
        "id": job["id"],
        "deleted": True,
    }
    assert not (tmp_path / job["id"]).exists()
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404


@pytest.fixture
def wan_mesh(monkeypatch):
    """Serve Wan as on a Blackhole mesh of the given shape."""

    wan_policy = importlib.import_module(WAN_POLICY)

    def serve_on(mesh_shape):
        monkeypatch.setattr(wan_policy, "_is_blackhole", lambda: True)
        monkeypatch.setattr(
            wan_request,
            "get_settings",
            lambda: SimpleNamespace(device_mesh_shape=mesh_shape),
        )

    return serve_on


@requires_tt_metal_policy
@pytest.mark.parametrize(
    ("size", "aspect_ratio", "status"),
    [("480p", "16:9", 202), ("480p", "9:16", 202), ("720p", "16:9", 422)],
)
async def test_wan_takes_only_the_sizes_the_mesh_serves(
    generations_app, wan_mesh, size, aspect_ratio, status
):
    wan_mesh((2, 2))
    client, device = _wan_app(generations_app)
    response = await client.post(
        _WAN,
        json=_wan_body(first_frame=_data_uri(), size=size, aspect_ratio=aspect_ratio),
    )
    assert response.status_code == status, response.text
    if status == 422:
        (error,) = response.json()["detail"]
        assert error["loc"] == ["body", "size"]
        assert error["msg"] == (
            "unsupported size 1280x720; this deployment serves [(832, 480), (480, 832)]"
        )
        assert device.requests == [] and (await client.get("/v1/jobs")).json() == []


@requires_tt_metal_policy
async def test_video_job(generations_app, wan_mesh):
    wan_mesh((4, 8))
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
    import av

    # H.264, yuv420p, faststart; the profile depends on the encoder preset.
    with av.open(BytesIO(content.content)) as container:
        assert not container.streams.audio
        (stream,) = container.streams.video
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.pix_fmt == "yuv420p"
    assert content.content.index(b"moov") < content.content.index(b"mdat")


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
    client, device = _wan_app(generations_app)
    not_an_image = "data:image/png;base64," + base64.b64encode(b"nope").decode()
    response = await client.post(
        _WAN, json=_wan_body(first_frame=_data_uri(), last_frame=not_an_image)
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "last_frame"]
    assert device.requests == [] and (await client.get("/v1/jobs")).json() == []


async def test_disallowed_url_is_400_and_large_image_413(generations_app, monkeypatch):
    from utils.media_downloader import MediaDownloadPolicyError

    async def refuse(url, **kwargs):
        raise MediaDownloadPolicyError("host is not allowed")

    monkeypatch.setattr("model_services.generation_service.download_media_url", refuse)
    client, _ = _wan_app(generations_app)
    disallowed = await client.post(
        _WAN, json=_wan_body(first_frame="https://example.com/a.png")
    )
    assert disallowed.status_code == 400
    assert disallowed.json()["detail"] == "host is not allowed"

    monkeypatch.setitem(MEDIA_MAX_BYTES, "image", 10)
    large = await client.post(_WAN, json=_wan_body(first_frame=_data_uri()))
    assert large.status_code == 413
    assert (await client.get("/v1/jobs")).json() == []


async def test_an_http_url_is_400(generations_app, monkeypatch):
    from config.settings import settings

    # Even where any public host may be fetched: the spec counts a URL that
    # isn't https as against the fetch policy.
    monkeypatch.setattr(settings, "media_url_allow_public_hosts", True)
    client, device = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="http://example.com/a.png")
    )
    assert response.status_code == 400
    assert "must be https" in response.json()["detail"]
    assert device.requests == []


@pytest.mark.parametrize(
    ("error", "message", "status"),
    [
        ("MediaDownloadFetchError", "Media URL returned HTTP 403", 422),
        ("MediaDownloadTooLargeError", "Remote media exceeds the cap", 413),
    ],
)
async def test_a_failed_or_oversized_fetch(
    generations_app, monkeypatch, error, message, status
):
    import utils.media_downloader

    async def fetch(url, **kwargs):
        raise getattr(utils.media_downloader, error)(message)

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    client, device = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="https://example.com/a.png")
    )
    assert response.status_code == status, response.text
    if status == 422:
        (detail,) = response.json()["detail"]
        assert detail["loc"] == ["body", "first_frame"]
        assert "HTTP 403" in detail["msg"]
    assert device.requests == []


async def test_an_oversized_fetch_names_the_deployments_url_cap(
    generations_app, monkeypatch
):
    import utils.media_downloader
    from model_services import generation_service

    async def fetch(url, **kwargs):
        raise utils.media_downloader.MediaDownloadTooLargeError("too big")

    monkeypatch.setattr(generation_service, "download_media_url", fetch)
    monkeypatch.setattr(generation_service.settings, "media_url_max_bytes", 1000)
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="https://example.com/a.png")
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "An image input is over 1000 bytes"


def _encoded(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "image",
    [
        _data_uri(image_format="GIF"),
        _data_uri(image_format="BMP"),
        # TIFF bytes labelled as PNG: the bytes decide.
        "data:image/png;base64," + base64.b64encode(_encoded("TIFF")).decode(),
    ],
)
async def test_images_must_be_png_jpeg_or_webp(generations_app, image):
    client, device = _wan_app(generations_app)
    response = await client.post(_WAN, json=_wan_body(first_frame=image))
    assert response.status_code == 422, response.text
    (detail,) = response.json()["detail"]
    assert detail["loc"] == ["body", "first_frame"]
    assert "must be PNG, JPEG or WebP" in detail["msg"]
    assert device.requests == []


def _mpo_data_uri() -> str:
    """A JPEG holding two pictures, as phones write a photo with a depth map."""
    buffer = BytesIO()
    first, second = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64))
    first.save(buffer, format="MPO", save_all=True, append_images=[second])
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(
    "image",
    [_data_uri(image_format=f) for f in ("PNG", "JPEG", "WEBP")] + [_mpo_data_uri()],
    ids=["png", "jpeg", "webp", "jpeg with two pictures"],
)
async def test_png_jpeg_and_webp_images_are_accepted(generations_app, image):
    client, _ = _wan_app(generations_app)
    response = await client.post(_WAN, json=_wan_body(first_frame=image))
    assert response.status_code == 202, response.text


@pytest.mark.parametrize("payload", ["!!!", ""], ids=["not base64", "empty"])
async def test_a_bad_data_uri_payload_is_422(generations_app, payload):
    client, device = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="data:image/png;base64," + payload)
    )
    assert response.status_code == 422, response.text
    (detail,) = response.json()["detail"]
    assert detail["loc"] == ["body", "first_frame"]
    assert device.requests == []


@pytest.mark.parametrize(
    "output_format, options",
    [("jpeg", {"quality": 95, "subsampling": 0}), ("webp", {"quality": 90})],
)
def test_lossy_outputs_are_saved_at_high_quality(tmp_path, output_format, options):
    from model_services.generation_service import GenerationService

    pixels = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    request = SimpleNamespace(
        OUTPUT_TYPE="image", format=output_format, _seed=0, _prompt="x"
    )
    output = GenerationService._write_output(tmp_path, 0, request, image)
    expected = BytesIO()
    image.save(expected, format=output_format.upper(), **options)
    assert (tmp_path / output["file"]).read_bytes() == expected.getvalue()


async def test_full_queue_is_429_before_media_is_fetched(generations_app, monkeypatch):
    async def fetch(url, **kwargs):
        raise AssertionError("fetched")

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    import utils.job_manager

    settings = utils.job_manager.get_settings()
    settings.max_queue_size = 0
    monkeypatch.setattr("utils.job_manager.get_settings", lambda: settings)
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="https://example.com/a.png")
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
    client, device = _wan_app(generations_app)
    response = await client.post(
        _WAN,
        json=_wan_body(
            first_frame="https://example.com/a.png",
            last_frame="https://example.com/b.png",
        ),
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])
    (request,) = device.requests
    # The runner decodes the frames: only their files and sizes go to the device.
    assert request._images == {}
    assert request._image_sizes == {"first_frame": [(64, 64)], "last_frame": [(64, 64)]}
    assert set(request._files) == {"first_frame", "last_frame"}


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
    client, _ = _wan_app(generations_app)
    response = await asyncio.wait_for(
        client.post(
            _WAN,
            json=_wan_body(
                first_frame="https://example.com/a.png",
                last_frame="https://example.com/b.png",
            ),
        ),
        timeout=5,
    )
    assert response.status_code == 400
    assert stopped.is_set()


async def test_schema_errors_are_422(generations_app):
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame=_data_uri(), size="8K", colour="red")
    )
    assert response.status_code == 422
    locs = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert {("body", "colour"), ("body", "size")} <= locs
    other = await client.post(_LTX[1], json={})
    assert other.status_code == 404


async def test_cancel_reaches_the_device_worker(generations_app):
    client, device = _ltx_app(generations_app, delay=0.5)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    while not device.requests:
        await asyncio.sleep(0.01)
    (request,) = device.requests
    assert not request._cancel_event.is_set()
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    # A worker skips a request whose cancel event is set, or stops early.
    assert request._cancel_event.is_set()
    assert (await _wait_final(client, job["id"]))["status"] == "cancelled"


async def test_cancel_and_delete_rules(generations_app):
    client, _ = _ltx_app(generations_app, delay=0.5)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()

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


async def test_a_job_is_queued_then_in_progress(generations_app):
    client, device = _ltx_app(generations_app, delay=0.5)
    device.queued = 0.2
    response = await client.post(_LTX[1], json={"prompt": "a cat"})
    assert response.json()["status"] == "queued"
    job = await _wait_status(client, response.json()["id"], "in_progress")
    assert "completed_at" not in job and "expires_at" not in job
    assert (await _wait_final(client, job["id"]))["status"] == "completed"


async def test_cancelling_a_queued_job_cancels_it_at_once(generations_app):
    client, device = _ltx_app(generations_app)
    device.queued = 5.0
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    cancelled = (await client.post(f"/v1/jobs/{job['id']}/cancel")).json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["expires_at"] == cancelled["completed_at"] + _RETENTION_SECONDS
    assert await _wait_final(client, job["id"]) == cancelled


# The job manager polls a job's start every 0.5 s, so a job that runs for 2 s
# is seen in_progress well before it ends.


async def test_cancelling_a_running_job(generations_app):
    client, _ = _ltx_app(generations_app, delay=2.0)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    await _wait_status(client, job["id"], "in_progress")
    cancelling = await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert cancelling.status_code == 200
    assert cancelling.json()["status"] == "cancelling"
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 409
    job = await _wait_final(client, job["id"])
    assert job["status"] == "cancelled"
    assert job["expires_at"] == job["completed_at"] + _RETENTION_SECONDS


async def test_the_list_holds_jobs_of_every_status(generations_app):
    client, device = _ltx_app(generations_app, delay=2.0)
    first = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    await _wait_status(client, first["id"], "in_progress")
    # The fake device holds the second job queued, past the start poll.
    device.queued = 5.0
    second = (await client.post(_LTX[1], json={"prompt": "a dog"})).json()
    statuses = {
        job["id"]: job["status"] for job in (await client.get("/v1/jobs")).json()
    }
    assert statuses == {first["id"]: "in_progress", second["id"]: "queued"}
    await client.post(f"/v1/jobs/{second['id']}/cancel")
    await _wait_final(client, first["id"])
    statuses = {
        job["id"]: job["status"] for job in (await client.get("/v1/jobs")).json()
    }
    assert statuses == {first["id"]: "completed", second["id"]: "cancelled"}


async def test_an_expired_job_is_gone(generations_app, monkeypatch):
    from config.settings import settings

    client, _ = _ltx_app(generations_app)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    job = await _wait_final(client, job["id"])
    assert job["status"] == "completed"
    # Past expires_at, before the retention sweep has run.
    monkeypatch.setattr(settings, "job_retention_seconds", -1)
    assert (await client.get(f"/v1/jobs/{job['id']}")).status_code == 404
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 404
    assert (await client.post(f"/v1/jobs/{job['id']}/cancel")).status_code == 404
    assert (await client.delete(f"/v1/jobs/{job['id']}")).status_code == 404
    assert (await client.get("/v1/jobs")).json() == []


async def test_an_omitted_seed_is_reported(generations_app):
    client, device = _ltx_app(generations_app)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    (output,) = (await _wait_final(client, job["id"]))["outputs"]
    (request,) = device.requests
    assert isinstance(output["seed"], int) and 0 <= output["seed"] <= 2**31 - 1
    assert output["seed"] == request._seed


@pytest.mark.parametrize(
    ("seed", "status"), [(0, 202), (2**31 - 1, 202), (-1, 422), (2**31, 422)]
)
async def test_seed_range(generations_app, seed, status):
    client, _ = _ltx_app(generations_app)
    response = await client.post(_LTX[1], json={"prompt": "a cat", "seed": seed})
    assert response.status_code == status, response.text


async def test_a_full_queue_is_429_and_too_many_jobs_503(generations_app, monkeypatch):
    import utils.job_manager

    settings = utils.job_manager.get_settings()
    monkeypatch.setattr("utils.job_manager.get_settings", lambda: settings)
    settings.max_queue_size = 0
    client, _ = _ltx_app(generations_app)
    full = await client.post(_LTX[1], json={"prompt": "a cat"})
    assert full.status_code == 429 and full.headers["retry-after"] == "60"
    settings.max_queue_size, settings.max_jobs = 10, 0
    client, _ = _ltx_app(generations_app)
    too_many = await client.post(_LTX[1], json={"prompt": "a cat"})
    assert too_many.status_code == 503 and too_many.headers["retry-after"] == "60"


async def test_a_model_that_is_not_ready_is_503(generations_app):
    client, device = _ltx_app(generations_app)

    def not_ready():
        raise RuntimeError("warming up")

    device.check_is_model_ready = not_ready
    response = await client.post(_LTX[1], json={"prompt": "a cat"})
    assert response.status_code == 503 and response.headers["retry-after"] == "30"
    assert device.requests == []


async def test_every_route_needs_the_api_key(generations_app):
    from security import api_key_checker

    if api_key_checker.NO_AUTH:
        pytest.skip("the server runs with NO_AUTH")
    client, _ = generations_app(_LTX[0], auth=True)
    routes = [
        ("POST", _LTX[1]),
        ("GET", "/v1/jobs"),
        ("GET", "/v1/jobs/j"),
        ("DELETE", "/v1/jobs/j"),
        ("GET", "/v1/jobs/j/content/0"),
        ("POST", "/v1/jobs/j/cancel"),
    ]
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        for method, path in routes:
            response = await client.request(
                method, path, headers=headers, json={"prompt": "a cat"}
            )
            assert response.status_code == 401, (method, path, headers)
            assert isinstance(response.json()["detail"], str)
    key = {"Authorization": f"Bearer {api_key_checker.API_KEY}"}
    assert (await client.get("/v1/jobs", headers=key)).status_code == 200


async def test_failed_job_reports_error(generations_app, tmp_path):
    client, device = _ltx_app(generations_app)
    device.error = RuntimeError("device fell over")
    response = await client.post(_LTX[1], json={"prompt": "a cat"})
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "failed" and "outputs" not in job
    assert job["error"] == {"code": "processing_error", "message": "device fell over"}
    assert job["expires_at"] > job["completed_at"]
    assert (await client.get(f"/v1/jobs/{job['id']}/content/0")).status_code == 409
    # Made at submit, empty, and deleted with the job.
    assert list(tmp_path.iterdir()) == [tmp_path / job["id"]]
    assert list((tmp_path / job["id"]).iterdir()) == []


@pytest.fixture
def video_stats(monkeypatch):
    """The VideoGenerationStats the jobs report."""
    from model_services import generation_service

    stats = []
    client = SimpleNamespace(record_video_generation_async=stats.append)
    monkeypatch.setattr(generation_service, "get_telemetry_client", lambda: client)
    return stats


def _sample(name: str, **labels) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name, labels) or 0.0


_RATIO = "tt_media_server_video_requested_aspect_ratio_total"
_DURATION = "tt_media_server_video_requested_duration_seconds_sum"
_IN_FLIGHT = "tt_media_server_video_generations_in_progress"


@pytest.fixture
def runner_label(monkeypatch):
    """Sets the runner the metrics are labelled with. Earlier tests can leave
    generation_service with another settings object."""
    from model_services import generation_service

    def set_runner(runner: str) -> None:
        monkeypatch.setattr(generation_service.settings, "model_runner", runner)

    return set_runner


async def test_a_completed_job_reports_its_video(
    generations_app, video_stats, runner_label
):
    runner_label(_LTX[0])
    labels = {"model_type": _LTX[0], "request_type": "t2v"}
    before = {
        ratio: _sample(_RATIO, **labels, aspect_ratio=ratio)
        for ratio in ("unset", "16:9")
    }
    durations = _sample(_DURATION, **labels)
    client, device = _ltx_app(generations_app, delay=0.2)
    response = await client.post(_LTX[1], json={"prompt": "a violinist"})
    while not device.requests:
        await asyncio.sleep(0.01)
    assert _sample(_IN_FLIGHT, **labels) == 1
    assert (await _wait_final(client, response.json()["id"]))["status"] == "completed"

    (stats,) = video_stats
    assert (stats.status, stats.request_type) == ("success", "t2v")
    # The fake device's clip: 9 frames of 64x64 at 25 fps.
    assert (stats.width, stats.height, stats.num_frames) == (64, 64, 9)
    assert stats.content_seconds == 9 / 25 and stats.output_bytes > 0
    assert _sample(_IN_FLIGHT, **labels) == 0
    # Omitted fields count as unset, not as the defaults the model fills in.
    assert _sample(_RATIO, **labels, aspect_ratio="unset") == before["unset"] + 1
    assert _sample(_DURATION, **labels) == durations

    body = {"prompt": "a violinist", "aspect_ratio": "16:9", "duration": 6}
    response = await client.post(_LTX[1], json=body)
    await _wait_final(client, response.json()["id"])
    assert _sample(_RATIO, **labels, aspect_ratio="16:9") == before["16:9"] + 1
    assert _sample(_DURATION, **labels) == durations + 6


@requires_tt_metal_policy
async def test_failed_jobs_report_their_status(
    generations_app, video_stats, runner_label
):
    runner_label("tt-wan2.2-i2v")
    client, device = generations_app("tt-wan2.2-i2v")
    device.error = RuntimeError("device fell over")
    response = await client.post(_WAN, json=_wan_body(first_frame=_data_uri()))
    await _wait_final(client, response.json()["id"])
    # H3 carries its keyframes as files, not pixels.
    runner_label("tt-minimax-h3-fl2va")
    client, device = generations_app("tt-minimax-h3-fl2va")
    device.error = RuntimeError("device fell over")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-fl2va",
        json={
            "prompt": "x",
            "first_frame": _data_uri(600, 800),
            "last_frame": _data_uri(600, 800),
        },
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])
    assert [(s.status, s.request_type) for s in video_stats] == [("failure", "i2v")] * 2
    assert [s.conditioning_images for s in video_stats] == [1, 2]
    assert all(s.width is None for s in video_stats)
    for runner in ("tt-wan2.2-i2v", "tt-minimax-h3-fl2va"):
        assert _sample(_IN_FLIGHT, model_type=runner, request_type="i2v") == 0


async def _wait_for_stats(video_stats, count: int) -> None:
    for _ in range(100):
        if len(video_stats) == count:
            return
        await asyncio.sleep(0.01)


async def test_cancelled_jobs_report_their_status(
    generations_app, video_stats, runner_label
):
    runner_label(_LTX[0])
    labels = {"model_type": _LTX[0], "request_type": "t2v"}
    # Cancelled while running: the runner returns, the job ends cancelled.
    client, device = _ltx_app(generations_app, delay=0.5)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    while not device.requests:
        await asyncio.sleep(0.01)
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert (await _wait_final(client, job["id"]))["status"] == "cancelled"
    await _wait_for_stats(video_stats, 1)
    # Cancelled while queued: the job's task is cancelled.
    client, device = _ltx_app(generations_app)
    device.queued = 5.0
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
    while not device.requests:
        await asyncio.sleep(0.01)
    await client.post(f"/v1/jobs/{job['id']}/cancel")
    assert (await _wait_final(client, job["id"]))["status"] == "cancelled"
    await _wait_for_stats(video_stats, 2)
    assert [s.status for s in video_stats] == ["cancelled"] * 2
    assert _sample(_IN_FLIGHT, **labels) == 0


@pytest.mark.parametrize(
    ("runner", "forwards"),
    [
        ("tt-wan2.2-i2v", 40),
        ("tt-ltx-2.3-distilled", 11),
        ("tt-minimax-h3-fl2va", 49),
        ("tt-minimax-h3-ref2va", 49),
        ("tt-minimax-h3-fl2va-lightx2v", 4),
        ("tt-minimax-h3-ref2va-lightx2v", 4),
    ],
)
def test_video_metrics_count_the_forwards_each_runner_runs(runner, forwards):
    from config.constants import video_executed_inference_steps

    assert video_executed_inference_steps(None, runner) == forwards


async def test_a_generations_dir_in_any_spelling_works(
    generations_app, monkeypatch, tmp_path
):
    monkeypatch.setenv("TT_GENERATIONS_DIR", f"{tmp_path}//./")
    client, _ = _ltx_app(generations_app)
    response = await client.post(_LTX[1], json={"prompt": "a cat"})
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
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame=_data_uri(size, size))
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
    client, _ = _wan_app(generations_app)
    first = (
        await client.post(_WAN, json=_wan_body(first_frame=_data_uri(1600, 900)))
    ).json()
    (output,) = (await _wait_final(client, first["id"]))["outputs"]

    response = await client.post(_WAN, json=_wan_body(first_frame=output["url"]))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "first_frame"]


@requires_tt_metal_policy
def test_stored_request_digests_data_uris():
    from model_services.generation_service import _request_parameters

    image = _data_uri()
    request = MiniMaxH3Ref2VARequest(
        prompt="x", images=[image, "DATA" + image[4:], "https://x/a.png"]
    )
    digest, upper_digest, url = _request_parameters(request)["images"]
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert upper_digest.startswith("sha256:")
    assert url == "https://x/a.png"


def test_stored_request_drops_url_credentials():
    from model_services.generation_service import _request_parameters

    signed = "https://user:secret@bucket.example/a.png?X-Amz-Signature=abc#part"
    clean = "https://bucket.example/a.png"
    references = MiniMaxH3Ref2VARequest(prompt="x", images=[signed])
    assert _request_parameters(references)["images"] == [clean]
    keyframe = Wan22i2vA14bRequest(
        prompt="x", first_frame=signed, aspect_ratio="16:9", size="480p"
    )
    assert _request_parameters(keyframe)["first_frame"] == clean


def test_images_are_turned_upright():
    from utils.media_probe import decode_image

    # Stored landscape, shown portrait: EXIF orientation 6 is "rotate 90° CW".
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = BytesIO()
    Image.new("RGB", (40, 30)).save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (30, 40)


async def test_lost_outputs_are_404(generations_app, tmp_path):
    client, _ = _ltx_app(generations_app)
    response = await client.post(_LTX[1], json={"prompt": "a cat"})
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
    client, _ = _ltx_app(generations_app)
    job = (await client.post(_LTX[1], json={"prompt": "a cat"})).json()
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
@requires_tt_metal_policy
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
@requires_tt_metal_policy
def test_h3_output_size(fields, images, expected):
    request = _with_image_sizes(MiniMaxH3FL2VARequest(prompt="x", **fields), **images)
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
        ({"aspect_ratio": "2:1"}, "2:1 is not served; supported: 21:9"),
        ({"aspect_ratio": "16:9:1"}, "must look like 'W:H'"),
        ({"negative_prompt": "blur"}, "extra"),
    ],
)
@requires_tt_metal_policy
def test_h3_rejects(fields, message):
    with pytest.raises(ValidationError, match=message):
        MiniMaxH3FL2VARequest(prompt="x", **fields)


@requires_tt_metal_policy
def test_h3_aspect_ratio_defaults_to_adaptive_with_a_frame():
    assert MiniMaxH3FL2VARequest(prompt="x").aspect_ratio == "16:9"
    with_frame = MiniMaxH3FL2VARequest(prompt="x", last_frame="https://x/a.png")
    assert with_frame.aspect_ratio == "adaptive"


@pytest.mark.parametrize("ratio", ["16x9", "16/9", " 16 : 9 ", "016:9"])
@requires_tt_metal_policy
def test_h3_ratio_spellings_of_the_earlier_api(ratio):
    assert MiniMaxH3FL2VARequest(prompt="x", aspect_ratio=ratio).aspect_ratio == "16:9"


@pytest.mark.parametrize("ratio", ["17:9", "17x9"])
@requires_tt_metal_policy
def test_h3_unserved_ratio_gets_tt_metals_message(ratio):
    with pytest.raises(ValidationError, match="17:9 is not served; supported: 21:9"):
        MiniMaxH3FL2VARequest(prompt="x", aspect_ratio=ratio)


@requires_tt_metal_policy
def test_h3_output_size_and_frames_are_tt_metals():
    policy = importlib.import_module(H3_POLICY)
    for ratio in policy.MINIMAX_H3_ASPECT_RATIOS:
        # tt-metal's canvas is (height, width).
        height, width = policy.resolve_canvas_size(*ratio)
        request = MiniMaxH3FL2VARequest(prompt="x", aspect_ratio="{}:{}".format(*ratio))
        assert request.output_size() == (width, height)
    for duration in policy.MINIMAX_H3_DURATIONS_S:
        request = MiniMaxH3FL2VARequest(prompt="x", duration=duration)
        assert request.frames == policy.get_num_frames(duration)


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
@requires_tt_metal_policy
def test_h3_keyframe_limits(images, errors):
    request = _with_image_sizes(MiniMaxH3FL2VARequest(prompt="x"), **images)
    got = request.media_errors()
    assert [(name, index) for name, index, _ in got] == [(n, 0) for n, _ in errors]
    for (_, _, message), (_, expected) in zip(got, errors):
        assert expected in message


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"audios": ["https://x/a.wav"]}, "paired with at least one image or video"),
        ({}, "needs at least one reference"),
        (
            {
                "images": ["https://x/a.png"] * 9,
                "videos": ["https://x/a.mp4"] * 3,
                "audios": ["https://x/a.wav"],
            },
            "at most 12",
        ),
        ({"images": ["https://x/a.png"] * 10}, "at most 9"),
        (
            {"images": ["https://x/a.png"], "videos": ["data:image/png;base64,"]},
            "video",
        ),
        ({"images": ["https://x/a.png"], "audios": ["/v1/jobs/j/content/0"]}, "audio"),
    ],
)
@requires_tt_metal_policy
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
            [("audios", 0, "2 to 15 s"), ("audios", None, "16 s of audio together")],
        ),
        # A video's soundtrack counts toward the audio limits.
        (
            {"videos": [_probe(5.0)], "audios": [_probe(11.0, None, None)]},
            [("audios", None, "16 s of audio together")],
        ),
        (
            {"videos": [_probe(3.0)] * 3, "audios": [_probe(3.0, None, None)]},
            [("audios", None, "4 audio references")],
        ),
        ({"videos": [_probe(8.0), _probe(8.0)]}, [("videos", None, "16 s together")]),
        ({"videos": [_probe(size=None)]}, [("videos", 0, "no video stream")]),
        ({"videos": [_probe(size=(64, 300))]}, [("videos", 0, "within 1:4 and 4:1")]),
        ({"videos": [_probe(fps=None)]}, [("videos", 0, "no frame rate")]),
        ({"audios": [_probe(audio_channels=None)]}, [("audios", 0, "no audio stream")]),
        ({"audios": [_probe(audio_channels=6)]}, [("audios", 0, "mono or stereo")]),
    ],
)
@requires_tt_metal_policy
def test_h3_reference_clip_limits(probes, errors):
    request = MiniMaxH3Ref2VARequest(prompt="x", images=["https://x/a.png"])
    request._image_sizes = {"images": [(64, 64)]}
    request._probes = probes
    got = request.media_errors()
    assert [(name, index) for name, index, _ in got] == [(n, i) for n, i, _ in errors]
    for (_, _, message), (_, _, expected) in zip(got, errors):
        assert expected in message


def _admission_failing_with(monkeypatch, errors):
    import domain.minimax_h3_request as h3

    def admit_request(**kwargs):
        raise h3.MiniMaxH3AdmissionError(errors)

    monkeypatch.setattr(h3, "admit_request", admit_request)


def test_h3_admission_errors_land_at_their_input_or_the_request(monkeypatch):
    _admission_failing_with(
        monkeypatch, [("images[0]", "bad image"), (None, "bad request")]
    )
    request = MiniMaxH3Ref2VARequest(prompt="x", images=["https://x/a.png"])
    request._image_sizes = {"images": [(64, 64)]}
    assert request.media_errors() == [
        ("images", 0, "bad image"),
        ("references", None, "bad request"),
    ]


async def test_a_request_level_error_is_at_the_field_alone(
    generations_app, monkeypatch
):
    _admission_failing_with(
        monkeypatch, [("images[0]", "bad image"), (None, "bad request")]
    )
    client, device = generations_app("tt-minimax-h3-ref2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={"prompt": "x", "images": [_data_uri()]},
    )
    assert response.status_code == 422
    assert [(error["loc"], error["msg"]) for error in response.json()["detail"]] == [
        (["body", "images", 0], "bad image"),
        (["body", "references"], "bad request"),
    ]
    assert device.requests == []


@requires_tt_metal_policy
async def test_h3_ref2va_reports_every_bad_input_in_one_422(generations_app):
    client, device = generations_app("tt-minimax-h3-ref2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={
            "prompt": "x",
            "images": [_data_uri(64, 300), _data_uri(), _data_uri(300, 64)],
            "videos": [_clip_data_uri(8), _clip_data_uri(8)],
        },
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert [error["loc"] for error in errors] == [
        ["body", "images", 0],
        ["body", "images", 2],
        ["body", "videos"],
    ]
    assert errors[0]["msg"] == "images[0] must be within 1:4 and 4:1, got 64x300"
    assert errors[1]["msg"] == "images[2] must be within 1:4 and 4:1, got 300x64"
    assert "the clips are 16 s together" in errors[2]["msg"]
    assert device.requests == []


@requires_tt_metal_policy
async def test_h3_keyframe_ratio_is_checked_by_tt_metal(generations_app):
    client, device = generations_app("tt-minimax-h3-fl2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-fl2va",
        json={"prompt": "x", "first_frame": _data_uri(300, 1500)},
    )
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "first_frame"]
    assert error["msg"] == (
        "first_frame is 300x1500; its aspect ratio must be from 1:4 to 4:1"
    )
    assert device.requests == []


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


@requires_tt_metal_policy
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
    # The runner decodes the keyframe: only its file goes to the device, and
    # the field holds the data URI's digest.
    assert request._images == {}
    assert request._image_sizes == {"first_frame": [(600, 800)]}
    assert Image.open(BytesIO(request._files["first_frame"][0])).size == (600, 800)
    assert request.first_frame.startswith("sha256:")
    (output,) = job["outputs"]
    assert output["audio"] == {"codec": "aac", "sample_rate": 48000, "channels": 2}
    assert output["fps"] == 24 and output["num_frames"] == 124
    assert output["duration"] == 124 / 24


@requires_tt_metal_policy
async def test_h3_sees_a_keyframes_upright_size(generations_app):
    client, device = generations_app("tt-minimax-h3-fl2va")
    device.video = lambda request: _sound_video(request.frames, 48000, 24)
    # Stored landscape, shown portrait: EXIF orientation 6 is "rotate 90° CW".
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = BytesIO()
    Image.new("RGB", (800, 600)).save(buffer, format="JPEG", exif=exif)
    keyframe = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-fl2va",
        json={"prompt": "a lighthouse", "first_frame": keyframe},
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])
    (request,) = device.requests
    assert request._image_sizes == {"first_frame": [(600, 800)]}
    assert (request._width, request._height) == (768, 1024)


@requires_tt_metal_policy
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


@requires_tt_metal_policy
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
    assert request._image_sizes == {"images": [(64, 96)]}
    assert Image.open(BytesIO(request._files["images"][0])).size == (64, 96)
    # Only the files go to the device: each field holds its data URI's digest,
    # as the stored request does.
    from model_services.generation_service import _without_data

    inputs = {"images": _data_uri(64, 96), "videos": video, "audios": audio}
    for name, uri in inputs.items():
        assert getattr(request, name) == [_without_data(uri)]
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


_INLINE_CAP = "model_services.generation_service.MAX_INLINE_MEDIA_CHARS"


async def test_inline_media_over_the_total_cap_is_413(generations_app, monkeypatch):
    image, video = _data_uri(), _clip_data_uri(3)
    total = len(image) + len(video)
    monkeypatch.setattr(_INLINE_CAP, total - 1)
    client, device = generations_app("tt-minimax-h3-ref2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={"prompt": "x", "images": [image], "videos": [video]},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == (
        f"Inline media totals {total} characters, over the {total - 1}-character "
        "cap; send large assets by URL"
    )
    assert device.requests == [] and (await client.get("/v1/jobs")).json() == []


async def test_the_inline_cap_is_checked_before_any_fetch(generations_app, monkeypatch):
    async def fetch(url, **kwargs):
        raise AssertionError("fetched")

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    monkeypatch.setattr(_INLINE_CAP, 1)
    client, _ = generations_app("tt-minimax-h3-ref2va")
    response = await client.post(
        "/v1/generations/minimaxai/minimax-h3-ref2va",
        json={
            "prompt": "x",
            "images": [_data_uri()],
            "videos": ["https://example.com/a.mp4"],
        },
    )
    assert response.status_code == 413


async def test_media_urls_do_not_count_toward_the_inline_cap(
    generations_app, monkeypatch
):
    png = base64.b64decode(_data_uri().split(",")[1])

    async def fetch(url, **kwargs):
        return png

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    monkeypatch.setattr(_INLINE_CAP, 0)
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN, json=_wan_body(first_frame="https://example.com/a.png")
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])


async def test_inline_media_at_the_cap_is_accepted(generations_app, monkeypatch):
    png = base64.b64decode(_data_uri().split(",")[1])

    async def fetch(url, **kwargs):
        return png

    monkeypatch.setattr("model_services.generation_service.download_media_url", fetch)
    image = _data_uri()
    monkeypatch.setattr(_INLINE_CAP, len(image))
    client, _ = _wan_app(generations_app)
    response = await client.post(
        _WAN,
        json=_wan_body(first_frame=image, last_frame="https://example.com/b.png"),
    )
    assert response.status_code == 202, response.text
    await _wait_final(client, response.json()["id"])


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


# --- FLUX.3 Image klein ----------------------------------------------------------


@pytest.fixture
def flux3_upsampler(monkeypatch):
    """Where FLUX.3's prompt upsampler runs: "resident" (the default) or
    "ahead" of the deployment."""

    def place(where: str):
        monkeypatch.setenv(
            "FLUX3_PROMPT_UPSAMPLER_FROZEN", {"resident": "0", "ahead": "1"}[where]
        )

    place("resident")
    return place


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, (2048, 2048)),
        ({"size": "1K"}, (1024, 1024)),
        ({"size": "4K"}, (4096, 4096)),
    ],
)
def test_flux3_output_size(flux3_upsampler, fields, expected):
    flux3_upsampler("ahead")
    assert Flux3ImageKleinRequest(prompt="a cat", **fields).output_size() == expected


def test_flux3_serves_4k_only_with_the_upsampler_ahead(flux3_upsampler):
    with pytest.raises(ValidationError, match="4K is served with the prompt upsampler"):
        Flux3ImageKleinRequest(prompt="a cat", size="4K")
    assert Flux3ImageKleinRequest.warmup_sizes() == [(1024, 1024), (2048, 2048)]
    flux3_upsampler("ahead")
    assert Flux3ImageKleinRequest(prompt="a cat", size="4K").output_size() == (
        4096,
        4096,
    )
    assert Flux3ImageKleinRequest.warmup_sizes() == [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]


@pytest.mark.parametrize("with_image", [False, True])
async def test_flux3_refuses_4k_at_submit_with_the_upsampler_resident(
    generations_app, flux3_upsampler, with_image
):
    client, device = generations_app("tt-flux-3-image-klein")
    body = {"prompt": "a cat", "size": "4K"}
    if with_image:
        body["images"] = [_data_uri(1600, 900)]
    response = await client.post(
        "/v1/generations/black-forest-labs/flux-3-image-klein", json=body
    )
    assert response.status_code == 422, response.text
    assert "4K is served with the prompt upsampler ahead" in response.text
    assert device.requests == []
    # 2K is served
    response = await client.post(
        "/v1/generations/black-forest-labs/flux-3-image-klein",
        json={**body, "size": "2K"},
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job


@pytest.mark.parametrize(
    "fields", [{"aspect_ratio": "16:9"}, {"images": ["https://x/a.png"] * 2}]
)
def test_flux3_serves_one_square_shape_per_size(fields):
    with pytest.raises(ValidationError):
        Flux3ImageKleinRequest(prompt="a cat", **fields)


def test_flux3_bounds_the_prompt():
    assert Flux3ImageKleinRequest(prompt="a" * 20_000).prompt
    with pytest.raises(ValidationError):
        Flux3ImageKleinRequest(prompt="a" * 20_001)


def test_flux3_has_3_sizes(flux3_upsampler):
    flux3_upsampler("ahead")
    assert Flux3ImageKleinRequest.warmup_sizes() == [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]


async def test_flux3_keeps_only_the_image_sizes(generations_app):
    client, device = generations_app("tt-flux-3-image-klein")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux-3-image-klein",
        json={"prompt": "make it night", "images": [_data_uri(1600, 900)]},
    )
    assert response.status_code == 202, response.text
    job = await _wait_final(client, response.json()["id"])
    assert job["status"] == "completed", job
    (request,) = device.requests
    assert request._images == {}
    assert request._image_sizes == {"images": [(1600, 900)]}
    assert Image.open(BytesIO(request._files["images"][0])).size == (1600, 900)


async def test_flux3_reports_the_upsampled_caption(generations_app, monkeypatch):
    process = _FakeDevice.process

    async def upsampled(self, request):
        image = await process(self, request)
        image.info["prompt"] = "a detailed caption"
        return image

    monkeypatch.setattr(_FakeDevice, "process", upsampled)
    client, device = generations_app("tt-flux-3-image-klein")
    response = await client.post(
        "/v1/generations/black-forest-labs/flux-3-image-klein",
        json={"prompt": "a cat", "size": "1K"},
    )
    job = await _wait_final(client, response.json()["id"])
    assert device.requests[0]._prompt == "a cat"
    (output,) = job["outputs"]
    assert output["prompt"] == "a detailed caption"
