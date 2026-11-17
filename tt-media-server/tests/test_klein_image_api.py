# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""FLUX.3 Image klein's synchronous /v1/images/generations and /edits: an
ordinary Klein job, waited for and returned inline, with a fake device."""

import asyncio
import base64
import importlib.util
import pathlib
import threading
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from config.settings import get_settings, settings
from fastapi import FastAPI
from PIL import Image

from tests.test_generations_api import _data_uri, _FakeDevice

_KLEIN = "tt-flux-3-image-klein"


class _ClaimingDevice(_FakeDevice):
    """Publishes its claim while it renders, as a device worker does."""

    async def process(self, request):
        start = request._start_event.set

        def claim():
            request._worker_assignment.identity = ("worker", 0)
            start()

        request._start_event.set = claim
        try:
            return await super().process(request)
        finally:
            request._worker_assignment.identity = None


def _image_module(monkeypatch, runner: str = _KLEIN):
    monkeypatch.setattr(settings, "model_runner", runner)
    path = pathlib.Path(__file__).resolve().parents[1] / "open_ai_api" / "image.py"
    spec = importlib.util.spec_from_file_location("image_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def klein_app(monkeypatch, tmp_path):
    """(client, device, service, module) serving Klein on a Galaxy quad with
    the prompt upsampler ahead, as the quad does."""
    from model_services.generation_service import GenerationService
    from resolver.service_resolver import service_resolver
    from security.api_key_checker import get_api_key
    from utils.job_manager import JobManager

    monkeypatch.setenv("TT_GENERATIONS_DIR", str(tmp_path))
    monkeypatch.setenv("FLUX3_PROMPT_UPSAMPLER_FROZEN", "1")
    monkeypatch.setattr(get_settings(), "device", "galaxy_quad")
    monkeypatch.setattr(
        "utils.job_manager.get_settings",
        lambda: SimpleNamespace(
            enable_job_persistence=False,
            max_queue_size=10,
            max_jobs=10,
            job_cleanup_interval_seconds=3600,
            job_retention_seconds=3600,
            job_max_stuck_time_seconds=3600,
        ),
    )
    device = _ClaimingDevice()
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

    module = _image_module(monkeypatch)
    app = FastAPI()
    app.include_router(module.generation_router, prefix="/v1/images")
    app.dependency_overrides[service_resolver] = lambda: service
    app.dependency_overrides[get_api_key] = lambda: None
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    yield client, device, service, module
    await service._job_manager.shutdown()
    await client.aclose()


def _decoded(response) -> Image.Image:
    (image,) = response.json()["images"]
    return Image.open(BytesIO(base64.b64decode(image)))


def _bare_base64(width: int, height: int) -> str:
    return _data_uri(width, height).split(",", 1)[1]


def _unpadded_jpeg(width: int, height: int) -> str:
    """A JPEG data URL with its base64 padding trimmed, as JSON transport can."""
    return _data_uri(width, height, "JPEG").rstrip("=")


@pytest.mark.parametrize(
    ("size", "image_format"),
    [
        ({}, "JPEG"),
        ({"width": None, "height": None}, "PNG"),
        ({"width": 1024, "height": 1024}, "JPEG"),
        ({"width": 2048, "height": 2048}, "PNG"),
        ({"width": 4096, "height": 4096}, "PNG"),
        ({"width": 4096, "height": 2304}, "JPEG"),
        ({"width": 2304, "height": 4096}, "JPEG"),
    ],
)
async def test_generations_waits_for_the_image(klein_app, size, image_format):
    client, device, service, _ = klein_app
    response = await client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "seed": 7,
            "image_return_format": image_format,
            **size,
        },
    )
    assert response.status_code == 200, response.text
    assert set(response.json()) == {"images", "generation_time"}
    image = _decoded(response)
    assert image.format == image_format
    # Console sends no size: 4096x4096.
    assert image.size == (size.get("width") or 4096, size.get("height") or 4096)
    (request,) = device.requests
    assert (request.prompt, request._seed, request.images) == ("a cat", 7, [])
    # Returned, so not kept as a job.
    assert service.get_all_jobs_metadata() == []


@pytest.mark.parametrize(
    ("size", "message"),
    [
        ({"width": 1536, "height": 1536}, "1536x1536 is not served"),
        ({"width": 1024, "height": 2048}, "1024x2048 is not served"),
        ({"width": 4096, "height": 4097}, "less than or equal to 4096"),
        ({"width": 1024}, "width and height must be provided together"),
    ],
)
async def test_generations_refuses_shapes_klein_does_not_serve(
    klein_app, size, message
):
    client, device, _, _ = klein_app
    response = await client.post(
        "/v1/images/generations", json={"prompt": "a cat", **size}
    )
    assert response.status_code == 422, response.text
    assert message in response.text
    assert device.requests == []


async def test_without_4k_the_default_is_2048(klein_app, monkeypatch):
    client, _, _, _ = klein_app
    monkeypatch.setenv("FLUX3_PROMPT_UPSAMPLER_FROZEN", "0")
    response = await client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert response.status_code == 200, response.text
    assert _decoded(response).size == (2048, 2048)


async def test_4k_is_refused_with_the_upsampler_in_the_deployment(
    klein_app, monkeypatch
):
    client, device, _, _ = klein_app
    monkeypatch.setenv("FLUX3_PROMPT_UPSAMPLER_FROZEN", "0")
    response = await client.post(
        "/v1/images/generations",
        json={"prompt": "a cat", "width": 4096, "height": 4096},
    )
    assert response.status_code == 422, response.text
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "width"]
    assert "4K is served with the prompt upsampler ahead" in error["msg"]
    assert device.requests == []


@pytest.mark.parametrize("count", [2, 4])
async def test_one_image_per_request(klein_app, count):
    client, device, _, _ = klein_app
    response = await client.post(
        "/v1/images/generations", json={"prompt": "a cat", "number_of_images": count}
    )
    assert response.status_code == 422, response.text
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "number_of_images"]
    assert "1 image per request; send one request per image" in error["msg"]
    assert device.requests == []


# As Console's worker sends them (tt-cloud-console ttservice.go imageGenRequest).
_CONSOLE_FIELDS = {
    "negative_prompt": "dogs",
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "number_of_images": 1,
    "image_return_format": "JPEG",
    "lora_scale": 0.5,
    "lora_path": "some/lora",
}


async def test_steps_guidance_negative_prompt_and_lora_are_ignored(klein_app):
    client, device, _, _ = klein_app
    response = await client.post(
        "/v1/images/generations", json={"prompt": "a cat", **_CONSOLE_FIELDS}
    )
    assert response.status_code == 200, response.text
    (request,) = device.requests
    assert request.model_dump(exclude={"images"}) == {
        "prompt": "a cat",
        "count": 1,
        "aspect_ratio": "1:1",
        "size": "4K",
        "format": "jpeg",
        "seed": None,
    }


async def test_jpeg_quality_is_honoured(klein_app):
    client, _, _, _ = klein_app
    luminance_dc = []
    for quality in (50, 100):
        response = await client.post(
            "/v1/images/generations",
            json={"prompt": "a cat", "image_quality": quality},
        )
        assert response.status_code == 200, response.text
        luminance_dc.append(_decoded(response).quantization[0][0])
    # Quality 100 quantizes by 1.
    assert luminance_dc[0] > 1 and luminance_dc[1] == 1


def test_other_image_runners_keep_their_limits(monkeypatch):
    from domain.image_generate_request import ImageGenerateRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="less than or equal to 1536"):
        ImageGenerateRequest(prompt="a cat", width=4096, height=4096)
    module = _image_module(monkeypatch, "tt-wan2.2-i2v")
    assert module.generation_router.routes == []


@pytest.mark.parametrize("route", ["image-to-image", "edits"])
@pytest.mark.parametrize("encode", [_bare_base64, _data_uri, _unpadded_jpeg])
async def test_image_is_the_reference(klein_app, encode, route):
    client, device, _, _ = klein_app
    response = await client.post(
        f"/v1/images/{route}",
        json={
            "prompt": "make it night",
            "image": encode(1600, 900),
            "strength": 0.3,
            **_CONSOLE_FIELDS,
        },
    )
    assert response.status_code == 200, response.text
    assert _decoded(response).size == (4096, 4096)
    (request,) = device.requests
    assert request._image_sizes == {"images": [(1600, 900)]}
    assert Image.open(BytesIO(request._files["images"][0])).size == (1600, 900)


async def test_image_to_image_takes_a_served_size(klein_app):
    client, _, _, _ = klein_app
    response = await client.post(
        "/v1/images/image-to-image",
        json={"prompt": "x", "image": _data_uri(), "width": 2304, "height": 4096},
    )
    assert response.status_code == 200, response.text
    assert _decoded(response).size == (2304, 4096)


async def test_edits_refuses_a_mask(klein_app):
    client, device, _, _ = klein_app
    response = await client.post(
        "/v1/images/edits",
        json={"prompt": "x", "image": _data_uri(), "mask": _data_uri(), "strength": 1},
    )
    assert response.status_code == 422, response.text
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "mask"]
    assert "send the image to /v1/images/image-to-image" in error["msg"]
    assert device.requests == []


@pytest.mark.parametrize(
    ("route", "fields", "field"),
    [
        ("edits", {"mask": "m" * 1_000_000}, "mask"),
        ("image-to-image", {"strength": 5.0, "seed": "x" * 1_000_000}, "seed"),
    ],
)
async def test_a_422_does_not_echo_the_input(klein_app, route, fields, field):
    client, _, _, _ = klein_app
    response = await client.post(
        f"/v1/images/{route}",
        json={"prompt": "x", "image": "i" * 1_000_000, **fields},
    )
    assert response.status_code == 422, response.text[:500]
    assert len(response.content) < 2_000
    errors = response.json()["detail"]
    assert field in {error["loc"][1] for error in errors}
    assert all("input" not in error and error["msg"] for error in errors)


async def test_a_reference_over_16_mp_is_refused(klein_app):
    client, device, _, _ = klein_app
    buffer = BytesIO()
    Image.new("L", (4097, 4096)).save(buffer, "PNG", compress_level=1)
    response = await client.post(
        "/v1/images/image-to-image",
        json={
            "prompt": "make it night",
            "image": base64.b64encode(buffer.getvalue()).decode(),
        },
    )
    assert response.status_code == 422, response.text
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "image"]
    assert "at most 16 MP (16,777,216 pixels) is accepted" in error["msg"]
    assert device.requests == []


async def test_an_image_that_is_not_one_is_refused(klein_app):
    client, device, _, _ = klein_app
    response = await client.post(
        "/v1/images/image-to-image",
        json={"prompt": "make it night", "image": base64.b64encode(b"nope").decode()},
    )
    assert response.status_code == 422, response.text
    assert device.requests == []


async def test_a_wait_past_the_timeout_leaves_the_render_running(
    klein_app, monkeypatch
):
    client, device, service, module = klein_app
    monkeypatch.setattr(module, "KLEIN_WAIT_SECONDS", 0.2)
    device.delay = 1.0
    response = await client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert response.status_code == 504, response.text
    assert "still rendering" in response.text
    (job,) = service.get_all_jobs_metadata()
    for _ in range(300):
        if service.get_job_metadata(job["id"])["status"] == "completed":
            break
        await asyncio.sleep(0.01)
    assert service.get_job_metadata(job["id"])["status"] == "completed"


async def test_a_wait_past_the_timeout_cancels_a_job_still_queued(
    klein_app, monkeypatch
):
    client, device, service, module = klein_app
    monkeypatch.setattr(module, "KLEIN_WAIT_SECONDS", 0.2)
    device.queued = 1.0
    response = await client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert response.status_code == 504, response.text
    assert "cancelled before it started" in response.text
    (job,) = service.get_all_jobs_metadata()
    assert job["status"] == "cancelled"


@pytest.mark.parametrize(("queued", "status"), [(0.0, "completed"), (1.0, "cancelled")])
async def test_a_client_that_leaves_never_stops_a_render(klein_app, queued, status):
    _, device, service, module = klein_app
    device.queued, device.delay = queued, 0.3
    request = module._klein_request(
        module.KleinImageGenerateRequest(prompt="a cat"), image=None
    )
    job = await service.submit(request, "black-forest-labs/flux-3-image-klein")
    await asyncio.sleep(0.1)  # the device claims it unless it is held queued

    async def gone():
        return True

    with pytest.raises(module.HTTPException) as error:
        await module._wait_for_job(service, job["id"], gone)
    assert error.value.status_code == 504
    for _ in range(300):
        if service.get_job_metadata(job["id"])["status"] in ("completed", "cancelled"):
            break
        await asyncio.sleep(0.01)
    assert service.get_job_metadata(job["id"])["status"] == status


async def test_a_failed_render_is_500(klein_app):
    client, device, service, _ = klein_app
    device.error = RuntimeError("device fell over")
    response = await client.post("/v1/images/generations", json={"prompt": "a cat"})
    assert response.status_code == 500
    assert response.json() == {"detail": "device fell over"}


def test_v1_models_lists_the_image_api_fields(monkeypatch):
    import open_ai_api.models as models

    monkeypatch.setattr(settings, "model_service", "generation")
    monkeypatch.setattr(settings, "model_runner", _KLEIN)
    (entry,) = models.list_models()["data"]
    assert entry["id"] == "black-forest-labs/flux-3-image-klein"
    properties = entry["schema"]["properties"]
    assert {"prompt", "image", "strength", "width", "height"} <= set(properties)
    assert properties["width"]["anyOf"][0]["maximum"] == 4096
    assert entry["schema"]["required"] == ["prompt"]
