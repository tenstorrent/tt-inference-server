# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wan2.2 I2V spec tests (create contract, lifecycle) against an in-process server."""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import json
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module._test_common import wan_i2v_generation_api as api
from test_module._test_common.large_images import MAX_IMAGE_BYTES
from test_module._test_common.video_quality_metrics import parse_ffmpeg_streams
from test_module.load_param_tests import wan_i2v_create_contract_test as contract
from test_module.load_param_tests.video_generation_i2v_test import (
    VideoGenerationI2VTest,
)
from tests.test_module.fake_generation_api import FakeJobsServer
from tests.test_module.video_clips import gradient, moving_frames, write_clip

web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

MEDIA_SERVER = Path(__file__).resolve().parents[2] / "tt-media-server"
API_KEY = "test-key"


def test_default_api_key_matches_the_server():
    checker = (MEDIA_SERVER / "security/api_key_checker.py").read_text()
    default = re.search(r'os\.getenv\("API_KEY", "([^"]+)"\)', checker).group(1)
    assert api.DEFAULT_API_KEY == default


def test_tiny_png_decodes():
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.open(io.BytesIO(base64.b64decode(contract.TINY_PNG_BASE64)))
    assert image.size == (1, 1)
    assert api.data_uri(base64.b64decode(contract.TINY_PNG_BASE64)) == (
        contract.TINY_PNG_DATA_URI
    )


def test_case_names_are_unique():
    names = [case.name for case in contract.cases()]
    assert len(names) == len(set(names))


def test_valid_payload_is_valid():
    # Each case must break only what it names, so the base request breaks nothing.
    assert not _validation_errors(contract.valid_payload())


def test_the_unfetchable_url_can_point_at_an_allowed_host():
    url = "https://allowed.example.com/missing.png"
    [case] = [c for c in contract.cases(url) if c.name == "unfetchable_first_frame_url"]
    assert case.payload["first_frame"] == url


def test_output_shapes_are_the_specs():
    assert api.OUTPUT_SHAPES == {
        ("16:9", "480p"): (832, 480),
        ("9:16", "480p"): (480, 832),
        ("16:9", "720p"): (1280, 720),
        ("9:16", "720p"): (720, 1280),
    }


async def _with_server(routes, body):
    """Run ``body(base_url)`` against an aiohttp app serving ``routes``."""
    # The real server has no body limit; the over-limit image case sends ~45 MB.
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.add_routes(routes)
    async with test_utils.TestServer(app) as server:
        return await body(str(server.make_url("")).rstrip("/"))


def _error(loc: str, msg: str, type_: str, value=None) -> dict:
    # Like FastAPI, an error echoes the input that failed.
    return {"loc": ["body", loc], "msg": msg, "type": type_, "input": value}


def _validation_errors(body: dict) -> list[dict]:
    """The request schema's errors, all of them, as Pydantic would list them."""
    errors = [
        _error(key, "Extra inputs are not permitted", "extra_forbidden", body[key])
        for key in sorted(body.keys() - api.REQUEST_FIELDS)
    ]
    errors += [
        _error(key, "Field required", "missing")
        for key in ("prompt", "first_frame", "aspect_ratio", "size")
        if key not in body
    ]
    if body.get("prompt") == "":
        errors.append(_error("prompt", "at least 1 character", "string_too_short"))
    for key, allowed in (("aspect_ratio", api.ASPECT_RATIOS), ("size", api.SIZES)):
        if key in body and body[key] not in allowed:
            errors.append(_error(key, f"Input should be {allowed}", "enum", body[key]))
    if "frames" in body and body["frames"] != api.NUM_FRAMES:
        errors.append(_error("frames", "Input should be 81", "literal_error"))
    if "seed" in body and not 0 <= body["seed"] <= api.SEED_MAX:
        errors.append(_error("seed", "out of range", "less_than_equal"))
    for key in ("first_frame", "last_frame"):
        value = body.get(key)
        if value is not None and not value.startswith(("data:", "https://", "http://")):
            errors.append(_error(key, "a data URI or an URL", "value_error", value))
    return errors


def _media_refusal(key: str, value: str, oversize_status: int):
    """The fetch policy and image checks, after the schema passed."""
    if value.startswith("http://"):
        return web.json_response({"detail": f"{key}: https only"}, status=400)
    if value.startswith("https://"):
        host = urlsplit(value).hostname
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            return web.json_response({"detail": f"{key}: not public"}, status=400)
        error = _error(key, "could not be fetched", "fetch_error", value)
        return web.json_response({"detail": [error]}, status=422)
    image = base64.b64decode(value.split(",", 1)[1])
    if len(image) > MAX_IMAGE_BYTES:
        if oversize_status == 422:  # today's server: a 422 echoing the image
            error = _error(key, "too large", "value_error", value)
            return web.json_response({"detail": [error]}, status=422)
        return web.json_response({"detail": f"{key}: over 30 MB"}, status=413)
    if not image.startswith((b"\x89PNG", b"\xff\xd8\xff")):
        error = _error(key, "not a supported image", "value_error")
        return web.json_response({"detail": [error]}, status=422)
    return None


def _contract_routes(
    *,
    accept_everything: bool = False,
    first_error_only: bool = False,
    string_detail: bool = False,
    no_loc: bool = False,
    oversize_status: int = 413,
    cancelled: list | None = None,
):
    """A server that refuses requests as the spec says; the knobs break it."""

    async def submit(request):
        auth = request.headers.get("Authorization")
        if auth != f"Bearer {API_KEY}":
            return web.json_response(
                {"detail": "Invalid or missing API Key"}, status=401
            )
        if request.path != api.SUBMIT_PATH:
            return web.json_response({"detail": "Model not served"}, status=404)
        body = await request.json()
        if accept_everything:
            return web.json_response({"id": str(uuid.uuid4())}, status=202)
        if errors := _validation_errors(body):
            if first_error_only:
                errors = errors[:1]
            if no_loc:
                errors = [{**e, "loc": ["body"]} for e in errors]
            detail = "invalid request" if string_detail else errors
            return web.json_response({"detail": detail}, status=422)
        for key in ("first_frame", "last_frame"):
            if key in body:
                refusal = _media_refusal(key, body[key], oversize_status)
                if refusal is not None:
                    return refusal
        return web.json_response({"id": str(uuid.uuid4())}, status=202)

    async def cancel(request):
        if cancelled is not None:
            cancelled.append(request.match_info["job_id"])
        return web.json_response({}, status=200)

    return [
        web.post("/v1/generations/{org}/{model}", submit),
        web.post("/v1/jobs/{job_id}/cancel", cancel),
    ]


def _run_contract(**route_kwargs) -> dict:
    return asyncio.run(
        _with_server(
            _contract_routes(**route_kwargs),
            lambda url: contract.run_create_contract(base_url=url, api_key=API_KEY),
        )
    )


def _by_check(result: dict) -> dict:
    return {r["check"]: r for r in result["detailed_test_results"]}


def test_contract_passes_against_a_conforming_server():
    result = _run_contract()
    assert result["success"], [
        r for r in result["detailed_test_results"] if not r["passed"]
    ]
    assert (
        result["summary"]
        == f"{len(contract.cases())}/{len(contract.cases())} checks passed"
    )


def test_contract_fails_and_cancels_when_invalid_requests_create_jobs():
    cancelled = []
    checks = _by_check(_run_contract(accept_everything=True, cancelled=cancelled))
    assert not checks["frames_above_81"]["passed"]
    assert "created job" in checks["frames_above_81"]["message"]
    # Auth still refuses, so the auth checks keep passing.
    assert checks["invalid_bearer_authentication"]["passed"]
    # Every job a refused request created was cancelled.
    assert len(cancelled) == sum(not c["passed"] for c in checks.values())


def test_contract_wants_every_problem_listed():
    checks = _by_check(_run_contract(first_error_only=True))
    failed = {name for name, c in checks.items() if not c["passed"]}
    assert failed == {"every_problem_is_reported"}
    assert "at least 2" in checks["every_problem_is_reported"]["message"]


def test_contract_wants_a_validation_error_body_for_a_422():
    checks = _by_check(_run_contract(string_detail=True))
    assert not checks["unsupported_size"]["passed"]
    assert "expected a list of errors" in checks["unsupported_size"]["message"]


def test_contract_wants_the_errors_to_name_the_field():
    checks = _by_check(_run_contract(no_loc=True))
    assert "no error names 'seed'" in checks["negative_seed"]["message"]


def test_contract_wants_413_for_an_image_over_30_mb():
    # Today's server refuses a big image with a 422 that echoes all ~45M
    # characters of it; the report keeps only the start.
    over = _by_check(_run_contract(oversize_status=422))["first_frame_over_size_limit"]
    assert not over["passed"]
    assert over["message"] == "expected 413, got 422"
    assert len(str(over["response"])) < contract.MAX_REPORTED_RESPONSE_CHARS + 100


def test_contract_reports_an_unreachable_server():
    result = asyncio.run(
        contract.run_create_contract(
            base_url="http://127.0.0.1:9", api_key=API_KEY, request_timeout=2
        )
    )
    assert not result["success"]
    assert {r["actual_status"] for r in result["detailed_test_results"]} == {
        "request_error"
    }


def _run_lifecycle(
    server: FakeJobsServer,
    device: str = "T3K",
    env_vars: dict | None = None,
    output_path=None,
    targets: dict | None = None,
) -> dict:
    async def body(url):
        ctx = SimpleNamespace(
            service_port=0,
            base_url=url,
            model_spec=SimpleNamespace(
                hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                env_vars=env_vars or {},
            ),
            device=SimpleNamespace(name=device),
            output_path=output_path,
        )
        test = VideoGenerationI2VTest(
            _TestConfig.create_default(),
            {"poll_interval": 0, "poll_timeout": 30, **(targets or {})},
            ctx=ctx,
        )
        return await test._run_specific_test_async()

    return asyncio.run(_with_server(server.routes(), body))


@pytest.fixture
def wan_480p_clip(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("imageio.v3")
    return write_clip(tmp_path / "wan.mp4", moving_frames(81, 832, 480), faststart=True)


def _server(clip, **kwargs) -> FakeJobsServer:
    return FakeJobsServer(lambda request: clip, **kwargs)


def test_lifecycle_passes_for_a_good_job(wan_480p_clip):
    server = _server(wan_480p_clip)
    result = _run_lifecycle(server)
    assert result["success"], result["problems"]
    # The job, the queued one and the running one were all cancelled or deleted.
    assert not server.jobs
    [first, queued, running] = server.requests
    assert first["first_frame"].startswith("data:image/jpeg;base64,")
    assert (first["aspect_ratio"], first["size"]) == ("16:9", "480p")
    assert {first["seed"], queued["seed"], running["seed"]} == {42, 43, 44}


def test_lifecycle_fails_when_the_job_is_not_listed(wan_480p_clip):
    result = _run_lifecycle(_server(wan_480p_clip, list_jobs=False))
    assert not result["success"]
    assert "job is missing from the jobs list" in result["problems"]


def test_lifecycle_fails_when_the_output_does_not_match_the_request(wan_480p_clip):
    result = _run_lifecycle(_server(wan_480p_clip, output={"seed": 7, "fps": 24}))
    assert "output: seed is 7, expected 42" in result["problems"]
    assert "output: fps is 24, expected 16" in result["problems"]


def test_lifecycle_wants_jobs_to_expire_a_day_after_they_finish(wan_480p_clip):
    result = _run_lifecycle(_server(wan_480p_clip, retention_seconds=3600))
    assert any(
        p.startswith("job: expires_at") and "+ 86400" in p for p in result["problems"]
    )


def test_lifecycle_wants_cancel_to_stop_a_running_job(wan_480p_clip):
    result = _run_lifecycle(_server(wan_480p_clip, cancel_running=False))
    problems = [p for p in result["problems"] if p.startswith("cancel running job")]
    assert any("left it 'in_progress'" in p for p in problems), problems
    assert any("ended completed, expected cancelled" in p for p in problems), problems
    # Cancelling a queued job still works.
    assert not any(p.startswith("cancel queued job") for p in result["problems"])


def test_lifecycle_wants_an_active_job_kept_from_deletion(wan_480p_clip):
    result = _run_lifecycle(_server(wan_480p_clip, delete_active_status=200))
    assert any(
        p.startswith("cancel queued job: deleting an active job")
        for p in result["problems"]
    ), result["problems"]


def test_lifecycle_fails_on_a_frozen_clip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    clip = write_clip(
        tmp_path / "frozen.mp4", [gradient(832, 480)] * 81, faststart=True
    )
    result = _run_lifecycle(_server(clip), output_path=tmp_path / "out")
    assert not result["success"]
    assert any(problem.startswith("clip: frozen") for problem in result["problems"])
    saved = list((tmp_path / "out/videos/failed/spec_tests").iterdir())
    assert [p.name for p in saved] == [f"lifecycle_{result['job_id']}.mp4"]


def test_lifecycle_wants_a_faststart_mp4(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    clip = write_clip(tmp_path / "slow.mp4", moving_frames(81, 832, 480))
    result = _run_lifecycle(_server(clip))
    assert any(p.startswith("container: not faststart") for p in result["problems"])


def test_lifecycle_asks_for_720p_on_a_galaxy_and_checks_the_clip_has_it(wan_480p_clip):
    server = _server(wan_480p_clip)
    result = _run_lifecycle(server, device="GALAXY")
    assert (server.requests[0]["aspect_ratio"], server.requests[0]["size"]) == (
        "16:9",
        "720p",
    )
    assert any("expected 1280x720" in problem for problem in result["problems"])


def test_lifecycle_asks_for_480p_on_a_bh_galaxy_pinned_to_480p(wan_480p_clip):
    result = _run_lifecycle(
        _server(wan_480p_clip),
        device="BLACKHOLE_GALAXY",
        env_vars={"WAN22_RESOLUTION": "480p"},
    )
    assert result["success"], result["problems"]


def test_lifecycle_takes_its_shape_from_the_targets(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    clip = write_clip(
        tmp_path / "tall.mp4", moving_frames(81, 480, 832), faststart=True
    )
    server = _server(clip)
    result = _run_lifecycle(server, targets={"aspect_ratio": "9:16", "size": "480p"})
    assert result["success"], result["problems"]
    assert result["shape"] == [480, 832]


def test_lifecycle_fails_when_the_api_is_not_served():
    # Today's tt-media-server: every generation API route is a 404.
    result = asyncio.run(
        _with_server(
            [],
            lambda url: VideoGenerationI2VTest(
                _TestConfig.create_default(),
                {"poll_interval": 0, "poll_timeout": 5},
                ctx=SimpleNamespace(
                    service_port=0,
                    base_url=url,
                    model_spec=SimpleNamespace(
                        hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers", env_vars={}
                    ),
                    device=SimpleNamespace(name="T3K"),
                    output_path=None,
                ),
            )._run_specific_test_async(),
        )
    )
    assert result["reason"] == "submit_failed"
    assert result["problems"][0].startswith("submit: status 404")


def test_ffmpeg_stream_lines_parse_like_ffprobe():
    stderr = """
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, progressive), 832x480, 16 fps
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 32000 Hz, stereo, fltp
"""
    assert parse_ffmpeg_streams(stderr) == [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "profile": "High",
            "pix_fmt": "yuv420p",
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
            "profile": "LC",
            "pix_fmt": None,
        },
    ]


def test_suite_runs_the_contract_before_the_lifecycle():
    suites = json.loads(
        (
            Path(__file__).resolve().parents[2] / "test_module/test_suites/video.json"
        ).read_text()
    )
    [wan_i2v] = [m for m in suites["test_matrices"] if m["models"] == ["wan_i2v"]]
    templates = [case["template"] for case in wan_i2v["test_cases"]]
    assert templates.index("WanI2VCreateContractTest") < templates.index(
        "VideoGenerationI2VTest"
    )
