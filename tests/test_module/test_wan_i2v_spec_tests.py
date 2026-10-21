# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Wan2.2 I2V spec tests (create contract, lifecycle) against an in-process server."""

from __future__ import annotations

import ast
import asyncio
import base64
import io
import json
import re
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module.load_param_tests import wan_i2v_create_contract_test as contract
from test_module.load_param_tests.video_generation_i2v_test import (
    VideoGenerationI2VTest,
)
from tests.test_module.video_clips import gradient, moving_frames, write_clip

web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

MEDIA_SERVER = Path(__file__).resolve().parents[2] / "tt-media-server"
API_KEY = "test-key"


def _module_constants(path: Path) -> dict:
    """Top-level ``NAME = <literal>`` assignments, read without importing the module."""
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return constants


def test_contract_bounds_match_the_server():
    server = _module_constants(MEDIA_SERVER / "config/constants.py")
    assert contract.WAN22_NUM_FRAMES == server["WAN22_NUM_FRAMES"]
    assert contract.MIN_VIDEO_INFERENCE_STEPS == server["MIN_VIDEO_INFERENCE_STEPS"]
    assert contract.MAX_VIDEO_INFERENCE_STEPS == server["MAX_VIDEO_INFERENCE_STEPS"]


def test_default_api_key_matches_the_server():
    checker = (MEDIA_SERVER / "security/api_key_checker.py").read_text()
    default = re.search(r'os\.getenv\("API_KEY", "([^"]+)"\)', checker).group(1)
    assert contract.DEFAULT_API_KEY == default


def test_tiny_png_decodes():
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.open(io.BytesIO(base64.b64decode(contract.TINY_PNG_BASE64)))
    assert image.size == (1, 1)


def test_case_names_are_unique():
    names = [case.name for case in contract.cases()]
    assert len(names) == len(set(names))


async def _with_server(routes, body):
    """Run ``body(base_url)`` against an aiohttp app serving ``routes``."""
    app = web.Application()
    app.add_routes(routes)
    async with test_utils.TestServer(app) as server:
        return await body(str(server.make_url("")).rstrip("/"))


def _auth_error(request):
    auth = request.headers.get("Authorization")
    if auth is None:
        return web.json_response({"detail": "Not authenticated"}, status=403)
    if auth != f"Bearer {API_KEY}":
        return web.json_response({"detail": "Invalid or missing API Key"}, status=401)
    return None


def _contract_routes(*, accept_everything: bool = False, refuse_t2v: bool = True):
    """A server that accepts exactly the valid I2V body and refuses the rest."""

    async def i2v(request):
        if (error := _auth_error(request)) is not None:
            return error
        body = await request.json()
        if accept_everything or body == contract.valid_payload():
            return web.json_response({"id": str(uuid.uuid4())}, status=202)
        return web.json_response({"detail": [{"msg": "invalid"}]}, status=422)

    async def t2v(request):
        if refuse_t2v:
            return web.json_response(
                {"detail": "requires image conditioning"}, status=422
            )
        return web.json_response({"detail": [{"msg": "prompt missing"}]}, status=422)

    return [
        web.post(contract.I2V_CREATE_PATH, i2v),
        web.post(contract.T2V_CREATE_PATH, t2v),
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


def test_contract_fails_when_invalid_requests_create_jobs():
    checks = _by_check(_run_contract(accept_everything=True))
    assert not checks["duplicate_frame_pos"]["passed"]
    assert "created job" in checks["duplicate_frame_pos"]["message"]
    # Auth still refuses, so the auth checks keep passing.
    assert checks["invalid_bearer_authentication"]["passed"]


def test_contract_tells_a_route_refusal_from_field_validation():
    checks = _by_check(_run_contract(refuse_t2v=False))
    refusal = checks["text_to_video_route_is_refused"]
    assert not refusal["passed"]
    assert "not a route refusal" in refusal["message"]


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


class _FakeVideoServer:
    """Just enough of the video V1 API for one I2V job's lifecycle."""

    def __init__(self, clip: Path, *, list_job: bool = True, echo: bool = True):
        self.clip = clip
        self.list_job = list_job
        self.echo = echo
        self.job: dict | None = None

    def routes(self):
        return [
            web.post("/v1/videos/generations/i2v", self.submit),
            web.get("/v1/videos/generations/{job_id}", self.metadata),
            web.get("/v1/videos/jobs", self.jobs),
            web.get("/v1/videos/generations/{job_id}/download", self.download),
        ]

    async def submit(self, request):
        body = await request.json()
        request_parameters = body if self.echo else {**body, "seed": None}
        self.job = {
            "id": str(uuid.uuid4()),
            "job_type": "video",
            "status": "queued",
            "request_parameters": request_parameters,
        }
        return web.json_response({"id": self.job["id"]}, status=202)

    async def metadata(self, request):
        job, self.job["status"] = dict(self.job), "completed"
        return web.json_response(job)

    async def jobs(self, request):
        return web.json_response([self.job] if self.list_job else [])

    async def download(self, request):
        return web.Response(body=self.clip.read_bytes(), content_type="video/mp4")


def _run_lifecycle(
    server: _FakeVideoServer, device: str = "T3K", env_vars: dict | None = None
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
        )
        test = VideoGenerationI2VTest(
            _TestConfig.create_default(),
            {"poll_interval": 0, "poll_timeout": 30},
            ctx=ctx,
        )
        return await test._run_specific_test_async()

    return asyncio.run(_with_server(server.routes(), body))


@pytest.fixture
def wan_t3k_clip(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("imageio.v3")
    return write_clip(tmp_path / "wan.mp4", moving_frames(81, 832, 480))


def test_lifecycle_passes_for_a_good_job(wan_t3k_clip):
    result = _run_lifecycle(_FakeVideoServer(wan_t3k_clip))
    assert result["success"], result["problems"]


def test_lifecycle_fails_when_the_job_is_not_listed(wan_t3k_clip):
    result = _run_lifecycle(_FakeVideoServer(wan_t3k_clip, list_job=False))
    assert not result["success"]
    assert "job is missing from the jobs list" in result["problems"]


def test_lifecycle_fails_when_metadata_does_not_echo_the_request(wan_t3k_clip):
    result = _run_lifecycle(_FakeVideoServer(wan_t3k_clip, echo=False))
    assert not result["success"]
    assert any("['seed']" in problem for problem in result["problems"])


def test_lifecycle_fails_on_a_frozen_clip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    clip = write_clip(tmp_path / "frozen.mp4", [gradient(832, 480)] * 81)
    result = _run_lifecycle(_FakeVideoServer(clip))
    assert not result["success"]
    assert any(problem.startswith("clip: frozen") for problem in result["problems"])


def test_lifecycle_fails_on_the_wrong_resolution_for_the_mesh(wan_t3k_clip):
    result = _run_lifecycle(_FakeVideoServer(wan_t3k_clip), device="GALAXY")
    assert any("expected 1280x720" in problem for problem in result["problems"])


def test_lifecycle_expects_480p_on_a_bh_galaxy_pinned_to_480p(wan_t3k_clip):
    result = _run_lifecycle(
        _FakeVideoServer(wan_t3k_clip),
        device="BLACKHOLE_GALAXY",
        env_vars={"WAN22_RESOLUTION": "480p"},
    )
    assert result["success"], result["problems"]


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
