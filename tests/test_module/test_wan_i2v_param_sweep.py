# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Wan2.2 I2V parameter sweep, graded on real (synthetic) clips."""

from __future__ import annotations

import asyncio
import io
import uuid
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module.load_param_tests import video_generation_i2v_param_test as sweep
from tests.test_module.video_clips import moving_frames, write_clip

np = pytest.importorskip("numpy")
pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("imageio.v3")
Image = pytest.importorskip("PIL.Image")
web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

IMAGE_BYTES = sweep.FIXTURE_IMAGE_PATH.read_bytes()


def _clip(path, *, offset=0, image_at=0, size=(64, 48), frames=24):
    """A moving clip whose frame ``image_at`` is the conditioning image."""
    width, height = size
    clip = moving_frames(frames, width, height)
    clip = [np.roll(frame, offset * 8, axis=1) for frame in clip]
    image = Image.open(io.BytesIO(IMAGE_BYTES)).convert("RGB").resize(size)
    clip[image_at] = np.asarray(image)
    return str(write_clip(path, clip))


@pytest.fixture
def clips(tmp_path):
    return {
        "a": _clip(tmp_path / "a.mp4"),
        "b": _clip(tmp_path / "b.mp4", offset=3),
        "c": _clip(tmp_path / "c.mp4", offset=5),
        "d": _clip(tmp_path / "d.mp4", offset=7),
        "image_at_4": _clip(tmp_path / "e.mp4", offset=9, image_at=4),
    }


def _jobs(clips, **overrides):
    jobs = {
        "default": {"elapsed": 20.0, "video_path": clips["a"]},
        "duplicate_default": {"elapsed": 20.0, "video_path": clips["a"]},
        "seed_123": {"elapsed": 20.0, "video_path": clips["b"]},
        "different_prompt": {"elapsed": 20.0, "video_path": clips["c"]},
        "fewer_steps": {"elapsed": 11.0, "video_path": clips["d"]},
        "no_negative_prompt": {"elapsed": 20.0, "video_path": clips["b"]},
        "frame_pos_4": {"elapsed": 20.0, "video_path": clips["image_at_4"]},
    }
    jobs.update(overrides)
    return jobs


def _by_name(checks):
    return {check["check"]: check for check in checks}


def test_a_working_sweep_passes_every_check(clips):
    checks = _by_name(sweep.evaluate_sweep(_jobs(clips), IMAGE_BYTES))
    failed = [c for c in checks.values() if not c["passed"]]
    assert not failed, failed
    assert {
        "seed_123_differs_from_default",
        "different_prompt_differs_from_default",
        "fewer_steps_differs_from_default",
        "fewer_steps_is_faster",
        "identical_request_reproduces_clip",
        "no_negative_prompt_differs_from_default",
        "conditioning_lands_at_frame_pos",
    } <= set(checks)


def test_a_seed_that_changes_nothing_fails_a_gated_check(clips):
    jobs = _jobs(clips, seed_123={"elapsed": 20.0, "video_path": clips["a"]})
    check = _by_name(sweep.evaluate_sweep(jobs, IMAGE_BYTES))[
        "seed_123_differs_from_default"
    ]
    assert check["gated"] and not check["passed"]


def test_fewer_steps_that_are_not_faster_fail(clips):
    jobs = _jobs(clips, fewer_steps={"elapsed": 25.0, "video_path": clips["d"]})
    check = _by_name(sweep.evaluate_sweep(jobs, IMAGE_BYTES))["fewer_steps_is_faster"]
    assert check["gated"] and not check["passed"]


def test_ungated_checks_report_without_failing(clips):
    jobs = _jobs(
        clips,
        duplicate_default={"elapsed": 20.0, "video_path": clips["b"]},
        frame_pos_4={"elapsed": 20.0, "video_path": clips["a"]},
    )
    checks = _by_name(sweep.evaluate_sweep(jobs, IMAGE_BYTES))
    for name in (
        "identical_request_reproduces_clip",
        "conditioning_lands_at_frame_pos",
    ):
        assert not checks[name]["passed"] and not checks[name]["gated"]


def test_a_failed_job_is_a_gated_failure_and_skips_its_comparisons(clips):
    jobs = _jobs(clips, seed_123={"problem": "job did not complete"})
    checks = _by_name(sweep.evaluate_sweep(jobs, IMAGE_BYTES))
    assert not checks["seed_123_completed"]["passed"]
    assert "seed_123_differs_from_default" not in checks


def test_variants_change_one_thing_each():
    variants = sweep.build_variants("img", 20)
    base = variants["default"]
    assert variants["duplicate_default"] == base
    assert variants["fewer_steps"]["num_inference_steps"] == 10
    assert variants["frame_pos_4"]["image_prompts"][0]["frame_pos"] == 4
    assert "negative_prompt" not in variants["no_negative_prompt"]
    for name, payload in variants.items():
        changed = {
            k for k in base.keys() | payload.keys() if base.get(k) != payload.get(k)
        }
        assert len(changed) <= 1, (name, changed)


class _SweepServer:
    """Video V1 API that returns a clip chosen by what the request asked for."""

    def __init__(self, pick):
        self.pick = pick
        self.jobs: dict[str, dict] = {}

    def routes(self):
        return [
            web.post("/v1/videos/generations/i2v", self.submit),
            web.get("/v1/videos/generations/{job_id}", self.metadata),
            web.get("/v1/videos/generations/{job_id}/download", self.download),
        ]

    async def submit(self, request):
        body = await request.json()
        # Generation time follows the step count, so fewer steps finish sooner.
        await asyncio.sleep(body["num_inference_steps"] * 0.01)
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "job_type": "video",
            "status": "completed",
            "request_parameters": body,
            "clip": self.pick(body),
        }
        return web.json_response({"id": job_id}, status=202)

    async def metadata(self, request):
        job = self.jobs[request.match_info["job_id"]]
        return web.json_response({k: v for k, v in job.items() if k != "clip"})

    async def download(self, request):
        clip = self.jobs[request.match_info["job_id"]]["clip"]
        with open(clip, "rb") as f:
            return web.Response(body=f.read(), content_type="video/mp4")


def _run_sweep(pick, output_path=None) -> dict:
    async def body():
        app = web.Application()
        app.add_routes(_SweepServer(pick).routes())
        async with test_utils.TestServer(app) as server:
            ctx = SimpleNamespace(
                service_port=0,
                base_url=str(server.make_url("")).rstrip("/"),
                model_spec=SimpleNamespace(
                    hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers", env_vars={}
                ),
                device=SimpleNamespace(name="T3K"),
                output_path=output_path,
            )
            test = sweep.VideoGenerationI2VParamTest(
                _TestConfig.create_default(), {"poll_interval": 0}, ctx=ctx
            )
            return await test._run_specific_test_async()

    return asyncio.run(body())


@pytest.fixture(scope="module")
def wan_clips(tmp_path_factory):
    """81-frame 832x480 clips, the shape a T3K deployment serves."""
    root = tmp_path_factory.mktemp("sweep")

    def make(name, **kwargs):
        return _clip(root / f"{name}.mp4", size=(832, 480), frames=81, **kwargs)

    return {
        "a": make("a"),
        "b": make("b", offset=3),
        "c": make("c", offset=5),
        "d": make("d", offset=7),
        "image_at_4": make("e", offset=9, image_at=4),
    }


def test_sweep_passes_when_every_parameter_changes_the_clip(wan_clips, tmp_path):
    def pick(body):
        if body["image_prompts"][0]["frame_pos"] == 4:
            return wan_clips["image_at_4"]
        if body["seed"] == 123:
            return wan_clips["b"]
        if body["prompt"] == sweep.ALT_PROMPT:
            return wan_clips["c"]
        if body["num_inference_steps"] == 10:
            return wan_clips["d"]
        return wan_clips["a"]

    result = _run_sweep(pick, output_path=tmp_path)
    assert result["success"], [c for c in result["checks"] if not c["passed"]]
    # Only no_negative_prompt (ungated) can fail here, so at most its pair is kept.
    saved_dir = tmp_path / "videos/failed/spec_tests/param_sweep"
    saved = {p.name for p in saved_dir.iterdir()} if saved_dir.exists() else set()
    assert saved <= {"default.mp4", "no_negative_prompt.mp4"}
    assert set(result["jobs"]) == set(sweep.build_variants("x", 20))


def test_sweep_fails_when_the_server_ignores_the_parameters(wan_clips, tmp_path):
    result = _run_sweep(lambda body: wan_clips["a"], output_path=tmp_path)
    assert not result["success"]
    # The clips behind the failing checks are kept for upload, and only those.
    saved = {
        p.name for p in (tmp_path / "videos/failed/spec_tests/param_sweep").iterdir()
    }
    assert {"default.mp4", "seed_123.mp4", "different_prompt.mp4"} <= saved
    assert "duplicate_default.mp4" not in saved
    failed = {c["check"] for c in result["checks"] if c["gated"] and not c["passed"]}
    assert "seed_123_differs_from_default" in failed
    assert "different_prompt_differs_from_default" in failed
