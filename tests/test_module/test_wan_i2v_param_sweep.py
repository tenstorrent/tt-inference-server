# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Wan2.2 I2V parameter sweep, graded on real (synthetic) clips."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest

from test_module._test_common import TestConfig as _TestConfig
from test_module._test_common.large_images import near_limit_image
from test_module._test_common.video_generation_routing import (
    FIXTURE_IMAGE_PATH,
    WAN22_I2V_SHAPES,
    WAN22_NEGATIVE_PROMPT,
)
from test_module.load_param_tests import video_generation_i2v_param_test as sweep
from tests.test_module.fake_generation_api import CHOSEN_SEED, FakeJobsServer
from tests.test_module.video_clips import moving_frames, write_clip

np = pytest.importorskip("numpy")
pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("imageio.v3")
Image = pytest.importorskip("PIL.Image")
web = pytest.importorskip("aiohttp.web")
test_utils = pytest.importorskip("aiohttp.test_utils")

IMAGE_BYTES = FIXTURE_IMAGE_PATH.read_bytes()
LAST_IMAGE_BYTES = sweep.LAST_FRAME_IMAGE_PATH.read_bytes()


def _clip(path, *, offset=0, size=(64, 48), images=None):
    """A moving 81-frame clip; ``images`` maps frame index -> image bytes there."""
    width, height = size
    clip = moving_frames(81, width, height)
    clip = [np.roll(frame, offset * 8, axis=1) for frame in clip]
    for index, data in (images if images is not None else {0: IMAGE_BYTES}).items():
        image = Image.open(io.BytesIO(data)).convert("RGB").resize(size)
        clip[index] = np.asarray(image)
    return str(write_clip(path, clip, faststart=True))


@pytest.fixture
def clips(tmp_path):
    return {
        "a": _clip(tmp_path / "a.mp4"),
        "b": _clip(tmp_path / "b.mp4", offset=3),
        "c": _clip(tmp_path / "c.mp4", offset=5),
        "d": _clip(tmp_path / "d.mp4", offset=7),
        "e": _clip(tmp_path / "e.mp4", offset=9),
        "ends_on_last": _clip(
            tmp_path / "f.mp4",
            offset=11,
            images={0: IMAGE_BYTES, sweep.LAST_FRAME: LAST_IMAGE_BYTES},
        ),
        "starts_on_last": _clip(
            tmp_path / "g.mp4", offset=13, images={0: LAST_IMAGE_BYTES}
        ),
    }


def _jobs(clips, **overrides):
    def job(clip):
        return {"elapsed": 20.0, "seed": 42, "video_path": clips[clip]}

    jobs = {
        "default": job("a"),
        "seed_123": job("b"),
        "different_prompt": job("c"),
        "custom_negative_prompt": job("d"),
        "wan_negative_prompt": job("a"),
        "no_seed": job("e"),
        "no_seed_replay": job("e"),
        "last_frame": job("ends_on_last"),
        # Conditioned on the last-frame image, passed as the large image below.
        "large_image": job("starts_on_last"),
    }
    jobs.update({name: job(clip) for name, clip in overrides.items()})
    return jobs


def _evaluate(jobs):
    checks = sweep.evaluate_sweep(
        jobs,
        IMAGE_BYTES,
        last_image_bytes=LAST_IMAGE_BYTES,
        large_image_bytes=LAST_IMAGE_BYTES,
    )
    return {check["check"]: check for check in checks}


RECORDED = {
    "custom_negative_prompt_differs_from_default",
    "default_negative_prompt_is_wans",
    "reported_seed_reproduces_clip",
    "last_frame_ends_on_its_image",
    "large_image_starts_from_its_image",
}


def test_a_working_sweep_passes_every_check(clips):
    checks = _evaluate(_jobs(clips))
    failed = [c for c in checks.values() if not c["passed"]]
    assert not failed, failed
    assert {
        "seed_123_differs_from_default",
        "different_prompt_differs_from_default",
        *RECORDED,
    } <= set(checks)
    assert {name for name, c in checks.items() if not c["gated"]} == RECORDED


def test_a_seed_that_changes_nothing_fails_a_gated_check(clips):
    check = _evaluate(_jobs(clips, seed_123="a"))["seed_123_differs_from_default"]
    assert check["gated"] and not check["passed"]


def test_ungated_checks_report_without_failing(clips):
    checks = _evaluate(
        _jobs(
            clips,
            custom_negative_prompt="a",
            wan_negative_prompt="b",
            no_seed_replay="b",
            last_frame="a",
            large_image="a",
        )
    )
    for name in RECORDED:
        assert not checks[name]["passed"] and not checks[name]["gated"], name


def test_a_failed_job_is_a_gated_failure_and_skips_its_comparisons(clips):
    jobs = _jobs(clips)
    jobs["seed_123"] = {"problem": "job ended failed"}
    checks = _evaluate(jobs)
    assert not checks["seed_123_completed"]["passed"]
    assert "seed_123_differs_from_default" not in checks


def test_variants_change_one_thing_each():
    variants = sweep.build_variants(
        "img", (832, 480), last_image_uri="last", large_image_uri="large"
    )
    base = variants["default"]
    assert "negative_prompt" not in base
    assert variants["wan_negative_prompt"]["negative_prompt"] == WAN22_NEGATIVE_PROMPT
    assert "seed" not in variants["no_seed"]
    assert variants["last_frame"]["last_frame"] == "last"
    # The default and the shape variants cover every output shape once.
    shapes = [
        (p["aspect_ratio"], p["size"])
        for name, p in variants.items()
        if name == "default" or name.startswith("shape_")
    ]
    assert sorted(shapes) == sorted(WAN22_I2V_SHAPES.values())
    for name, payload in variants.items():
        changed = {
            k for k in base.keys() | payload.keys() if base.get(k) != payload.get(k)
        }
        if name.startswith("shape_"):
            assert changed <= {"aspect_ratio", "size"}, (name, changed)
        else:
            assert len(changed) <= 1, (name, changed)
    # No knob the API does not have.
    assert not any("num_inference_steps" in p for p in variants.values())


def _run_sweep(server, output_path=None) -> dict:
    async def body():
        # The large-image variant posts ~40M characters; aiohttp's default is 1 MiB.
        app = web.Application(client_max_size=128 * 1024 * 1024)
        app.add_routes(server.routes())
        async with test_utils.TestServer(app) as test_server:
            ctx = SimpleNamespace(
                service_port=0,
                base_url=str(test_server.make_url("")).rstrip("/"),
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
    """81-frame clips at each output shape; 832x480 is a T3K's default."""
    root = tmp_path_factory.mktemp("sweep")

    def make(name, size=(832, 480), **kwargs):
        return _clip(root / f"{name}.mp4", size=size, **kwargs)

    return {
        "a": make("a"),
        "b": make("b", offset=3),
        "c": make("c", offset=5),
        "d": make("d", offset=7),
        "e": make("e", offset=9),
        "last": make("f", offset=11, images={0: IMAGE_BYTES, 80: LAST_IMAGE_BYTES}),
        "large": make("g", offset=13, images={0: near_limit_image()}),
        **{
            shape: make(f"{shape[0]}x{shape[1]}", size=shape)
            for shape in WAN22_I2V_SHAPES
            if shape != (832, 480)
        },
    }


def _pick(wan_clips):
    """Return the clip each request asked for, as a working server would."""

    def pick(body):
        shape = sweep.requested_shape(body)
        if shape != (832, 480):
            return wan_clips[shape]
        if len(body["first_frame"]) > 1_000_000:
            return wan_clips["large"]
        if "last_frame" in body:
            return wan_clips["last"]
        if body.get("seed", CHOSEN_SEED) == CHOSEN_SEED:
            return wan_clips["e"]
        if body["seed"] == 123:
            return wan_clips["b"]
        if body["prompt"] == sweep.ALT_PROMPT:
            return wan_clips["c"]
        if body.get("negative_prompt") == sweep.NEGATIVE_PROMPT:
            return wan_clips["d"]
        return wan_clips["a"]

    return pick


def test_sweep_passes_when_every_parameter_takes_effect(wan_clips, tmp_path):
    server = FakeJobsServer(_pick(wan_clips))
    result = _run_sweep(server, output_path=tmp_path)
    assert result["success"], [c for c in result["checks"] if not c["passed"]]
    checks = {c["check"]: c for c in result["checks"]}
    failed = [c for c in checks.values() if not c["passed"]]
    assert not failed, failed
    assert not (tmp_path / "videos/failed").exists()
    variants = sweep.build_variants(
        "x", (832, 480), last_image_uri="l", large_image_uri="g"
    )
    assert set(result["jobs"]) == {*variants, "no_seed_replay"}
    # The replay asked for the seed the seedless job reported.
    assert result["jobs"]["no_seed"]["seed"] == CHOSEN_SEED
    assert server.requests[-1]["seed"] == CHOSEN_SEED
    # Nothing is left behind on the server but finished jobs.
    assert all(job["status"] == "completed" for job in server.jobs.values())


def test_sweep_fails_when_the_server_ignores_the_parameters(wan_clips, tmp_path):
    server = FakeJobsServer(lambda body: wan_clips["a"])
    result = _run_sweep(server, output_path=tmp_path)
    assert not result["success"]
    failed = {c["check"] for c in result["checks"] if c["gated"] and not c["passed"]}
    assert {
        "seed_123_differs_from_default",
        "different_prompt_differs_from_default",
        "shape_9x16_480p_completed",
        "shape_16x9_720p_completed",
        "shape_9x16_720p_completed",
    } <= failed
    # The clips behind the failing checks are kept for upload.
    saved = {
        p.name for p in (tmp_path / "videos/failed/spec_tests/param_sweep").iterdir()
    }
    assert {
        "default.mp4",
        "seed_123.mp4",
        "different_prompt.mp4",
        "shape_9x16_480p.mp4",
    } <= saved
