# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for video benchmark dispatch guards and T2V/I2V routing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from test_module._test_common import SkipTest
from test_module.benchmark_tests import video_benchmark_tests as mod
from tests.test_module.video_clips import (
    black_frames,
    gradient,
    moving_frames,
    write_clip,
)


def _ctx(model_name, device="t3k"):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model_name, hf_model_repo=model_name),
        device=SimpleNamespace(name=device),
    )


def test_unsupported_video_model_raises_skip():
    # A model with no inference-step profile is a visible, non-blocking SkipTest
    # (which run_media_task maps to SKIP) rather than a KeyError crash.
    with pytest.raises(SkipTest) as exc:
        mod.run_video_benchmark(_ctx("some-unlisted-video-model"))
    assert "not implemented" in str(exc.value)
    assert "some-unlisted-video-model" in str(exc.value)


def test_i2v_model_has_inference_step_profile():
    # I2V benchmarks are implemented: the model must carry a step profile so it
    # is not skipped, mirroring model_performance_reference.json.
    assert "Wan-AI/Wan2.2-I2V-A14B-Diffusers" in mod.VIDEO_INFERENCE_STEPS


def test_i2v_generation_routes_to_i2v_endpoint_and_payload(monkeypatch):
    # An I2V benchmark call must hit the I2V submit endpoint with an image prompt,
    # matching the eval flow and the main-branch behaviour.
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-123"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", lambda *a, **k: "/tmp/out.mp4")

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-I2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
    )
    ok, _elapsed, job_id, video_path = mod._generate_video(
        ctx,
        prompt="a volcano erupting",
        num_inference_steps=40,
        image_b64="ZmFrZQ==",
    )

    assert ok is True
    assert job_id == "job-123"
    assert video_path == "/tmp/out.mp4"
    assert captured["url"].endswith("v1/videos/generations/i2v")
    assert captured["payload"]["image_prompts"][0]["image"] == "ZmFrZQ=="
    assert captured["payload"]["num_inference_steps"] == 40


def test_t2v_generation_routes_to_base_endpoint(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-t2v"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", lambda *a, **k: "/tmp/out.mp4")

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-T2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
    )
    mod._generate_video(ctx, prompt="a sunset", num_inference_steps=40)

    assert captured["url"].endswith("v1/videos/generations")
    assert "image_prompts" not in captured["payload"]


def test_every_dev_wan_i2v_variant_but_prodia_has_a_step_profile():
    # Keys must match the catalog's `weights` names exactly, or the variant is
    # silently skipped again.
    catalog = (
        Path(__file__).resolve().parents[3] / "workflows/model_specs/dev/video.yaml"
    )
    names = {
        entry["weights"][0]
        for entry in yaml.safe_load(catalog.read_text())["templates"]
        if entry["weights"][0].startswith("Wan-AI/Wan2.2-I2V-")
    }
    profiled = {name for name in mod.VIDEO_INFERENCE_STEPS if "-I2V-" in name}
    assert profiled == names - {"Wan-AI/Wan2.2-I2V-A14B-Prodia"}


@pytest.mark.parametrize(
    "model,device,expected",
    [
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "GALAXY", (81, 1280, 720)),
        ("Wan-AI/Wan2.2-I2V-Lightning", "BLACKHOLE_GALAXY", (81, 1280, 720)),
        ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "T3K", (81, 832, 480)),
        ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "P300X2", (81, 832, 480)),
        ("genmo/mochi-1-preview", "T3K", None),
    ],
)
def test_expected_video_shape(model, device, expected):
    assert mod._expected_video_shape(_ctx(model, device)) == expected


class TestCheckVideoOutput:
    """Real clips through the benchmark's output check (T3K: 832x480, 81 frames)."""

    @pytest.fixture(autouse=True)
    def _deps(self):
        pytest.importorskip("numpy")
        pytest.importorskip("imageio_ffmpeg")
        pytest.importorskip("imageio.v3")

    def test_valid_wan_clip_passes(self, tmp_path):
        clip = write_clip(tmp_path / "ok.mp4", moving_frames(81, 832, 480))
        ctx = _ctx("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "T3K")
        assert mod._check_video_output(ctx, str(clip)) is None

    def test_wrong_frame_count_fails(self, tmp_path):
        clip = write_clip(tmp_path / "short.mp4", moving_frames(80, 832, 480))
        ctx = _ctx("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "T3K")
        assert "80 frames, expected 81" in mod._check_video_output(ctx, str(clip))

    def test_wrong_resolution_fails(self, tmp_path):
        clip = write_clip(tmp_path / "small.mp4", moving_frames(81, 832, 480))
        ctx = _ctx("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "GALAXY")
        assert "832x480, expected 1280x720" in mod._check_video_output(ctx, str(clip))

    def test_frozen_clip_fails(self, tmp_path):
        clip = write_clip(tmp_path / "frozen.mp4", [gradient(832, 480)] * 81)
        ctx = _ctx("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "T3K")
        assert "frozen" in mod._check_video_output(ctx, str(clip))

    def test_black_clip_fails_for_unpinned_models_too(self, tmp_path):
        clip = write_clip(tmp_path / "black.mp4", black_frames(24))
        assert "black" in mod._check_video_output(
            _ctx("genmo/mochi-1-preview"), str(clip)
        )

    def test_undecodable_clip_fails(self, tmp_path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"not a video")
        problem = mod._check_video_output(_ctx("genmo/mochi-1-preview"), str(junk))
        assert problem.startswith("clip could not be checked")


@pytest.mark.parametrize("problem,expected_status", [(None, True), ("black", False)])
def test_failed_output_check_marks_the_generation_failed(
    monkeypatch, problem, expected_status
):
    monkeypatch.setattr(
        mod, "_generate_video", lambda *a, **k: (True, 10.0, "job-1", "/tmp/x.mp4")
    )
    monkeypatch.setattr(mod, "_check_video_output", lambda ctx, path: problem)
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    [result] = mod._run_video_generation_benchmark(ctx, num_calls=1)
    assert result.status is expected_status
