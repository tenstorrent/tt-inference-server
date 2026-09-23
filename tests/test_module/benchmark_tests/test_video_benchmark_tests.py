# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for video benchmark dispatch guards and T2V/I2V routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import SkipTest
from test_module.benchmark_tests import video_benchmark_tests as mod


def _ctx(model_name):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name=model_name, hf_model_repo=model_name),
        device=SimpleNamespace(name="t3k"),
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


def test_minimax_h3_has_fixed_step_profile():
    from test_module.benchmark_tests import video_benchmark_tests as mod

    assert mod.VIDEO_INFERENCE_STEPS["MiniMaxAI/MiniMax-H3"] == 50
    assert mod.is_minimax_h3_model("MiniMaxAI/MiniMax-H3")
    assert mod.is_minimax_h3_model("MiniMax-H3")
    assert not mod.is_minimax_h3_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")


def test_minimax_h3_generation_sends_shape_fields_not_steps(monkeypatch):
    from test_module.benchmark_tests import video_benchmark_tests as mod

    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-h3"}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    def _fake_poll(ctx, job_id, headers, timeout=mod.DEFAULT_VIDEO_TIMEOUT_SECONDS):
        captured["poll_timeout"] = timeout
        return "/tmp/out.mp4"

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(
            model_name="MiniMax-H3", hf_model_repo="MiniMaxAI/MiniMax-H3"
        ),
        base_url="http://localhost:8000",
    )
    ok, _elapsed, job_id, path = mod._generate_video(
        ctx, prompt="a fox", num_inference_steps=50
    )
    assert ok and job_id == "job-h3" and path == "/tmp/out.mp4"
    assert captured["url"].endswith("v1/videos/generations")
    assert captured["payload"] == {
        "prompt": "a fox",
        "aspect_ratio": "16:9",
        "duration_seconds": 5,
        "seed": 0,
    }
    assert "num_inference_steps" not in captured["payload"]
    assert captured["poll_timeout"] == mod.MINIMAX_H3_VIDEO_TIMEOUT_SECONDS
