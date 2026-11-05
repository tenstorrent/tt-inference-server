# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for video benchmark dispatch guards and T2V/I2V routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import SkipTest
from test_module._test_common import video_generation_routing as routing
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


def _capture_submit(monkeypatch, job_id: str) -> dict:
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": job_id}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", lambda *a, **k: "/tmp/out.mp4")
    return captured


@pytest.mark.parametrize(
    ("device", "size"),
    [("T3K", "480p"), ("GALAXY", "480p"), ("BLACKHOLE_GALAXY", "720p")],
)
def test_wan_i2v_a14b_routes_to_the_generations_api(monkeypatch, device, size):
    # Wan2.2-I2V-A14B is served by the generations API: its own submit path,
    # the image as a data URI, and a size the device's tt_dit preset serves.
    captured = _capture_submit(monkeypatch, "job-123")
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-I2V-A14B-Diffusers"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name=device),
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
    assert captured["url"].endswith("v1/generations/wan-ai/wan2.2-i2v-a14b")
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "negative_prompt": "blurry, low quality, distorted",
        "seed": 42,
        "first_frame": "data:image/jpeg;base64,ZmFrZQ==",
        "size": size,
        "aspect_ratio": "16:9",
    }


def test_other_i2v_variants_keep_the_video_api(monkeypatch):
    captured = _capture_submit(monkeypatch, "job-456")
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Wan2.2-I2V-Lightning"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="GALAXY"),
    )
    mod._generate_video(ctx, prompt="x", num_inference_steps=4, image_b64="ZmFrZQ==")
    assert captured["url"].endswith("v1/videos/generations/i2v")
    assert captured["payload"]["image_prompts"][0]["image"] == "ZmFrZQ=="
    assert captured["payload"]["num_inference_steps"] == 4


@pytest.mark.parametrize(
    ("model_name", "status", "download"),
    [
        (
            "Wan2.2-I2V-A14B-Diffusers",
            "v1/jobs/job-1",
            "http://h/v1/jobs/job-1/content/0",
        ),
        (
            "MiniMax-H3-Ref2VA-LightX2V",
            "v1/jobs/job-1",
            "http://h/v1/jobs/job-1/content/0",
        ),
        (
            "Wan2.2-I2V-Lightning",
            "v1/videos/generations/job-1",
            "http://h/v1/videos/generations/job-1/download",
        ),
    ],
)
def test_job_status_and_download_urls(monkeypatch, model_name, status, download):
    # The generations API's video is behind the completed job's output url.
    requested = []

    class _Result:
        @staticmethod
        def raise_for_status():
            pass

        @staticmethod
        def json():
            return {"outputs": [{"url": "/v1/jobs/job-1/content/0"}]}

    def _fake_get(url, headers, timeout):
        requested.append(url)
        return _Result()

    monkeypatch.setattr(routing.requests, "get", _fake_get)
    assert routing.get_video_job_endpoint(model_name, "job-1") == status
    assert (
        routing.get_video_download_url("http://h", model_name, "job-1", {}) == download
    )
    if routing.uses_generations_api(model_name):
        assert requested == ["http://h/v1/jobs/job-1"]


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
        device=SimpleNamespace(name="T3K"),
    )
    mod._generate_video(ctx, prompt="a sunset", num_inference_steps=40)

    assert captured["url"].endswith("v1/videos/generations")
    assert "image_prompts" not in captured["payload"]


def test_minimax_h3_fl2va_lightx2v_has_its_four_step_profile():
    assert mod.VIDEO_INFERENCE_STEPS["MiniMaxAI/MiniMax-H3-FL2VA-LightX2V"] == 4


def test_minimax_h3_fl2va_generation_sends_only_fields_h3_accepts(monkeypatch):
    # H3 takes no num_inference_steps or negative_prompt; the shape is chosen with
    # duration and an explicit canvas, and the keyframe goes in as the first frame.
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-h3"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    def _fake_poll(ctx, job_id, headers, **kwargs):
        captured["poll_timeout"] = kwargs.get("timeout")
        return "/tmp/h3.mp4"

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="MiniMax-H3-FL2VA-LightX2V"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )
    ok, _elapsed, job_id, _path = mod._generate_video(
        ctx, prompt="a volcano erupting", num_inference_steps=4, image_b64="ZmFrZQ=="
    )

    assert ok is True
    assert job_id == "job-h3"
    assert captured["url"].endswith(
        "v1/generations/tenstorrent/minimax-h3-fl2va-lightx2v"
    )
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "duration": 5,
        "seed": 0,
        "first_frame": "data:image/jpeg;base64,ZmFrZQ==",
    }
    assert captured["poll_timeout"] > mod.DEFAULT_VIDEO_TIMEOUT_SECONDS


def test_minimax_h3_fl2va_benchmark_loads_the_keyframe(monkeypatch):
    seen = {}

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64):
        seen["image_b64"] = image_b64
        return True, 9.0, "job", "/tmp/h3.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    monkeypatch.setattr(mod, "_load_fixture_image_base64", lambda: "a2V5")
    ctx = _ctx("MiniMaxAI/MiniMax-H3-FL2VA-LightX2V")
    ctx.model_spec.model_name = "MiniMax-H3-FL2VA-LightX2V"
    (status,) = mod._run_video_generation_benchmark(ctx, 1)
    assert seen["image_b64"] == "a2V5"
    assert status.num_inference_steps == 4


def test_minimax_h3_ref2va_lightx2v_has_its_four_step_profile():
    assert mod.VIDEO_INFERENCE_STEPS["MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V"] == 4


def test_minimax_h3_ref2va_generation_sends_one_reference_image(monkeypatch):
    # The image goes in as a reference, and the adapter's own canvas is left to the server
    # (aspect_ratio only).
    captured = {}

    class _Resp:
        status_code = 202

        @staticmethod
        def json():
            return {"id": "job-ref"}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    def _fake_poll(ctx, job_id, headers, **kwargs):
        captured["poll_timeout"] = kwargs.get("timeout")
        return "/tmp/ref.mp4"

    monkeypatch.setattr(mod.requests, "post", _fake_post)
    monkeypatch.setattr(mod, "_poll_video_completion", _fake_poll)

    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_name="MiniMax-H3-Ref2VA-LightX2V"),
        base_url="http://localhost:8000",
        device=SimpleNamespace(name="BLACKHOLE_GALAXY"),
    )
    ok, _elapsed, job_id, _path = mod._generate_video(
        ctx, prompt="a volcano erupting", num_inference_steps=4, image_b64="ZmFrZQ=="
    )

    assert ok is True
    assert job_id == "job-ref"
    assert captured["url"].endswith(
        "v1/generations/tenstorrent/minimax-h3-ref2va-lightx2v"
    )
    assert captured["payload"] == {
        "prompt": "a volcano erupting",
        "duration": 5,
        "aspect_ratio": "16:9",
        "seed": 0,
        "images": ["data:image/jpeg;base64,ZmFrZQ=="],
    }
    assert captured["poll_timeout"] > mod.DEFAULT_VIDEO_TIMEOUT_SECONDS


def test_minimax_h3_ref2va_benchmark_loads_the_reference(monkeypatch):
    seen = {}

    def _fake_generate(ctx, prompt, num_inference_steps, image_b64):
        seen["image_b64"] = image_b64
        return True, 9.0, "job", "/tmp/ref.mp4"

    monkeypatch.setattr(mod, "_generate_video", _fake_generate)
    monkeypatch.setattr(mod, "_load_fixture_image_base64", lambda: "a2V5")
    ctx = _ctx("MiniMaxAI/MiniMax-H3-Ref2VA-LightX2V")
    ctx.model_spec.model_name = "MiniMax-H3-Ref2VA-LightX2V"
    (status,) = mod._run_video_generation_benchmark(ctx, 1)
    assert seen["image_b64"] == "a2V5"
    assert status.num_inference_steps == 4


@pytest.mark.parametrize(
    ("model_name", "endpoint", "keys"),
    [
        (
            "MiniMax-H3-FL2VA",
            "v1/generations/minimaxai/minimax-h3-fl2va",
            {"prompt", "duration", "seed", "first_frame"},
        ),
        (
            "MiniMax-H3-Ref2VA",
            "v1/generations/minimaxai/minimax-h3-ref2va",
            {"prompt", "duration", "seed", "aspect_ratio", "images"},
        ),
    ],
)
def test_every_minimax_h3_model_routes_to_the_generations_api(
    model_name, endpoint, keys
):
    assert routing.get_video_generation_submit_endpoint(model_name) == endpoint
    payload = routing.build_video_generation_payload(
        prompt="x", num_inference_steps=50, model_name=model_name, image_b64="ZmFrZQ=="
    )
    assert set(payload) == keys
