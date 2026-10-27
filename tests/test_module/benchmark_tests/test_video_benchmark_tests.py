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


@pytest.mark.parametrize(
    "resolution,device,expected",
    [
        ("480p", "BLACKHOLE_GALAXY", (81, 832, 480)),
        ("720p", "T3K", (81, 1280, 720)),
    ],
)
def test_deployment_resolution_overrides_the_mesh_default(resolution, device, expected):
    ctx = _ctx("Wan-AI/Wan2.2-I2V-A14B-Diffusers", device)
    ctx.model_spec.env_vars = {"WAN22_RESOLUTION": resolution}
    assert mod._expected_video_shape(ctx) == expected


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
    monkeypatch.setattr(
        mod, "_check_video_output", lambda ctx, path, shape=None: problem
    )
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    [result] = mod._run_video_generation_benchmark(ctx, num_calls=1)
    assert result.status is expected_status


def test_only_a_failed_benchmark_clip_is_kept(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"mp4")
    monkeypatch.setattr(
        mod, "_generate_video", lambda *a, **k: (True, 10.0, "job-1", str(clip))
    )
    ctx = _ctx("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    ctx.output_path = str(tmp_path / "out")
    kept = tmp_path / "out/videos/failed/benchmarks/clip.mp4"

    monkeypatch.setattr(mod, "_check_video_output", lambda ctx, path, shape=None: None)
    mod._run_video_generation_benchmark(ctx, num_calls=1)
    assert not kept.exists()

    monkeypatch.setattr(
        mod, "_check_video_output", lambda ctx, path, shape=None: "black"
    )
    mod._run_video_generation_benchmark(ctx, num_calls=1)
    assert kept.read_bytes() == b"mp4"


WAN_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
SHAPES = [(832, 480), (480, 832), (1280, 720), (720, 1280)]


def test_bh_galaxy_wan_i2v_has_a_target_per_shape():
    from test_module._test_common.target_check import (
        get_performance_targets,
        performance_target_shapes,
    )

    assert performance_target_shapes(WAN_I2V, "blackhole_galaxy") == SHAPES
    ttft = {
        shape: get_performance_targets(WAN_I2V, "blackhole_galaxy", shape).ttft_ms
        for shape in SHAPES
    }
    assert ttft == {
        (832, 480): 43400,
        (480, 832): 43400,
        (1280, 720): 144300,
        (720, 1280): 144300,
    }
    # Devices without per-shape entries keep one benchmark at the served shape.
    assert performance_target_shapes(WAN_I2V, "t3k") == []
    assert get_performance_targets(WAN_I2V, "t3k").ttft_ms == 540000
    assert get_performance_targets(WAN_I2V, "t3k", (832, 480)).ttft_ms is None


@pytest.mark.parametrize(
    "shape,aspect_ratio,size",
    [((832, 480), "16:9", "480p"), ((720, 1280), "9:16", "720p")],
)
def test_i2v_payload_asks_for_the_shape(shape, aspect_ratio, size):
    from test_module._test_common.video_generation_routing import (
        build_video_generation_payload,
    )

    payload = build_video_generation_payload(
        prompt="p",
        num_inference_steps=40,
        model_name=WAN_I2V,
        image_b64="ZmFrZQ==",
        shape=shape,
    )
    assert (payload["aspect_ratio"], payload["size"]) == (aspect_ratio, size)
    plain = build_video_generation_payload(
        prompt="p", num_inference_steps=40, model_name=WAN_I2V, image_b64="ZmFrZQ=="
    )
    assert "aspect_ratio" not in plain and "size" not in plain


def test_requested_shape_overrides_the_deployment_shape(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("imageio_ffmpeg")
    clip = write_clip(tmp_path / "ok.mp4", moving_frames(81, 832, 480))
    ctx = _ctx(WAN_I2V, "BLACKHOLE_GALAXY")
    assert mod._check_video_output(ctx, str(clip), (832, 480)) is None
    assert "832x480, expected 480x832" in mod._check_video_output(
        ctx, str(clip), (480, 832)
    )


def test_every_shape_runs_and_one_unserved_shape_fails_the_block(monkeypatch):
    # The deployment serves 1280x720 only: the other three shapes come back
    # the wrong size, so their generations fail and so does the block.
    calls = []

    def fake_generate(ctx, prompt, num_inference_steps, image_b64=None, shape=None):
        calls.append(shape)
        return True, 140.0, "job", "/tmp/x.mp4"

    monkeypatch.setattr(mod, "_generate_video", fake_generate)
    monkeypatch.setattr(mod, "_load_fixture_image_base64", lambda: "ZmFrZQ==")
    monkeypatch.setattr(
        mod,
        "_check_video_output",
        lambda ctx, path, shape=None: None if shape == (1280, 720) else "wrong size",
    )
    monkeypatch.setattr(mod, "save_clip", lambda *a, **k: None)
    ctx = _ctx(WAN_I2V, "BLACKHOLE_GALAXY")
    ctx.model_spec.cli_args = {"device": "blackhole_galaxy"}

    block = mod._run_video_benchmark_by_shape(ctx, 2, SHAPES)

    assert calls == [s for s in SHAPES for _ in range(2)]
    rows = {r["shape"]: r for r in block.data["records"]}
    assert list(rows) == ["832x480", "480x832", "1280x720", "720x1280"]
    assert rows["1280x720"]["num_successful"] == 2
    assert rows["1280x720"]["ttft"] == 140.0
    assert rows["832x480"]["num_successful"] == 0
    assert block.data["num_requests"] == 8
    assert block.data["status"] == "fail"
    target = block.data["target_checks"]["target"]
    assert target["ttft_1280x720"] == 144300
    assert target["ttft_1280x720_check"] == mod.ReportCheckTypes.PASS
    assert target["ttft_832x480_check"] == mod.ReportCheckTypes.NA
