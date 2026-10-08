# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The LightX2V H3 runners: a Turbo adapter, 5 grid points, and the adapter's own shift."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# conftest stubs ``tt_model_runners.dit_runners``; reuse the real module that file loads from disk.
from tests.test_video_stage_metrics_wiring import dit_runners

LIGHTX2V_RUNNERS = [
    ("TTMiniMaxH3FL2VALightX2VRunner", "TTMiniMaxH3FL2VARunner", "t2va"),
    ("TTMiniMaxH3Ref2VALightX2VRunner", "TTMiniMaxH3Ref2VARunner", "ref2va"),
]

# (video, audio) shift each adapter was distilled at, from the lightx2v model card.
ADAPTER_SHIFTS = {
    "TTMiniMaxH3FL2VALightX2VRunner": (6.0, 3.0),
    "TTMiniMaxH3Ref2VALightX2VRunner": (12.0, 3.0),
}


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    path = tmp_path / "turbo.safetensors"
    path.write_bytes(b"")
    monkeypatch.setenv("MINIMAX_H3_LORA_PATH", str(path))
    monkeypatch.delenv("MINIMAX_H3_VIDEO_SHIFT", raising=False)
    monkeypatch.delenv("MINIMAX_H3_AUDIO_SHIFT", raising=False)
    return str(path)


def _runner(class_name):
    cls = getattr(dit_runners, class_name)
    runner = cls.__new__(cls)
    runner.device_id = "0"
    runner.logger = Mock()
    runner.ttnn_device = Mock()
    runner.settings = SimpleNamespace(model_weights_path=None)
    return runner


@pytest.mark.parametrize(("name", "base", "task"), LIGHTX2V_RUNNERS)
def test_keeps_the_base_task_and_runs_five_grid_points(name, base, task):
    cls = getattr(dit_runners, name)
    base_cls = getattr(dit_runners, base)
    assert issubclass(cls, base_cls)
    assert cls.pipeline_task == task
    assert cls.dit_fsdp == base_cls.dit_fsdp
    assert cls.num_inference_steps == 5


@pytest.mark.parametrize(("name", "base", "task"), LIGHTX2V_RUNNERS)
def test_pipeline_gets_the_adapter_and_its_shift(
    name, base, task, adapter, monkeypatch
):
    turbo = Mock()
    monkeypatch.setattr(
        dit_runners._MiniMaxH3LightX2VMixin, "_pipeline_factory", lambda self: turbo
    )
    _runner(name).create_pipeline()

    turbo.create_pipeline.assert_called_once()
    kwargs = turbo.create_pipeline.call_args.kwargs
    assert kwargs["task"] == task
    assert kwargs["lora_path"] == adapter
    assert (kwargs["video_shift"], kwargs["audio_shift"]) == ADAPTER_SHIFTS[name]


@pytest.mark.parametrize(("name", "base", "task"), LIGHTX2V_RUNNERS)
def test_shift_env_overrides_the_default(name, base, task, adapter, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_VIDEO_SHIFT", "12")
    monkeypatch.setenv("MINIMAX_H3_AUDIO_SHIFT", "2.5")
    kwargs = _runner(name)._create_pipeline_kwargs()
    assert (kwargs["video_shift"], kwargs["audio_shift"]) == (12.0, 2.5)


@pytest.mark.parametrize(("name", "base", "task"), LIGHTX2V_RUNNERS)
@pytest.mark.parametrize("lora_path", [None, "/nonexistent/turbo.safetensors"])
def test_refuses_to_start_without_an_adapter_file(
    name, base, task, lora_path, monkeypatch
):
    if lora_path is None:
        monkeypatch.delenv("MINIMAX_H3_LORA_PATH", raising=False)
    else:
        monkeypatch.setenv("MINIMAX_H3_LORA_PATH", lora_path)
    with pytest.raises(ValueError, match="MINIMAX_H3_LORA_PATH"):
        _runner(name)._create_pipeline_kwargs()


@pytest.mark.parametrize(("name", "base", "task"), LIGHTX2V_RUNNERS)
def test_builds_through_the_turbo_factory(name, base, task):
    assert (
        getattr(dit_runners, name)._pipeline_factory
        is dit_runners._MiniMaxH3LightX2VMixin._pipeline_factory
    )
    assert (
        getattr(dit_runners, base)._pipeline_factory
        is dit_runners.TTMiniMaxH3Runner._pipeline_factory
    )


def test_fl2va_lightx2v_maps_keyframes_like_fl2va(monkeypatch):
    runner = _runner("TTMiniMaxH3FL2VALightX2VRunner")
    runner.image_manager = Mock(base64_to_pil_image=lambda b64: f"img:{b64}")
    request = SimpleNamespace(
        image_prompts=[
            SimpleNamespace(image="a", frame_pos=0),
            SimpleNamespace(image="b", frame_pos=-1),
        ]
    )
    assert runner._pipeline_extra_kwargs(request) == {
        "image": "img:a",
        "last_image": "img:b",
    }


def test_ref2va_lightx2v_keeps_the_ref2va_l1_pool(monkeypatch):
    captured = {}

    def fake_params(mesh_shape, *, l1_small_size=None):
        captured["l1_small_size"] = l1_small_size
        return {}

    monkeypatch.setattr(dit_runners, "_minimax_h3_device_params", fake_params)
    runner = _runner("TTMiniMaxH3Ref2VALightX2VRunner")
    runner.settings = SimpleNamespace(device_mesh_shape=(4, 8))
    runner.get_pipeline_device_params()
    assert captured["l1_small_size"] == 16384
