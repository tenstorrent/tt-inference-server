# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Mock video runner: pipeline output shape and export hook (conftest mocks torch/diffusers)."""

import os
from unittest.mock import MagicMock

import pytest

from tt_model_runners import mock_video_runner as mvr


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(mvr.time, "sleep", lambda _: None)


def test_mock_pipeline_produces_wan_style_batch(no_sleep):
    p = mvr.MockVideoPipeline()
    frames = p(prompt="test", num_frames=4, height=16, width=16, seed=2)
    assert frames.dtype.name.startswith("uint")
    assert frames.shape == (1, 4, 16, 16, 3)


def test_mock_video_runner_run_invokes_export_to_mp4(no_sleep, monkeypatch, tmp_path):
    """Same sequence as standalone bridge: frames -> VideoManager.export_to_mp4 -> path."""
    captured = {}

    def fake_export(self, frames):
        captured["shape"] = frames.shape
        out = tmp_path / "mock.mp4"
        out.write_bytes(
            b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso6" + b"\x00" * 100
        )
        return str(out)

    monkeypatch.setattr("utils.video_manager.VideoManager.export_to_mp4", fake_export)

    req = MagicMock()
    req.prompt = "p"
    req.negative_prompt = None
    req.num_inference_steps = 12
    req.seed = 0

    runner = mvr.MockVideoRunner("dev0")
    runner.pipeline = mvr.MockVideoPipeline()
    out = runner.run([req])

    assert out == [str(tmp_path / "mock.mp4")]
    assert captured["shape"] == (1, 81, 480, 832, 3)
    assert os.path.isfile(out[0])
