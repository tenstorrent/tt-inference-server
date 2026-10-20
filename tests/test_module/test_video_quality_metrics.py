# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Structural video metrics on real mp4s encoded with imageio-ffmpeg."""

from __future__ import annotations

import pytest

from test_module._test_common.video_quality_metrics import (
    probe_video,
    sample_video_frames,
    structural_frame_metrics,
)
from tests.test_module.video_clips import (
    black_frames,
    gradient,
    moving_frames,
    write_clip,
)

pytest.importorskip("numpy")
pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("imageio.v3")


def test_moving_clip_is_valid(tmp_path):
    clip = write_clip(tmp_path / "moving.mp4", moving_frames(24))
    metrics = structural_frame_metrics(sample_video_frames(clip))
    assert metrics["total_decoded_frames"] == 24
    assert not metrics["is_black"]
    assert not metrics["is_flat"]
    assert not metrics["is_frozen"]


def test_black_clip_is_black(tmp_path):
    clip = write_clip(tmp_path / "black.mp4", black_frames(24))
    assert structural_frame_metrics(sample_video_frames(clip))["is_black"]


def test_repeated_frame_is_frozen_not_black(tmp_path):
    clip = write_clip(tmp_path / "frozen.mp4", [gradient(64, 48)] * 24)
    metrics = structural_frame_metrics(sample_video_frames(clip))
    assert metrics["is_frozen"]
    assert not metrics["is_black"]


def test_probe_reports_the_encoded_size(tmp_path):
    clip = write_clip(tmp_path / "size.mp4", moving_frames(32, width=96, height=64))
    probe = probe_video(clip, expected_duration=2.0, expected_ratio="3:2")
    assert (probe["width"], probe["height"]) == (96, 64)
    assert probe["valid"]


def test_undecodable_file_raises_value_error(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        sample_video_frames(junk)
