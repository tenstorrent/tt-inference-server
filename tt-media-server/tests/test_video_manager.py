# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from utils.video_manager import (
    VideoManager,
    _normalize_channels,
    _normalize_dtype,
    _normalize_shape,
)


@pytest.fixture
def manager():
    return VideoManager()


class TestNormalizeShape:
    """Tests for _normalize_shape batch squeeze and validation."""

    def test_4d_passthrough(self):
        frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        result = _normalize_shape(frames)
        assert result.shape == (4, 64, 64, 3)

    def test_5d_squeezes_batch(self):
        frames = np.zeros((1, 4, 64, 64, 3), dtype=np.uint8)
        result = _normalize_shape(frames)
        assert result.shape == (4, 64, 64, 3)

    def test_3d_raises(self):
        frames = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unexpected frame dimensions"):
            _normalize_shape(frames)


class TestNormalizeChannels:
    """Tests for _normalize_channels grayscale/RGBA conversion."""

    def test_rgb_passthrough(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        result = _normalize_channels(frames)
        assert result.shape == (2, 8, 8, 3)

    def test_grayscale_to_rgb(self):
        frames = np.full((2, 8, 8, 1), 128, dtype=np.uint8)
        result = _normalize_channels(frames)
        assert result.shape == (2, 8, 8, 3)
        assert np.all(result == 128)

    def test_rgba_strips_alpha(self):
        frames = np.zeros((2, 8, 8, 4), dtype=np.uint8)
        frames[..., 3] = 255
        result = _normalize_channels(frames)
        assert result.shape == (2, 8, 8, 3)

    def test_invalid_channels_raises(self):
        frames = np.zeros((2, 8, 8, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected 1, 3, or 4"):
            _normalize_channels(frames)


class TestNormalizeDtype:
    """Tests for _normalize_dtype uint8 conversion."""

    def test_uint8_passthrough(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        result = _normalize_dtype(frames)
        assert result is frames

    def test_float32_0_to_1_scaled(self):
        frames = np.full((2, 8, 8, 3), 0.5, dtype=np.float32)
        result = _normalize_dtype(frames)
        assert result.dtype == np.uint8
        assert np.allclose(result, 128, atol=1)

    def test_float64_above_1_clipped(self):
        frames = np.full((2, 8, 8, 3), 200.0, dtype=np.float64)
        result = _normalize_dtype(frames)
        assert result.dtype == np.uint8
        assert np.all(result == 200)

    def test_int16_clipped(self):
        frames = np.array([[[[300, -10, 100]]]], dtype=np.int16)
        result = _normalize_dtype(frames)
        assert result.dtype == np.uint8
        assert result[0, 0, 0, 0] == 255
        assert result[0, 0, 0, 1] == 0
        assert result[0, 0, 0, 2] == 100


class TestProcessFramesForExport:
    """Integration tests for the full _process_frames_for_export pipeline."""

    def test_full_pipeline_uint8(self, manager):
        frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        result = manager._process_frames_for_export(frames)
        assert result.shape == (4, 64, 64, 3)
        assert result.dtype == np.uint8

    def test_non_contiguous_made_contiguous(self, manager):
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        frames = frames[::2]
        assert not frames.flags["C_CONTIGUOUS"]
        result = manager._process_frames_for_export(frames)
        assert result.flags["C_CONTIGUOUS"]

    def test_5d_float32_grayscale(self, manager):
        frames = np.full((1, 2, 8, 8, 1), 0.5, dtype=np.float32)
        result = manager._process_frames_for_export(frames)
        assert result.shape == (2, 8, 8, 3)
        assert result.dtype == np.uint8


class TestBuildEncodeCmd:
    """Tests for _build_encode_cmd command construction."""

    def test_lossless_crf0(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        cmd = VideoManager._build_encode_cmd(frames, "/tmp/out.mp4", 16, 0, "medium")
        assert "-crf" in cmd
        assert "yuv444p" in cmd
        assert "-preset" in cmd

    def test_lossy_crf_uses_yuv420p(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        cmd = VideoManager._build_encode_cmd(frames, "/tmp/out.mp4", 16, 23, "fast")
        assert "yuv420p" in cmd
        assert "fast" in cmd

    def test_empty_preset_omitted(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        cmd = VideoManager._build_encode_cmd(frames, "/tmp/out.mp4", 16, 0, "")
        assert "-preset" not in cmd

    def test_bad_channels_raises(self):
        frames = np.zeros((2, 8, 8, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3 RGB channels"):
            VideoManager._build_encode_cmd(frames, "/tmp/out.mp4", 16, 0, "medium")

    def test_faststart_and_output_path(self):
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        cmd = VideoManager._build_encode_cmd(frames, "/tmp/test.mp4", 24, 0, "medium")
        assert "+faststart" in cmd
        assert cmd[-1] == "/tmp/test.mp4"


class TestRunFfmpeg:
    """Tests for _run_ffmpeg process execution."""

    @patch("utils.video_manager.subprocess.Popen")
    def test_success(self, mock_popen):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        VideoManager._run_ffmpeg(["ffmpeg", "-version"])

    @patch("utils.video_manager.subprocess.Popen")
    def test_nonzero_returncode_raises(self, mock_popen):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"encoding error")
        proc.returncode = 1
        mock_popen.return_value = proc
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            VideoManager._run_ffmpeg(["ffmpeg", "-version"])

    @patch("utils.video_manager.subprocess.Popen")
    def test_timeout_kills_and_raises(self, mock_popen):
        proc = MagicMock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="ffmpeg", timeout=600
        )
        mock_popen.return_value = proc
        with pytest.raises(RuntimeError, match="timed out"):
            VideoManager._run_ffmpeg(["ffmpeg"], stdin_data=b"data")
        proc.kill.assert_called_once()

    @patch("utils.video_manager.subprocess.Popen")
    def test_stdin_pipe_when_data_provided(self, mock_popen):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        VideoManager._run_ffmpeg(["ffmpeg"], stdin_data=b"pixels")
        assert mock_popen.call_args[1]["stdin"] == subprocess.PIPE

    @patch("utils.video_manager.subprocess.Popen")
    def test_no_stdin_pipe_without_data(self, mock_popen):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        VideoManager._run_ffmpeg(["ffmpeg"])
        assert mock_popen.call_args[1]["stdin"] is None


class TestExportToMp4:
    """Tests for the top-level export_to_mp4 orchestrator."""

    @patch("utils.video_manager.subprocess.Popen")
    @patch("utils.video_manager.Path.mkdir")
    def test_returns_mp4_path(self, mock_mkdir, mock_popen, manager):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        path = manager.export_to_mp4(frames, fps=24)
        assert path.endswith(".mp4")

    @patch("utils.video_manager.subprocess.Popen")
    @patch("utils.video_manager.Path.mkdir")
    def test_wraps_exception_in_runtime_error(self, mock_mkdir, mock_popen, manager):
        mock_popen.side_effect = OSError("no ffmpeg")
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Failed to export video"):
            manager.export_to_mp4(frames)


class TestEnsureFaststart:
    """Tests for ensure_faststart delegating to _run_ffmpeg."""

    @patch("utils.video_manager.subprocess.Popen")
    def test_calls_ffmpeg_with_copy_and_faststart(self, mock_popen):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        VideoManager.ensure_faststart("/tmp/in.mp4", "/tmp/out.mp4")
        cmd = mock_popen.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd
        assert "faststart" in cmd

    @patch("utils.video_manager.subprocess.Popen")
    def test_uses_remux_timeout(self, mock_popen):
        from utils.video_manager import _FFMPEG_REMUX_TIMEOUT_S

        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        VideoManager.ensure_faststart("/tmp/in.mp4", "/tmp/out.mp4")
        actual_timeout = proc.communicate.call_args[1]["timeout"]
        assert actual_timeout == _FFMPEG_REMUX_TIMEOUT_S
