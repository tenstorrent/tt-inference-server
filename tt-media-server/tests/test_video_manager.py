# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utils.video_manager import VideoManager


@pytest.fixture
def manager():
    return VideoManager()


class TestProcessFramesForExport:
    """Tests for _process_frames_for_export normalisation logic."""

    def test_4d_rgb_uint8_passthrough(self, manager):
        frames = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        result = manager._process_frames_for_export(frames)
        assert result.shape == (4, 64, 64, 3)
        assert result.dtype == np.uint8

    def test_5d_input_squeezes_batch(self, manager):
        frames = np.zeros((1, 4, 64, 64, 3), dtype=np.uint8)
        result = manager._process_frames_for_export(frames)
        assert result.shape == (4, 64, 64, 3)

    def test_grayscale_converted_to_rgb(self, manager):
        frames = np.full((2, 8, 8, 1), 128, dtype=np.uint8)
        result = manager._process_frames_for_export(frames)
        assert result.shape == (2, 8, 8, 3)
        assert np.all(result == 128)

    def test_rgba_strips_alpha(self, manager):
        frames = np.zeros((2, 8, 8, 4), dtype=np.uint8)
        frames[..., 3] = 255
        result = manager._process_frames_for_export(frames)
        assert result.shape == (2, 8, 8, 3)

    def test_float32_0_to_1_scaled_to_uint8(self, manager):
        frames = np.full((2, 8, 8, 3), 0.5, dtype=np.float32)
        result = manager._process_frames_for_export(frames)
        assert result.dtype == np.uint8
        assert np.allclose(result, 128, atol=1)

    def test_float64_above_1_clipped_to_uint8(self, manager):
        frames = np.full((2, 8, 8, 3), 200.0, dtype=np.float64)
        result = manager._process_frames_for_export(frames)
        assert result.dtype == np.uint8
        assert np.all(result == 200)

    def test_int16_clipped_to_uint8(self, manager):
        frames = np.array([[[[300, -10, 100]]]], dtype=np.int16)
        result = manager._process_frames_for_export(frames)
        assert result.dtype == np.uint8
        assert result[0, 0, 0, 0] == 255
        assert result[0, 0, 0, 1] == 0
        assert result[0, 0, 0, 2] == 100

    def test_non_contiguous_made_contiguous(self, manager):
        frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        frames = frames[::2]
        assert not frames.flags["C_CONTIGUOUS"]
        result = manager._process_frames_for_export(frames)
        assert result.flags["C_CONTIGUOUS"]

    def test_invalid_shape_raises(self, manager):
        frames = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unexpected frame dimensions"):
            manager._process_frames_for_export(frames)

    def test_invalid_channels_raises(self, manager):
        frames = np.zeros((2, 8, 8, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="expected 1, 3, or 4"):
            manager._process_frames_for_export(frames)


class TestExportWithFfmpegPipe:
    """Tests for _export_with_ffmpeg_pipe with mocked subprocess."""

    def _make_mock_process(self, returncode=0, stderr=b""):
        proc = MagicMock()
        proc.communicate.return_value = (None, stderr)
        proc.returncode = returncode
        return proc

    @patch("utils.video_manager.subprocess.Popen")
    def test_lossless_crf0(self, mock_popen, manager):
        mock_popen.return_value = self._make_mock_process()
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        manager._export_with_ffmpeg_pipe(frames, "/tmp/out.mp4", 16, 0, "medium")
        cmd = mock_popen.call_args[0][0]
        assert "-crf" in cmd
        assert "yuv444p" in cmd

    @patch("utils.video_manager.subprocess.Popen")
    def test_lossy_crf_uses_yuv420p(self, mock_popen, manager):
        mock_popen.return_value = self._make_mock_process()
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        manager._export_with_ffmpeg_pipe(frames, "/tmp/out.mp4", 16, 23, "fast")
        cmd = mock_popen.call_args[0][0]
        assert "yuv420p" in cmd
        assert "-preset" in cmd

    @patch("utils.video_manager.subprocess.Popen")
    def test_ffmpeg_nonzero_returncode_raises(self, mock_popen, manager):
        mock_popen.return_value = self._make_mock_process(
            returncode=1, stderr=b"encoding error"
        )
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            manager._export_with_ffmpeg_pipe(frames, "/tmp/out.mp4", 16, 0, "medium")

    @patch("utils.video_manager.subprocess.Popen")
    def test_ffmpeg_timeout_raises(self, mock_popen, manager):
        proc = MagicMock()
        proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="ffmpeg", timeout=600
        )
        proc.kill = MagicMock()
        mock_popen.return_value = proc
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="timed out"):
            manager._export_with_ffmpeg_pipe(frames, "/tmp/out.mp4", 16, 0, "medium")
        proc.kill.assert_called_once()

    @patch("utils.video_manager.subprocess.Popen")
    def test_bad_channels_raises(self, mock_popen, manager):
        frames = np.zeros((2, 8, 8, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 3 RGB channels"):
            manager._export_with_ffmpeg_pipe(frames, "/tmp/out.mp4", 16, 0, "medium")


class TestExportToMp4:
    """Tests for the top-level export_to_mp4 orchestrator."""

    @patch("utils.video_manager.subprocess.Popen")
    @patch("utils.video_manager.os.makedirs")
    def test_returns_mp4_path(self, mock_makedirs, mock_popen, manager):
        proc = MagicMock()
        proc.communicate.return_value = (None, b"")
        proc.returncode = 0
        mock_popen.return_value = proc
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        path = manager.export_to_mp4(frames, fps=24)
        assert path.endswith(".mp4")

    @patch("utils.video_manager.subprocess.Popen")
    @patch("utils.video_manager.os.makedirs")
    def test_wraps_exception_in_runtime_error(self, mock_makedirs, mock_popen, manager):
        mock_popen.side_effect = OSError("no ffmpeg")
        frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Failed to export video"):
            manager.export_to_mp4(frames)
