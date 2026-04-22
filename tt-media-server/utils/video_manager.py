# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from __future__ import annotations

import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import simplejpeg
    _HAVE_SIMPLEJPEG = True
except ImportError:
    _HAVE_SIMPLEJPEG = False

from utils.decorators import log_execution_time
from utils.logger import TTLogger

_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_ENCODE_TIMEOUT_S = 600
_FFMPEG_REMUX_TIMEOUT_S = 60
_VIDEO_OUTPUT_DIR = Path("/tmp/videos")
_VALID_CHANNEL_COUNTS = (1, 3, 4)
_RGB_CHANNELS = 3
_MAX_PIXEL_VALUE = 255.0
_NORMALIZED_RANGE_MAX = 1.0
_MJPEG_WORKERS = 64

_pool = ThreadPoolExecutor(max_workers=_MJPEG_WORKERS)


class VideoManager:
    """MP4 export via FFmpeg subprocess pipe.

    Uses MJPEG + simplejpeg parallel encoding when available (~0.07s for 81
    720p frames), falling back to libx264 rawvideo otherwise.
    """

    def __init__(self):
        self._logger = TTLogger()

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames: NDArray, fps: int = 16) -> str:
        if hasattr(frames, "frames"):
            frames = frames.frames

        _VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_VIDEO_OUTPUT_DIR / f"{uuid.uuid4()}.mp4")

        try:
            processed = self._process_frames_for_export(frames)
            if _HAVE_SIMPLEJPEG:
                try:
                    self._encode_mjpeg(processed, output_path, fps)
                    return output_path
                except Exception as mjpeg_err:
                    self._logger.warning(f"MJPEG encode failed ({mjpeg_err}), falling back to x264")
            crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "23"))
            crf = max(_MIN_CRF, min(_MAX_CRF, crf))
            preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "ultrafast").strip()
            cmd = self._build_encode_cmd(processed, output_path, fps, crf, preset)
            self._run_ffmpeg(cmd, stdin_data=processed.tobytes())
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    def _encode_mjpeg(self, frames: NDArray, output_path: str, fps: int) -> None:
        """Encode frames to MJPEG MP4 using parallel simplejpeg + ffmpeg pipe."""
        quality = int(os.environ.get("TT_VIDEO_EXPORT_JPEG_QUALITY", "50"))
        _, height, width, _ = frames.shape

        cmd = [
            "ffmpeg", "-y",
            "-f", "mjpeg", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "copy",
            output_path,
        ]

        def enc(f):
            return simplejpeg.encode_jpeg(f, quality=quality, colorspace="RGB")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for jpg in _pool.map(enc, [frames[i] for i in range(len(frames))]):
                process.stdin.write(jpg)
            process.stdin.close()
            process.wait(timeout=_FFMPEG_ENCODE_TIMEOUT_S)
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError) as pipe_err:
            if "flush" in str(pipe_err) or "pipe" in str(pipe_err).lower():
                raise RuntimeError(f"FFmpeg MJPEG pipe error: {pipe_err}") from pipe_err
            raise
            process.kill()
            raise RuntimeError("FFmpeg MJPEG export timed out") from None

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode(errors='replace')}")

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames: NDArray) -> NDArray[np.uint8]:
        frames = _normalize_shape(frames)
        frames = _normalize_channels(frames)
        frames = _normalize_dtype(frames)
        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)
        return frames

    @staticmethod
    def _build_encode_cmd(
        frames: NDArray, output_path: str, fps: int, crf: int, preset: str
    ) -> list[str]:
        _, height, width, channels = frames.shape
        if channels != _RGB_CHANNELS:
            raise ValueError(
                f"Expected {_RGB_CHANNELS} RGB channels after processing, got {channels}"
            )
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
        ]
        if crf == 0:
            cmd.extend(["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"])
        else:
            cmd.extend([
                "-c:v", "libx264", "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-tune", "film",
                "-profile:v", "high", "-level", "4.2",
            ])
        if preset:
            cmd.extend(["-preset", preset])
        cmd.extend(["-movflags", "+faststart", output_path])
        return cmd

    @staticmethod
    def _run_ffmpeg(
        cmd: list[str],
        stdin_data: bytes | None = None,
        timeout: int = _FFMPEG_ENCODE_TIMEOUT_S,
    ) -> None:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = process.communicate(input=stdin_data, timeout=timeout)
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError) as pipe_err:
            if "flush" in str(pipe_err) or "pipe" in str(pipe_err).lower():
                raise RuntimeError(f"FFmpeg MJPEG pipe error: {pipe_err}") from pipe_err
            raise
            process.kill()
            raise RuntimeError("FFmpeg export timed out") from None
        if process.returncode != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @classmethod
    def ensure_faststart(cls, input_path: str, output_path: str) -> None:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-c", "copy", "-movflags", "faststart", output_path,
        ]
        cls._run_ffmpeg(cmd, timeout=_FFMPEG_REMUX_TIMEOUT_S)


def _normalize_shape(frames: NDArray) -> NDArray:
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"Unexpected frame dimensions: {frames.shape}")
    return frames


def _normalize_channels(frames: NDArray) -> NDArray:
    _, _, _, channels = frames.shape
    if channels not in _VALID_CHANNEL_COUNTS:
        raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")
    if channels == 1:
        return np.repeat(frames, _RGB_CHANNELS, axis=-1)
    if channels == 4:
        return frames[..., :_RGB_CHANNELS]
    return frames


def _normalize_dtype(frames: NDArray) -> NDArray[np.uint8]:
    if frames.dtype == np.uint8:
        return frames
    if frames.dtype in (np.float32, np.float64):
        max_val = float(np.max(frames)) if frames.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frames * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frames.clip(0, 255).astype(np.uint8)
    return frames.clip(0, 255).astype(np.uint8)
