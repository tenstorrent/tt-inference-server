# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import uuid

import numpy as np

from utils.decorators import log_execution_time
from utils.logger import TTLogger

_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_COMMUNICATE_TIMEOUT_S = 600
_VIDEO_OUTPUT_DIR = "/tmp/videos"
_VALID_CHANNEL_COUNTS = (1, 3, 4)
_RGB_CHANNELS = 3
_MAX_PIXEL_VALUE = 255.0
_NORMALIZED_RANGE_MAX = 1.0


class VideoManager:
    """MP4 export via FFmpeg subprocess pipe (raw RGB → libx264)."""

    def __init__(self):
        self._logger = TTLogger()

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames, fps=16):
        """
        Export frames to MP4 (H.264 via ffmpeg).

        Env (optional):
            TT_VIDEO_EXPORT_CRF: 0–51, lower = better quality. Default 0.
            TT_VIDEO_EXPORT_PRESET: ultrafast … veryslow. Default medium.
        """
        if hasattr(frames, "frames"):
            frames = frames.frames

        os.makedirs(_VIDEO_OUTPUT_DIR, exist_ok=True)
        output_path = f"{_VIDEO_OUTPUT_DIR}/{uuid.uuid4()}.mp4"

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "0"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "medium").strip()

        try:
            processed = self._process_frames_for_export(frames)
            cmd = self._build_encode_cmd(processed, output_path, fps, crf, preset)
            self._run_ffmpeg(cmd, stdin_data=processed.tobytes())
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames):
        """Normalize to contiguous uint8 (N, H, W, 3) for rawvideo rgb24."""
        frames = _normalize_shape(frames)
        frames = _normalize_channels(frames)
        frames = _normalize_dtype(frames)

        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)

        return frames

    @staticmethod
    def _build_encode_cmd(frames, output_path, fps, crf, preset):
        """Build the ffmpeg rawvideo → libx264 command list."""
        _, height, width, channels = frames.shape
        if channels != _RGB_CHANNELS:
            raise ValueError(
                f"Expected {_RGB_CHANNELS} RGB channels after processing, got {channels}"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
        ]

        if crf == 0:
            cmd.extend(["-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"])
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-tune",
                    "film",
                    "-profile:v",
                    "high",
                    "-level",
                    "4.2",
                ]
            )

        if preset:
            cmd.extend(["-preset", preset])

        cmd.extend(["-movflags", "+faststart", output_path])
        return cmd

    @staticmethod
    def _run_ffmpeg(cmd, stdin_data=None):
        """Execute an ffmpeg command, raising on failure or timeout."""
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr = process.communicate(
                input=stdin_data,
                timeout=_FFMPEG_COMMUNICATE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg export timed out") from None

        if process.returncode != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @staticmethod
    def ensure_faststart(input_path, output_path):
        """Rewrites the MP4 file with -movflags faststart using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-c",
            "copy",
            "-movflags",
            "faststart",
            output_path,
        ]
        VideoManager._run_ffmpeg(cmd)


def _normalize_shape(frames):
    """Squeeze batch dim and validate 4D (N, H, W, C)."""
    if len(frames.shape) == 5:
        frames = frames[0]

    if len(frames.shape) != 4:
        raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

    return frames


def _normalize_channels(frames):
    """Convert grayscale or RGBA to RGB."""
    _, _, _, channels = frames.shape

    if channels not in _VALID_CHANNEL_COUNTS:
        raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")

    if channels == 1:
        return np.repeat(frames, _RGB_CHANNELS, axis=-1)
    if channels == 4:
        return frames[..., :_RGB_CHANNELS]

    return frames


def _normalize_dtype(frames):
    """Convert to uint8, handling float [0,1] and [0,255] ranges."""
    if frames.dtype == np.uint8:
        return frames

    if frames.dtype in (np.float32, np.float64):
        max_val = float(np.max(frames)) if frames.size else 0.0
        if max_val <= _NORMALIZED_RANGE_MAX:
            return (frames * _MAX_PIXEL_VALUE).clip(0, 255).astype(np.uint8)
        return frames.clip(0, 255).astype(np.uint8)

    return frames.clip(0, 255).astype(np.uint8)
