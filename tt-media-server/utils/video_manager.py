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


class VideoManager:
    """MP4 export via FFmpeg subprocess pipe (raw RGB → libx264)."""

    def __init__(self):
        super().__init__()
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

        video_id = str(uuid.uuid4())
        video_dir = "/tmp/videos"
        os.makedirs(video_dir, exist_ok=True)
        output_path = f"{video_dir}/{video_id}.mp4"

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "0"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "medium").strip()

        try:
            processed_frames = self._process_frames_for_export(frames)
            self._export_with_ffmpeg_pipe(
                processed_frames, output_path, fps, crf, preset
            )
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    def _export_with_ffmpeg_pipe(self, frames, output_path, fps, crf, preset):
        """Stream NHWC uint8 RGB frames to ffmpeg rawvideo → libx264."""
        _, height, width, channels = frames.shape
        if channels != 3:
            raise ValueError(
                f"Expected 3 RGB channels after processing, got {channels}"
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

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            _, stderr = process.communicate(
                input=frames.tobytes(),
                timeout=_FFMPEG_COMMUNICATE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg export timed out") from None

        if process.returncode != 0:
            error_msg = stderr.decode(errors="replace") if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames):
        """Normalize to contiguous uint8 (N, H, W, 3) for rawvideo rgb24."""
        if len(frames.shape) == 5:
            frames = frames[0]

        if len(frames.shape) != 4:
            raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

        _, _, _, channels = frames.shape

        if channels not in [1, 3, 4]:
            raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")

        if channels == 1:
            frames = np.repeat(frames, 3, axis=-1)
        elif channels == 4:
            frames = frames[..., :3]

        if frames.dtype != np.uint8:
            if frames.dtype in (np.float32, np.float64):
                max_val = float(np.max(frames)) if frames.size else 0.0
                if max_val <= 1.0:
                    frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frames = frames.clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)

        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)

        return frames

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
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
