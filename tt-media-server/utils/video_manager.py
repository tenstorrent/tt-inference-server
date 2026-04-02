# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import uuid

import numpy as np

from utils.decorators import log_execution_time
from utils.logger import TTLogger

# H.264 CRF range (libx264)
_MIN_CRF = 0
_MAX_CRF = 51
_FFMPEG_COMMUNICATE_TIMEOUT_S = 600


class VideoManager:
    """MP4 export via FFmpeg stdin (raw RGB) — explicit CRF/preset (no imageio quality mapping)."""

    def __init__(self):
        super().__init__()
        self._logger = TTLogger()

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames, fps=16, timing_out=None):
        """
        Export frames to MP4 (H.264 via ffmpeg).

        Env (optional):
            TT_VIDEO_EXPORT_CRF: 0–51, lower = better quality. Default 0 (x264 lossless at
                CRF 0; very large files). For smaller files use e.g. 15–18.
            TT_VIDEO_EXPORT_PRESET: ultrafast … veryslow. Default medium.
                Set empty to omit -preset (encoder default).
        """
        import time

        self._logger.info(f"Starting video export with fps={fps}")

        if hasattr(frames, "frames"):
            frames = frames.frames
            self._logger.info(f"Extracted frames shape: {frames.shape}")

        self._logger.info(f"Input frames type: {type(frames)}")
        self._logger.info(
            f"Input frames shape: {getattr(frames, 'shape', 'No shape attribute')}"
        )

        video_id = str(uuid.uuid4())
        video_dir = "/tmp/videos"
        os.makedirs(video_dir, exist_ok=True)
        output_path = f"{video_dir}/{video_id}.mp4"
        self._logger.info(f"Generated output path: {output_path}")

        t_export_start = time.perf_counter()

        crf = int(os.environ.get("TT_VIDEO_EXPORT_CRF", "0"))
        crf = max(_MIN_CRF, min(_MAX_CRF, crf))
        preset = os.environ.get("TT_VIDEO_EXPORT_PRESET", "medium").strip()

        try:
            t_process_start = time.perf_counter()
            processed_frames = self._process_frames_for_export(frames)

            if timing_out is not None:
                timing_out["prep_before_first_frame_s"] = round(
                    time.perf_counter() - t_process_start, 4
                )

            self._logger.info(
                f"Using FFmpeg stdin (rawvideo rgb24), CRF={crf}, preset={preset!r}"
            )

            t_encode_start = time.perf_counter()
            self._export_with_ffmpeg_pipe(
                processed_frames, output_path, fps, crf, preset
            )

            t_end = time.perf_counter()

            if timing_out is not None:
                timing_out["encode_after_first_frame_s"] = round(
                    t_end - t_encode_start, 4
                )
                timing_out["export_wall_s"] = round(t_end - t_export_start, 4)
                timing_out["encoder_incremental"] = False
                timing_out["ttft_to_first_frame_appended_s"] = round(
                    t_end - t_export_start, 4
                )

            self._logger.info(f"Video export completed successfully: {output_path}")
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}") from e

    def _export_with_ffmpeg_pipe(self, frames, output_path, fps, crf, preset):
        """Stream NVHWC uint8 RGB frames to ffmpeg rawvideo → libx264."""
        num_frames, height, width, channels = frames.shape
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
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-crf",
                    "0",
                    "-pix_fmt",
                    "yuv444p",
                ]
            )
            if preset:
                cmd.extend(["-preset", preset])
            self._logger.info("Encoding: CRF 0 + yuv444p (no chroma subsampling)")
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
            self._logger.info(f"Encoding: CRF {crf} + yuv420p")

        cmd.extend(["-movflags", "+faststart", output_path])

        self._logger.info(f"FFmpeg: {' '.join(cmd)}")
        self._logger.info(f"Streaming {num_frames} frames {width}x{height} @ {fps} fps")

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
            self._logger.error(f"FFmpeg stderr: {error_msg}")
            raise RuntimeError(f"FFmpeg failed: {error_msg}")

        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            self._logger.info(f"Output file size: {size_mb:.2f} MB")

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames):
        """Normalize to contiguous uint8 (N, H, W, 3) for rawvideo rgb24."""

        self._logger.info(f"Processing frames with shape: {frames.shape}")

        if len(frames.shape) == 5:
            frames = frames[0]
            self._logger.info(f"Extracted first batch, new shape: {frames.shape}")

        if len(frames.shape) != 4:
            self._logger.error(f"Unexpected frame shape: {frames.shape}")
            raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

        num_frames, height, width, channels = frames.shape
        self._logger.info(
            f"Video details: {num_frames} frames, {height}x{width}, {channels} channels"
        )

        if channels not in [1, 3, 4]:
            self._logger.error(f"Unsupported channel count: {channels}")
            raise ValueError(f"Frames have {channels} channels, expected 1, 3, or 4")

        if channels == 1:
            frames = np.repeat(frames, 3, axis=-1)
            self._logger.info("Converted grayscale to RGB")
        elif channels == 4:
            frames = frames[..., :3]
            self._logger.info("Removed alpha channel")

        if frames.dtype != np.uint8:
            if frames.dtype in (np.float32, np.float64):
                max_val = float(np.max(frames)) if frames.size else 0.0
                if max_val <= 1.0:
                    frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frames = frames.clip(0, 255).astype(np.uint8)
            else:
                frames = frames.clip(0, 255).astype(np.uint8)
            self._logger.info("Converted to uint8")

        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)

        self._logger.info(f"Processed {num_frames} frames")
        self._logger.info(f"Output shape: {frames.shape}, dtype: {frames.dtype}")
        self._logger.info(f"Value range: [{frames.min()}, {frames.max()}]")

        return frames

    @staticmethod
    def ensure_faststart(input_path, output_path):
        """
        Rewrites the MP4 file with -movflags faststart using ffmpeg.
        """
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
