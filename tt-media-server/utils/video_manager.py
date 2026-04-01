# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import uuid

from utils.decorators import log_execution_time
from utils.logger import TTLogger


class VideoManager:
    """MP4 export via imageio/ffmpeg (see env TT_VIDEO_EXPORT_*)."""

    def __init__(self):
        super().__init__()
        self._logger = TTLogger()

    @staticmethod
    def _write_mp4_imageio(uint8_frames, output_path, fps, quality, x264_preset):
        import imageio

        ffmpeg_extra = ["-preset", x264_preset] if x264_preset else []
        with imageio.get_writer(
            output_path,
            fps=fps,
            quality=quality,
            macro_block_size=16,
            ffmpeg_params=ffmpeg_extra,
        ) as writer:
            for frame in uint8_frames:
                writer.append_data(frame)

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames, fps=16):
        """
        Export frames to MP4 (H.264 via ffmpeg).

        Env (optional):
            TT_VIDEO_EXPORT_QUALITY: 1–10 (imageio / libx264 CRF mapping). Default 5 —
                same as historical diffusers ``export_to_video`` default (unchanged quality).
            TT_VIDEO_EXPORT_X264_PRESET: e.g. ultrafast, veryfast, faster, fast, medium.
                Default faster — faster encode at the same quality (CRF); slightly larger files.
                Set empty to omit -preset (matches old encoder-time behavior more closely).
        """
        self._logger.info(f"Starting video export with fps={fps}")

        if hasattr(frames, "frames"):
            frames = frames.frames
            self._logger.info(f"Extracted frames shape: {frames.shape}")

        self._logger.info(f"Input frames type: {type(frames)}")
        self._logger.info(
            f"Input frames shape: {getattr(frames, 'shape', 'No shape attribute')}"
        )

        # Auto-generate path in videos directory
        video_id = str(uuid.uuid4())
        video_dir = "/tmp/videos"
        os.makedirs(video_dir, exist_ok=True)
        output_path = f"{video_dir}/{video_id}.mp4"
        self._logger.info(f"Generated output path: {output_path}")

        export_quality = float(os.environ.get("TT_VIDEO_EXPORT_QUALITY", "5"))
        export_quality = max(1.0, min(10.0, export_quality))
        x264_preset = os.environ.get("TT_VIDEO_EXPORT_X264_PRESET", "faster").strip()

        try:
            processed_frames = self._process_frames_for_export(frames)
            uint8_frames = [(f * 255).astype("uint8") for f in processed_frames]

            try:
                import imageio

                imageio.plugins.ffmpeg.get_exe()
                self._logger.info(
                    "Using imageio ffmpeg writer "
                    f"(quality={export_quality}, x264_preset={x264_preset!r})"
                )
                self._write_mp4_imageio(
                    uint8_frames, output_path, fps, export_quality, x264_preset
                )
            except (ImportError, AttributeError) as imageio_err:
                self._logger.warning(
                    "imageio ffmpeg unavailable (%s); using diffusers export_to_video",
                    imageio_err,
                )
                from diffusers.utils import export_to_video

                export_to_video(
                    processed_frames,
                    output_video_path=output_path,
                    fps=fps,
                    quality=export_quality,
                )

            self._logger.info(f"Video export completed successfully: {output_path}")
            return output_path

        except Exception as e:
            self._logger.error(f"Video export failed: {e}")
            raise RuntimeError(f"Failed to export video: {e}")

    @log_execution_time("Processing frames for export")
    def _process_frames_for_export(self, frames):
        """Process frames to ensure they're in the correct format for video export."""
        import numpy as np

        self._logger.info(f"Processing frames with shape: {frames.shape}")

        # Handle different possible shapes:
        # WAN output: (1, num_frames, height, width, channels)
        # Expected: (num_frames, height, width, channels)
        if len(frames.shape) == 5:
            # Shape: (batch, num_frames, height, width, channels)
            # Take first batch
            frames = frames[0]
            self._logger.info(f"Extracted first batch, new shape: {frames.shape}")

        if len(frames.shape) == 4:
            # Shape: (num_frames, height, width, channels) - this is what we want
            self._logger.info(f"Frames in correct 4D format: {frames.shape}")
            num_frames, height, width, channels = frames.shape
            self._logger.info(
                f"Video details: {num_frames} frames, {height}x{width}, {channels} channels"
            )
        else:
            self._logger.error(f"Unexpected frame shape: {frames.shape}")
            raise ValueError(f"Unexpected frame dimensions: {frames.shape}")

        # Validate channels
        if frames.shape[-1] not in [1, 3, 4]:
            self._logger.error(f"Unsupported channel count: {frames.shape[-1]}")
            raise ValueError(
                f"Frames have {frames.shape[-1]} channels, expected 1, 3, or 4"
            )

        # Convert to list of individual frames
        frame_list = []
        for i in range(frames.shape[0]):
            frame = frames[i]

            # Handle different channel counts
            if frame.shape[-1] == 3:
                # RGB - perfect
                frame_list.append(frame)
            elif frame.shape[-1] == 1:
                # Grayscale - convert to RGB
                frame = np.repeat(frame, 3, axis=-1)
                frame_list.append(frame)
            elif frame.shape[-1] == 4:
                # RGBA - remove alpha channel
                frame = frame[:, :, :3]
                frame_list.append(frame)

        self._logger.info(f"Processed {len(frame_list)} frames")
        if frame_list:
            sample_frame = frame_list[0]
            self._logger.info(
                f"Sample frame shape: {sample_frame.shape}, dtype: {sample_frame.dtype}"
            )
            self._logger.info(
                f"Sample frame value range: [{sample_frame.min():.4f}, {sample_frame.max():.4f}]"
            )

        return frame_list

    @staticmethod
    def ensure_faststart(input_path, output_path):
        """
        Rewrites the MP4 file with -movflags faststart using ffmpeg.
        """
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
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
