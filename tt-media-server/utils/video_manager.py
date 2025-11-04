# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import uuid
from utils.helpers import log_execution_time

class VideoManager:
    def __init__(self):
        super().__init__()

    @log_execution_time("Exporting video to MP4")
    def export_to_mp4(self, frames, output_path, fps=16):
        """
        Export a list/array of frames to an MP4 video file.
        """
        # Auto-generate path in videos directory
        video_id = str(uuid.uuid4())
        os.makedirs("videos", exist_ok=True)
        output_path = f"videos/{video_id}.mp4"

        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, output_video_path=output_path, fps=fps)
        except ImportError:
            # Assume frames are numpy arrays (H, W, C) or PIL images
            import imageio
            with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(
                        frame if hasattr(frame, 'shape') else np.array(frame)
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to export video: {e}")
