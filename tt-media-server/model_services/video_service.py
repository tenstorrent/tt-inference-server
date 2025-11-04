# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from utils.video_manager import VideoManager

class VideoService(BaseService):
    def __init__(self):
        super().__init__()
        self.video_manager = VideoManager()

    def post_process(self, result):
        """Convert frames to MP4 and return the file path"""
        output_path = self.video_manager.export_to_mp4(result)
        return output_path
