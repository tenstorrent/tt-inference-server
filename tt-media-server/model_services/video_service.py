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
        return result

    def stop_workers(self):
        return super().stop_workers()
