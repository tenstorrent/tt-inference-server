# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import numpy as np
from model_services.base_service import BaseService
from model_services.cpu_workload_handler import CpuWorkloadHandler


def create_video_worker_context():
    from utils.video_manager import VideoManager

    return VideoManager()


def video_worker_function(video_manager, video_frames):
    return video_manager.export_to_mp4(video_frames)


class VideoService(BaseService):
    def __init__(self):
        super().__init__()

        warmup_task_data = [np.zeros((1, 64, 64, 3), dtype=np.uint8)]
        self._cpu_workload_handler = CpuWorkloadHandler(
            name="VideoPostprocessing",
            worker_count=self.scheduler.get_worker_count(),
            worker_function=video_worker_function,
            worker_context_setup=create_video_worker_context,
            warmup_task_data=warmup_task_data,
        )

    async def post_process(self, result):
        """Asynchronous postprocessing using queue-based workers"""
        try:
            video_file = await self._cpu_workload_handler.execute_task(result)
        except Exception as e:
            self.logger.error(f"Video postprocessing failed: {e}")
            raise
        return video_file

    def stop_workers(self):
        self.logger.info("Shutting down video postprocessing workers")
        self._cpu_workload_handler.stop_workers()

        return super().stop_workers()
