# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from uuid import uuid4

from config.settings import settings
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_service import BaseService
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from utils.helpers import log_execution_time
from utils.logger import TTLogger

class ImageService(BaseService):

    @log_execution_time("SDXL service init")
    def __init__(self):
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()

    @log_execution_time("Scheduler image processing")
    async def process_image(self, image_generate_request: ImageGenerateRequest) -> str:
        # set task id
        task_id = str(uuid4())
        image_generate_request._task_id = task_id
        self.scheduler.process_request(image_generate_request)
        future = asyncio.get_running_loop().create_future()
        self.scheduler.result_futures[task_id] = future
        try:
            result = await future
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise e
        self.scheduler.result_futures.pop(task_id, None)
        return result

    def check_is_model_ready(self):
        """Detailed system status for monitoring"""
        return {
            'model_ready': self.scheduler.check_is_model_ready(),
            'queue_size': self.scheduler.task_queue.qsize() if hasattr(self.scheduler.task_queue, 'qsize') else 'unknown',
            'max_queue_size': settings.max_queue_size,
            'worker_count': len(self.scheduler.workers) if hasattr(self.scheduler, 'workers') else 'unknown',
            'runner_in_use': settings.model_runner,
        }

    @log_execution_time("Starting workers")
    def start_workers(self):
        self.scheduler.start_workers()

    def stop_workers(self):
        return self.scheduler.stop_workers()
