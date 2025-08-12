# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from uuid import uuid4

from config.settings import settings
from domain.image_generate_request import ImageGenerateRequest
from model_services.device_worker import device_worker
from model_services.base_model import BaseModel
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from utils.helpers import log_execution_time
from utils.image_manager import ImageManager
from utils.logger import TTLogger

class ImageService(BaseModel):

    @log_execution_time("SDXL service init")
    def __init__(self):
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()

    @log_execution_time("Scheduler image processing")
    async def processImage(self, imageGenerateRequest: ImageGenerateRequest) -> str:
        # set task id
        task_id = str(uuid4())
        imageGenerateRequest._task_id = task_id
        self.scheduler.process_request(imageGenerateRequest)
        future = asyncio.get_running_loop().create_future()
        self.scheduler.result_futures[task_id] = future
        try:
            result = await future
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise e
        self.scheduler.result_futures.pop(task_id, None)
        if (result):
            return ImageManager("img").convertImageToBytes(result)
        else:
            self.logger.error(f"Image processing failed for task {task_id}")
            raise ValueError("Image processing failed")

    def checkIsModelReady(self):
        """Detailed system status for monitoring"""
        return {
            'model_ready': self.scheduler.checkIsModelReady(),
            'queue_size': self.scheduler.task_queue.qsize() if hasattr(self.scheduler.task_queue, 'qsize') else 'unknown',
            'max_queue_size': settings.max_queue_size,
            'worker_count': len(self.scheduler.workers) if hasattr(self.scheduler, 'workers') else 'unknown',
            'runner_in_use': settings.model_runner,
        }

    async def deep_reset(self) -> bool:
        """Reset the device and all the scheduler workers and processes"""
        self.logger.info("Resetting device")
        # Create a task to run in the background
        asyncio.create_task(self.scheduler.deep_restart_workers())
        return True

    @log_execution_time("Starting workers")
    def startWorkers(self):
        self.scheduler.startWorkers()

    def stopWorkers(self):
        return self.scheduler.stopWorkers()
