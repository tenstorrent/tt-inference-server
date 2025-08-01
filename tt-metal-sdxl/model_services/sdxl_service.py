# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from uuid import uuid4

from domain.image_generate_request import ImageGenerateRequest
from model_services.device_worker import device_worker
from model_services.base_model import BaseModel
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from utils.helpers import log_execution_time
from utils.image_manager import ImageManager
from utils.logger import TTLogger

class SDXLService(BaseModel):

    @log_execution_time("SDXL service init")
    def __init__(self):
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()

    @log_execution_time("Scheduler image processing")
    async def processImage(self, imageGenerateRequest: ImageGenerateRequest) -> str:
        # don't do any work if model is not ready
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
        # pop the future from the result_futures to avoid memory leaks
        self.scheduler.result_futures.pop(task_id, None)
        return result

    def checkIsModelReady(self):
        return self.scheduler.checkIsModelReady()

    @log_execution_time("Starting workers")
    def startWorkers(self):
        self.scheduler.startWorkers()

    def stopWorkers(self):
        return self.scheduler.stopWorkers()
