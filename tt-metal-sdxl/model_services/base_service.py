# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from uuid import uuid4
from abc import ABC, abstractmethod
from asyncio import Queue

from config.settings import settings
from domain.base_request import BaseRequest
from model_services.scheduler import Scheduler
from resolver.scheduler_resolver import get_scheduler
from utils.helpers import log_execution_time
from utils.logger import TTLogger

class BaseService(ABC):
    @log_execution_time("Base service init")
    def __init__(self):
        self.task_queue = Queue()
        self.result_futures = {}
        self.scheduler: Scheduler = get_scheduler()
        self.logger = TTLogger()

    @log_execution_time("Scheduler request processing")
    async def process_request(self, request: BaseRequest) -> str:
        self.scheduler.process_request(request)
        future = asyncio.get_running_loop().create_future()
        self.scheduler.result_futures[request._task_id] = future
        try:
            result = await future
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise e
        self.scheduler.result_futures.pop(request._task_id, None)
        if (result):
            return self.post_processing(result)
        else:
            self.logger.error(f"Post processing failed for task {request._task_id}")
            raise ValueError("Post processing failed")

    def check_is_model_ready(self) -> dict:
        """Detailed system status for monitoring"""
        return {
            'model_ready': self.scheduler.check_is_model_ready(),
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
    def start_workers(self):
        self.scheduler.start_workers()

    @log_execution_time("Stopping workers")
    def stop_workers(self):
        return self.scheduler.stop_workers()

    @abstractmethod
    def post_processing(self, result):
        pass