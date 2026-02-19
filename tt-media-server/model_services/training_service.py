# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import os
from multiprocessing import Manager
import asyncio

from model_services.base_job_service import BaseJobService
from config.constants import JobTypes
from config.settings import get_settings
from domain.training_request import TrainingRequest


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        super().__init__()

    async def create_job(self, job_type: JobTypes, request: TrainingRequest) -> dict:
        os.makedirs("models_save", exist_ok=True)
        request._output_model_path = f"models_save/{request._task_id}.pt"
        self.logger.info(f"Generated output path: {request._output_model_path}")

        request._start_event = self._manager.Event()
        request._cancel_event = self._manager.Event()
        request._training_metrics_queue = self._manager.Queue()

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self.settings.model_runner,
            request=request,
            task_function=self.process_request,
            result_path=request._output_model_path,
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            training_metrics_queue=request._training_metrics_queue,
        )
    
    async def stream_job_metrics(self, job_id: str):
        metrics_queue = self._job_manager.get_training_metrics_queue(job_id)
        if not metrics_queue:
            return

        while True:
            metric = await asyncio.get_event_loop().run_in_executor(
                None, metrics_queue.get  
            )
            if metric is None:  # sentinel = training done
                return
            
            yield metric
