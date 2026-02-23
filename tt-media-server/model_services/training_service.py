# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import asyncio
import os
from multiprocessing import Manager

from model_services.base_job_service import BaseJobService
from config.constants import JobTypes
from config.settings import get_settings
from domain.training_request import TrainingRequest
from utils.job_manager import TrainingJobContext


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
        request._training_metrics = self._manager.list()

        training_ctx = TrainingJobContext(
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            training_metrics=request._training_metrics,
        )

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self.settings.model_runner,
            request=request,
            task_function=self.process_request,
            result_path=request._output_model_path,
            training_context=training_ctx,
        )

    async def stream_job_metrics(self, job_id: str):
        metrics_list = self._job_manager.get_training_metrics(job_id)
        if metrics_list is None:
            return

        last_sent = 0

        while True:
            current_len = len(metrics_list)
            if current_len > last_sent:
                for i in range(last_sent, current_len):
                    yield metrics_list[i]
                last_sent = current_len

            job_data = self._job_manager.get_job_metadata(job_id)
            if job_data and job_data["status"] in ("completed", "failed", "cancelled"):
                for i in range(last_sent, len(metrics_list)):
                    yield metrics_list[i]
                return

            await asyncio.sleep(1.0)
