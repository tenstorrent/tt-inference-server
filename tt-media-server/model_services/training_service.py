# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import os
from multiprocessing import Manager

from model_services.base_job_service import BaseJobService
from config.constants import JobTypes, ModelNames
from config.settings import get_settings
from domain.training_request import TrainingRequest
from config.constants import FINE_TUNING_STORE_ADAPTERS_DIR


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        super().__init__()

    async def create_job(self, job_type: JobTypes, request: TrainingRequest) -> dict:
        adapter_path = os.path.join(FINE_TUNING_STORE_ADAPTERS_DIR, request._task_id)
        os.makedirs(adapter_path, exist_ok=True)
        request._output_model_path = adapter_path
        self.logger.info(f"Generated output path: {request._output_model_path}")

        request._start_event = self._manager.Event()
        request._cancel_event = self._manager.Event()
        request._training_metrics = self._manager.list()

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=ModelNames.GEMMA_1_1_2B_IT.value,  # hardcoded for now
            request=request,
            task_function=self.process_request,
            result_path=request._output_model_path,
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            job_metrics=request._training_metrics,
        )

    def get_job_metrics(self, job_id: str, after: int = 0) -> list:
        metrics_list = super().get_job_metrics(job_id)
        if metrics_list is None:
            return []
        return list(metrics_list[after:])
