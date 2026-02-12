# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import os

from model_services.base_job_service import BaseJobService
from config.constants import JobTypes
from config.settings import get_settings
from domain.training_request import TrainingRequest


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        super().__init__()

    async def create_job(self, job_type: JobTypes, request: TrainingRequest) -> dict:
        os.makedirs("models_save", exist_ok=True)
        request._output_model_path = f"models_save/{request._task_id}.pt"
        self.logger.info(f"Generated output path: {request._output_model_path}")

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self.settings.model_runner,
            request=request,
            task_function=self.process_request,
            result_path=request._output_model_path,
        )
