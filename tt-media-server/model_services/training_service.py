# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from domain.base_request import BaseRequest
from config.constants import JobTypes
from config.settings import settings


class TrainingService(BaseService):
    def __init__(self):
        super().__init__()
    async def create_job(self, job_type: JobTypes, request: BaseRequest) -> dict:
        if job_type != JobTypes.TRAINING:
            raise ValueError("The job type must be TRAINING, since the chosen model service is TrainingService")
        if settings.dataset_loader == "" or settings.dataset_loader == None:
            raise ValueError("The dataset loader must be set")
        request.dataset=settings.dataset_loader.value
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=settings.model_runner.value,
            request=request,
            task_function=self.process_request,
        )
    