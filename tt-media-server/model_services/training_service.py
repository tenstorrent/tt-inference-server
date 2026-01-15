# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from config.constants import JobTypes
from config.settings import get_settings
from domain.base_request import BaseRequest

class TrainingService(BaseService):
    def __init__(self):
        self.settings = get_settings()
        super().__init__()
        
    async def create_job(self, job_type: JobTypes, request: BaseRequest) -> dict:
        if job_type != JobTypes.TRAINING:
            raise ValueError("The job type must be TRAINING, since the chosen model service is TrainingService")
        if self.settings.dataset_loader == "" or self.settings.dataset_loader == None:
            raise ValueError("The dataset loader must be set")
        
        request.dataset = self.settings.dataset_loader.value

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self.settings.model_runner.value,
            request=request,
            task_function=self.process_request,
        )
    