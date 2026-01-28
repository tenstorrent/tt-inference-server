# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.base_service import BaseService
from config.constants import JobTypes
from config.settings import get_settings
from domain.training_request import TrainingRequest

class TrainingService(BaseService):
    def __init__(self):
        self.settings = get_settings()
        super().__init__()
        
    async def create_job(self, job_type: JobTypes, request: TrainingRequest) -> dict:
        if job_type != JobTypes.TRAINING:
            raise ValueError("The job type must be TRAINING, since the chosen model service is TrainingService")
        if self.settings.dataset_loader == "" or self.settings.dataset_loader == None:
            raise ValueError("The dataset loader must be set")
        
        dataset_dict = {} 
        dataset_dict["dataset_loader"] = self.settings.dataset_loader
        dataset_dict["dataset_max_length"] = self.settings.dataset_max_length
        # TODO: place dataset_dict into job request parameters

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self.settings.model_runner,
            request=request,
            task_function=self.process_request,
        )
    