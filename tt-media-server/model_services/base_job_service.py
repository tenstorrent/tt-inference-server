# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any, Optional

from config.constants import JobTypes
from config.settings import settings
from domain.base_request import BaseRequest
from model_services.base_service import BaseService
from utils.decorators import log_execution_time
from utils.job_manager import get_job_manager


class BaseJobService(BaseService):
    @log_execution_time("Base job service init")
    def __init__(self):
        super().__init__()
        self._job_manager = get_job_manager()

    async def create_job(self, job_type: JobTypes, request: BaseRequest) -> dict:
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=settings.model_weights_path,
            request=request,
            task_function=self.process_request,
        )

    def get_all_jobs_metadata(self, job_type: JobTypes = None) -> list[dict]:
        return self._job_manager.get_all_jobs_metadata(job_type)

    def get_job_metadata(self, job_id: str) -> Optional[dict]:
        return self._job_manager.get_job_metadata(job_id)

    def get_job_result_path(self, job_id: str) -> Optional[Any]:
        return self._job_manager.get_job_result_path(job_id)

    def cancel_job(self, job_id: str) -> bool:
        return self._job_manager.cancel_job(job_id)
