# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import os
from multiprocessing import Manager

from model_services.base_job_service import BaseJobService
from config.constants import (
    TRAINING_STORE_ADAPTERS_DIR,
    JobTypes,
    ModelNames,
)
from config.settings import get_settings
from domain.training_request import TrainingRequest
from typing import Optional


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        self._model_name = ModelNames(self.settings.training_model).value
        super().__init__()

    async def create_job(
        self,
        job_type: JobTypes,
        request: TrainingRequest,
        org_id: Optional[str] = None,
    ) -> dict:
        request.device_type = self.settings.device
        adapter_path = os.path.join(TRAINING_STORE_ADAPTERS_DIR, request._task_id)
        os.makedirs(adapter_path, exist_ok=True)
        request._output_model_path = adapter_path
        self.logger.info(f"Generated output path: {request._output_model_path}")

        request._start_event = self._manager.Event()
        request._cancel_event = self._manager.Event()
        request._training_metrics = self._manager.list()
        request._training_logs = self._manager.list()
        request._training_checkpoints = self._manager.list()

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=self._model_name,
            request=request,
            task_function=self.process_request,
            result_path=request._output_model_path,
            start_event=request._start_event,
            cancel_event=request._cancel_event,
            job_metrics=request._training_metrics,
            job_logs=request._training_logs,
            job_checkpoints=request._training_checkpoints,
            org_id=org_id,
        )

    def get_job_metrics(
        self, job_id: str, org_id: Optional[str] = None, after: int = 0
    ) -> list:
        metrics_list = super().get_job_metrics(job_id, org_id=org_id)
        if metrics_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(metrics_list[after:])

    def get_job_logs(self, job_id: str, org_id: Optional[str] = None) -> list:
        logs_list = super().get_job_logs(job_id, org_id=org_id)
        if logs_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(logs_list)

    def get_job_checkpoints(self, job_id: str, org_id: Optional[str] = None) -> list:
        checkpoints_list = super().get_job_checkpoints(job_id, org_id=org_id)
        if checkpoints_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(checkpoints_list)

    def get_checkpoint_download_path(
        self, job_id: str, checkpoint_id: str, org_id: Optional[str] = None
    ) -> Optional[str]:
        checkpoints = self.get_job_checkpoints(job_id, org_id=org_id)
        if not any(ckpt["id"] == checkpoint_id for ckpt in checkpoints):
            return None
        result_path = self._job_manager.get_job_result_path(job_id, org_id=org_id)
        if not result_path:
            return None
        checkpoint_path = os.path.join(result_path, checkpoint_id)
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        return None
