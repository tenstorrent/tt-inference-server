# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from multiprocessing import Manager
from typing import Optional

from model_services.base_job_service import BaseJobService
from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    JobTypes,
    ModelRunners,
)
from config.settings import get_settings
from domain.training_request import TrainingRequest
from utils.adapter_storage import get_adapter_storage


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        self._adapter_storage = get_adapter_storage()
        runner_enum = ModelRunners(self.settings.model_runner)
        model_names = MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(runner_enum, set())
        self._model_name = next(iter(model_names)).value
        super().__init__()

    async def create_job(self, job_type: JobTypes, request: TrainingRequest) -> dict:
        request.device_type = self.settings.device
        adapter_path = self._adapter_storage.ensure_job_dir(request._task_id)
        request._output_model_path = adapter_path
        request._adapter_storage = self._adapter_storage
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
        )

    def get_job_metrics(self, job_id: str, after: int = 0) -> list:
        metrics_list = super().get_job_metrics(job_id)
        if metrics_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(metrics_list[after:])

    def get_job_logs(self, job_id: str) -> list:
        logs_list = super().get_job_logs(job_id)
        if logs_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(logs_list)

    def get_job_checkpoints(self, job_id: str) -> list:
        checkpoints_list = super().get_job_checkpoints(job_id)
        if checkpoints_list is None:
            raise ValueError(f"Job {job_id} not found")
        return list(checkpoints_list)

    def get_checkpoint_download_path(
        self, job_id: str, checkpoint_id: str
    ) -> Optional[str]:
        checkpoints = self.get_job_checkpoints(job_id)
        if not any(ckpt["id"] == checkpoint_id for ckpt in checkpoints):
            return None
        return self._adapter_storage.get_checkpoint_path(job_id, checkpoint_id)
