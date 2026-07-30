# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import asyncio
import concurrent.futures
import os
import shutil
from multiprocessing import Manager, get_context

from model_services.base_job_service import BaseJobService
from config.constants import (
    TRAINING_STORE_ADAPTERS_DIR,
    TRAINING_STORE_MERGED_MODELS_DIR,
    JobTypes,
    ModelNames,
    SupportedModels,
)
from config.settings import get_settings
from domain.adapter_merge_request import AdapterMergeRequest
from domain.training_request import TrainingRequest
from typing import Optional
from utils.adapter_merge_utils import merge_adapter


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        self._model_name = ModelNames(self.settings.training_model).value
        # Base model HF repo id backing the configured training model; used when
        # merging LoRA adapters back into the base weights.
        self._base_model_name = SupportedModels[
            ModelNames(self.settings.training_model).name
        ].value
        # Serializes adapter merges: loading a base model on CPU is memory-heavy,
        # so only one merge is allowed at a time within this container.
        self._adapter_merge_lock = asyncio.Lock()
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

    async def create_adapter_merge_job(
        self, request: AdapterMergeRequest, org_id: Optional[str] = None
    ) -> dict:
        """Create a job that merges a LoRA adapter checkpoint into its base model.

        The merge runs on CPU and does not touch the accelerator; it is tracked
        through the shared job manager like any other job. The merged checkpoint
        is written to ``merged_models/<task_id>`` and its path is available via
        the job's ``result_path``.
        """
        adapter_path = self.get_checkpoint_download_path(
            request.source_job_id, request.checkpoint_id, org_id=org_id
        )
        if not adapter_path:
            raise ValueError(
                f"Checkpoint '{request.checkpoint_id}' not found for job "
                f"'{request.source_job_id}'"
            )

        output_dir = os.path.join(TRAINING_STORE_MERGED_MODELS_DIR, request._task_id)
        request._adapter_path = adapter_path
        request._output_model_path = output_dir

        self.logger.info(
            f"Creating adapter merge job {request._task_id}: "
            f"base={self._base_model_name}, adapter={adapter_path}, output={output_dir}"
        )

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=JobTypes.ADAPTER_MERGE,
            model=self._base_model_name,
            request=request,
            task_function=self.run_adapter_merge,
            result_path=output_dir,
            org_id=org_id,
        )

    async def run_adapter_merge(self, request: AdapterMergeRequest) -> str:
        """Job task function: perform the LoRA adapter merge in a separate process.

        The merge is run in a freshly spawned subprocess (via a single-worker
        process pool) rather than a thread so that:
          - the large base-model memory footprint is fully reclaimed by the OS
            when the process exits, and
          - a crash or OOM in the merge cannot take down the API process.
        Merges are serialized via a lock to bound peak host memory usage.
        """
        async with self._adapter_merge_lock:
            self.logger.info(f"Starting adapter merge for job {request._task_id}")
            loop = asyncio.get_running_loop()
            try:
                # "spawn" gives a clean interpreter (no forked API state /
                # threads); the executor is torn down on exit from the `with`
                # block so the worker process exits and its memory is released.
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=1, mp_context=get_context("spawn")
                ) as executor:
                    output_dir = await loop.run_in_executor(
                        executor,
                        merge_adapter,
                        self._base_model_name,
                        request._adapter_path,
                        request._output_model_path,
                    )
            except Exception:
                shutil.rmtree(request._output_model_path, ignore_errors=True)
                raise
            self.logger.info(
                f"Completed adapter merge for job {request._task_id}: {output_dir}"
            )
            return output_dir
