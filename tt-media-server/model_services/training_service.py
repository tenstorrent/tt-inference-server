# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import asyncio
import functools
import json
import os
import re
import shutil
import tempfile
import time
from multiprocessing import Manager

from model_services.base_job_service import BaseJobService
from config.constants import (
    JobTypes,
    ModelNames,
    SupportedModels,
    adapters_root,
    merged_models_root,
)
from config.settings import get_settings
from domain.adapter_merge_request import AdapterMergeRequest
from domain.training_request import TrainingRequest
from typing import Optional
from utils.adapter_merge_utils import run_merge_subprocess


MERGE_INFO_FILE_NAME = "merge_info.json"


class TrainingService(BaseJobService):
    def __init__(self):
        self.settings = get_settings()
        self._manager = Manager()
        self._model_name = ModelNames(self.settings.training_model).value
        # Base model HF repo id backing the configured training model; used when
        # merging LoRA adapters back into the base weights.
        self._base_model_hf_repo_id = SupportedModels[
            ModelNames(self.settings.training_model).name
        ].value
        self._adapter_merge_lock = asyncio.Lock()
        super().__init__()

    async def create_job(
        self,
        job_type: JobTypes,
        request: TrainingRequest,
        org_id: Optional[str] = None,
    ) -> dict:
        request.device_type = self.settings.device
        adapter_path = os.path.join(adapters_root(), request._task_id)
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

        # `checkpoint_id` is a user-supplied URL segment, so
        # confirm the resolved path stays strictly inside the job's result dir
        if not self._is_within_dir(result_path, checkpoint_path):
            self.logger.warning(
                f"Rejecting checkpoint path outside job dir: {checkpoint_id!r}"
            )
            return None
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        return None

    async def create_adapter_merge_job(
        self, request: AdapterMergeRequest, org_id: Optional[str] = None
    ) -> dict:
        adapter_path = self.get_checkpoint_download_path(
            request.source_job_id, request.checkpoint_id, org_id=org_id
        )
        if not adapter_path:
            raise ValueError(
                f"Checkpoint '{request.checkpoint_id}' not found for job "
                f"'{request.source_job_id}'"
            )

        merged_model_dir_name = re.sub(
            r"[^a-zA-Z0-9_.-]", "-", f"{self._model_name}-{request._task_id}"
        )
        output_dir = os.path.join(merged_models_root(), merged_model_dir_name)
        request._adapter_path = adapter_path
        request._output_model_path = output_dir

        self.logger.info(
            f"Creating adapter merge job {request._task_id}: "
            f"base={self._base_model_hf_repo_id}, adapter={adapter_path}, output={output_dir}"
        )

        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=JobTypes.ADAPTER_MERGE,
            model=self._base_model_hf_repo_id,
            request=request,
            task_function=self.run_adapter_merge,
            result_path=output_dir,
            org_id=org_id,
        )

    async def run_adapter_merge(self, request: AdapterMergeRequest) -> str:
        """
        The merge runs in a dedicated virtual environment,
        launched as a subprocess so that:
          - the checkpoint is written by the same major `transformers` version
            that the vLLM inference container serves it.
          - the large base-model memory footprint is fully reclaimed by the OS
            when the process exits, and
          - a crash or OOM in the merge cannot take down the API process.
          - merges are serialized via a lock to allow only one merge at a time
            and to bound peak host memory usage.
        """
        async with self._adapter_merge_lock:
            self.logger.info(f"Starting adapter merge for job {request._task_id}")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        run_merge_subprocess,
                        self._base_model_hf_repo_id,
                        request._adapter_path,
                        request._output_model_path,
                    ),
                )
            except Exception:
                self._safe_rmtree_under_root(request._output_model_path)
                raise
            # Written last, in the parent, so its presence marks the checkpoint
            # fully complete and records provenance for `list_merged_checkpoints`.
            self._write_merge_info(request)
            self.logger.info(
                f"Completed adapter merge for job {request._task_id}: "
                f"{request._output_model_path}"
            )
            return request._output_model_path

    @staticmethod
    def _is_within_dir(base: str, path: str) -> bool:
        """True if `path` resolves to a location strictly inside `base`.
        Rejects traversal and absolute-path escapes when
        `path` was built from untrusted input.
        """
        base = os.path.realpath(base)
        target = os.path.realpath(path)
        return target != base and os.path.commonpath([base, target]) == base

    def _safe_rmtree_under_root(self, path: str) -> None:
        """Delete `path` only if it resolves to a location strictly inside the
        merged-models root."""
        if self._is_within_dir(merged_models_root(), path):
            shutil.rmtree(os.path.realpath(path), ignore_errors=True)
        else:
            self.logger.error(
                f"Refusing to delete path outside merged-models root: {path!r}"
            )

    def _write_merge_info(self, request: AdapterMergeRequest) -> None:
        # Enforce locally that we only ever write inside the merged-models root,
        # rather than trusting the caller to have set a safe output path.
        if not self._is_within_dir(merged_models_root(), request._output_model_path):
            raise ValueError(
                f"Refusing to write merge info outside merged-models root: "
                f"{request._output_model_path!r}"
            )
        info = {
            "merge_id": request._task_id,
            "model": self._base_model_hf_repo_id,
            "source_job_id": request.source_job_id,
            "checkpoint_id": request.checkpoint_id,
            "created_at": time.time(),
        }
        info_path = os.path.join(request._output_model_path, MERGE_INFO_FILE_NAME)

        fd, tmp_path = tempfile.mkstemp(
            dir=request._output_model_path, prefix=".merge_info.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(info, f)
            os.replace(tmp_path, info_path)
        except Exception:
            os.unlink(tmp_path)
            raise
