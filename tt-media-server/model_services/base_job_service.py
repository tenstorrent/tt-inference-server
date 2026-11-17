# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import asyncio
import os
import time
from multiprocessing import Manager
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
        self.scheduler.on_request_dropped = self._fail_dropped_job
        # Forked before the port opens, so it inherits no client connections.
        self._processManager = Manager()

    def _createStartEvent(self):
        return self._processManager.Event()

    async def finish_running_jobs(self) -> None:
        """Stopping: fail the queued jobs, then let the running one end, its
        output written, within the worker's stop budget. The job manager's
        shutdown would cancel it, and only the device would finish it."""
        deadline = time.monotonic() + settings.worker_shutdown_timeout_seconds
        # stop_workers' join gets what is left: one budget for the whole stop
        self.scheduler.stop_deadline = deadline
        self.scheduler.drop_queued_requests()
        while time.monotonic() < deadline:
            active = self._job_manager.count_active_jobs()
            if not (active["queued"] or active["in_progress"] or active["cancelling"]):
                return
            await asyncio.sleep(0.2)
        self.logger.warning("Stopping: a job still runs after the stop budget")

    def _fail_dropped_job(self, job_id: str) -> None:
        self._job_manager.fail_job(
            job_id, "server_stopping", "the server stopped before the job ran"
        )

    def stop_workers(self):
        result = super().stop_workers()
        # Not left to atexit: uvicorn exits a SIGTERM shutdown by re-raising it.
        if self._processManager is not None:
            self._processManager.shutdown()
            self._processManager = None
        return result

    async def create_job(
        self,
        job_type: JobTypes,
        request: BaseRequest,
        org_id: Optional[str] = None,
        request_parameters: Optional[dict] = None,
    ) -> dict:
        startEvent = self._createStartEvent()
        request._start_event = startEvent
        return await self._job_manager.create_job(
            job_id=request._task_id,
            job_type=job_type,
            model=os.environ.get("SERVED_MODEL_NAME") or settings.model_weights_path,
            request=request,
            task_function=self.process_request,
            start_event=startEvent,
            org_id=org_id,
            request_parameters=request_parameters,
        )

    def get_all_jobs_metadata(
        self, job_type: JobTypes = None, org_id: Optional[str] = None
    ) -> list[dict]:
        return self._job_manager.get_all_jobs_metadata(job_type, org_id=org_id)

    def get_job_metadata(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[dict]:
        return self._job_manager.get_job_metadata(job_id, org_id=org_id)

    def get_job_result_path(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[Any]:
        return self._job_manager.get_job_result_path(job_id, org_id=org_id)

    def cancel_job(self, job_id: str, org_id: Optional[str] = None) -> Optional[dict]:
        return self._job_manager.cancel_job(job_id, org_id=org_id)

    async def delete_job(self, job_id: str, org_id: Optional[str] = None) -> bool:
        return await self._job_manager.delete_job(job_id, org_id=org_id)

    def get_job_metrics(self, job_id: str, org_id: Optional[str] = None) -> list:
        return self._job_manager.get_job_metrics(job_id, org_id=org_id)

    def get_job_logs(self, job_id: str, org_id: Optional[str] = None) -> list:
        return self._job_manager.get_job_logs(job_id, org_id=org_id)

    def get_job_checkpoints(self, job_id: str, org_id: Optional[str] = None) -> list:
        return self._job_manager.get_job_checkpoints(job_id, org_id=org_id)
