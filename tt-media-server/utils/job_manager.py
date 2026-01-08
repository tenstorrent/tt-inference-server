# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Callable, Dict, Optional

from config.constants import JobTypes
from config.settings import get_settings
from domain.base_request import BaseRequest

from utils.logger import TTLogger


class JobStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    job_type: str
    model: str
    request_parameters: dict = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    created_at: int = None
    completed_at: Optional[int] = None
    result_path: Optional[str] = None
    error: Optional[dict] = None
    _task: Callable = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(time.time())

    def mark_in_progress(self):
        self.status = JobStatus.IN_PROGRESS

    def mark_completed(self, result_path: str):
        if self.result_path is not None and not isinstance(self.result_path, str):
            raise TypeError(f"result_path must be str, not {type(self.result_path)}")
        self.completed_at = int(time.time())
        self.status = JobStatus.COMPLETED
        self.result_path = result_path

    def is_in_progress(self) -> bool:
        return self.status == JobStatus.IN_PROGRESS

    def is_completed(self) -> bool:
        return self.status == JobStatus.COMPLETED

    def is_terminal(self) -> bool:
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED]

    def to_public_dict(self) -> dict:
        data = {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "model": self.model,
            "request_parameters": self.request_parameters,
            "result_path": self.result_path,
        }
        if self.completed_at:
            data["completed_at"] = self.completed_at
        if self.error:
            data["error"] = self.error
        return data


class JobManager:
    def __init__(self):
        self._logger = TTLogger()
        self._settings = get_settings()
        # In-memory storage for submitted jobs
        self._jobs: Dict[str, Job] = {}
        self._jobs_lock = Lock()

        self.db = None

        if self._settings.enable_job_persistence:
            from utils.job_database import JobDatabase

            self.db = JobDatabase()
            self._logger.info("Job persistence enabled with database")
            self._restore_jobs_from_db()

        # Background cleanup task
        self._cleanup_task: Callable = None
        self._start_cleanup_task()

    async def create_job(
        self,
        job_id: str,
        job_type: JobTypes,
        model: str,
        request: BaseRequest,
        task_function: Callable,
    ) -> dict:
        """Create job, start processing in background, and return initial job metadata."""
        with self._jobs_lock:
            if len(self._jobs) >= self._settings.max_jobs:
                raise Exception("Maximum job limit reached")
            job = Job(
                id=job_id,
                job_type=job_type.value,
                model=model,
                request_parameters=request.model_dump(mode="json"),
            )
            self._jobs[job_id] = job
            self._logger.info(f"Job {job_id} created.")

            if self.db:
                self.db.insert_job(
                    job_id=job.id,
                    job_type=job.job_type,
                    model=job.model,
                    request_parameters=job.request_parameters,
                    status=job.status.value,
                    created_at=job.created_at,
                )

        job._task = asyncio.create_task(self._process_job(job, request, task_function))

        return job.to_public_dict()

    def get_all_jobs_metadata(self, job_type: JobTypes = None) -> list[dict]:
        """Get metadata for all jobs, optionally filtered by job type."""
        with self._jobs_lock:
            return [
                job.to_public_dict()
                for job in self._jobs.values()
                if job_type is None or job.job_type == job_type.value
            ]

    def get_job_metadata(self, job_id: str) -> Optional[dict]:
        """Get job metadata (public fields only)."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job:
                return job.to_public_dict()
            return None

    def get_job_result_path(self, job_id: str) -> Optional[str]:
        """Get job result path if completed."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if job and job.is_terminal():
                return job.result_path if job.is_completed() else None
            return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel job, cancel if in progress, and return cancellation confirmation."""
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job:
                return False

            self._cleanup_job(job)

            self._jobs.pop(job_id)
            self._logger.info(f"Job {job_id} cancelled.")

            if self.db:
                self.db.delete_job(job_id)

            return True

    async def shutdown(self):
        """Gracefully shutdown job manager and cleanup task."""
        self._logger.info("Shutting down job manager")

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        running_tasks = []
        with self._jobs_lock:
            for job in list(self._jobs.values()):
                task = self._cleanup_job(job)
                if task:
                    running_tasks.append(task)

        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)

        self._logger.info("Job manager shutdown complete")

    async def _process_job(self, job: Job, request: BaseRequest, task_function):
        try:
            job.mark_in_progress()
            if self.db:
                self.db.update_job_status(job.id, job.status.value)

            result_path = await task_function(request)

            job.mark_completed(result_path=result_path)
            if self.db:
                self.db.update_job_status(
                    job.id,
                    job.status.value,
                    completed_at=job.completed_at,
                    result_path=result_path,
                )

        except asyncio.CancelledError:
            self._logger.info(f"Job {job.id} was cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Job {job.id} failed: {e}")
            job.mark_failed(error_code="processing_error", error_message=str(e))
            if self.db:
                self.db.update_job_status(
                    job.id,
                    job.status.value,
                    completed_at=job.completed_at,
                    error_message={"code": "processing_error", "message": str(e)},
                )
        finally:
            job._task = None

    def _start_cleanup_task(self):
        """Start background task to periodically clean up old jobs."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self._settings.job_cleanup_interval_seconds)
                    self._cleanup_old_jobs()
                except asyncio.CancelledError:
                    self._logger.info("Job cleanup task cancelled")
                    break
                except Exception as e:
                    self._logger.error(f"Error in job cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        self._logger.info("Job cleanup task started")

    def _cleanup_old_jobs(self):
        """Remove old completed, failed and stuck in-progress jobs."""
        current_time = time.time()
        cutoff_time = current_time - self._settings.job_retention_seconds
        stuck_cutoff_time = current_time - self._settings.job_max_stuck_time_seconds

        jobs_to_remove = []

        with self._jobs_lock:
            for job_id, job in self._jobs.items():
                if (
                    job.is_terminal()
                    and job.completed_at
                    and job.completed_at < cutoff_time
                ) or (job.is_in_progress() and job.created_at < stuck_cutoff_time):
                    jobs_to_remove.append(job_id)
                    self._cleanup_job(job)

        if jobs_to_remove:
            with self._jobs_lock:
                for job_id in jobs_to_remove:
                    self._jobs.pop(job_id, None)
                    if self.db:
                        self.db.delete_job(job_id)

            self._logger.info(
                f"Cleaned up {len(jobs_to_remove)} old job(s): {', '.join(jobs_to_remove)}"
            )

    def _cleanup_job(self, job: Job):
        running_task = None
        if job._task and not job._task.done():
            self._logger.warning(f"Cancelling in-progress job {job.id}")
            job._task.cancel()
            running_task = job._task

        # Delete result file if it's a file path
        if job.result_path and isinstance(job.result_path, str):
            try:
                if os.path.exists(job.result_path):
                    os.remove(job.result_path)
                    self._logger.debug(f"Deleted file for job {job.id}: {job.result_path}")
            except Exception as e:
                self._logger.debug(f"Failed to delete file for job {job.id}: {e}")

        return running_task

    def _restore_jobs_from_db(self):
        """Restore all jobs from database on server restart."""
        if not self.db:
            return

        try:
            db_jobs = self.db.get_all_jobs()

            for db_job in db_jobs:
                job = Job(
                    id=db_job["id"],
                    job_type=db_job["job_type"],
                    request_parameters=db_job["request_parameters"],
                    status=JobStatus(db_job["status"]),
                    created_at=db_job["created_at"],
                    completed_at=db_job.get("completed_at"),
                    error=db_job.get("error"),
                )

                with self._jobs_lock:
                    self._jobs[job.id] = job

                self._logger.debug(
                    f"Restored job {job.id} from database (status: {job.status.value})"
                )

            restored_count = len(db_jobs)
            if restored_count > 0:
                self._logger.info(f"Restored {restored_count} job(s) from database")
        except Exception as e:
            self._logger.error(f"Failed to restore jobs from database: {e}")


_job_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance
