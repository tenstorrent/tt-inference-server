# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Callable, Dict, Optional, Any
from pathlib import Path
from multiprocessing import Event
from sqlite3 import IntegrityError

from config.constants import JobTypes
from config.settings import get_settings
from domain.base_request import BaseRequest
from utils.logger import TTLogger


class JobStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"


@dataclass
class Job:
    id: str
    job_type: str
    model: str
    request_parameters: dict = field(default_factory=dict)
    org_id: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: int = None
    completed_at: Optional[int] = None
    result_path: Optional[str] = None
    error: Optional[dict] = None
    _task: Callable = None
    start_event: Optional[Event] = None
    cancel_event: Optional[Event] = None
    job_metrics: list = field(default_factory=list)
    job_logs: list = field(default_factory=list)
    job_checkpoints: list = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(time.time())

    def mark_in_progress(self):
        self.status = JobStatus.IN_PROGRESS

    def mark_completed(self, result_path: str):
        self.completed_at = int(time.time())
        self.status = JobStatus.COMPLETED
        self.result_path = result_path

    def mark_cancelling(self):
        self.status = JobStatus.CANCELLING

    def mark_cancelled(self):
        self.status = JobStatus.CANCELLED
        self.completed_at = int(time.time())

    def mark_failed(self, error_code: str, error_message: str):
        self.completed_at = int(time.time())
        self.status = JobStatus.FAILED
        self.error = {"code": error_code, "message": error_message}

    def is_in_progress(self) -> bool:
        return self.status == JobStatus.IN_PROGRESS

    def is_cancelling(self) -> bool:
        return self.status == JobStatus.CANCELLING

    def is_completed(self) -> bool:
        return self.status == JobStatus.COMPLETED

    def is_terminal(self) -> bool:
        return self.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ]

    def to_public_dict(self) -> dict:
        data = {
            "id": self.id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "model": self.model,
            "request_parameters": self.request_parameters,
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

            self.db = JobDatabase(db_path=Path(self._settings.job_database_path))
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
        result_path: Optional[str] = None,
        start_event: Optional[Event] = None,
        cancel_event: Optional[Event] = None,
        job_metrics: list = None,
        job_logs: list = None,
        job_checkpoints: list = None,
        org_id: Optional[str] = None,
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
                org_id=org_id,
            )

            if result_path:
                job.result_path = result_path
            if start_event:
                job.start_event = start_event
            if cancel_event:
                job.cancel_event = cancel_event
            if job_metrics is not None:
                job.job_metrics = job_metrics
            if job_logs is not None:
                job.job_logs = job_logs
            if job_checkpoints is not None:
                job.job_checkpoints = job_checkpoints

            if self.db:
                try:
                    self.db.insert_job(
                        job_id=job.id,
                        job_type=job.job_type,
                        model=job.model,
                        request_parameters=job.request_parameters,
                        status=job.status.value,
                        created_at=job.created_at,
                        org_id=job.org_id,
                    )
                    if result_path:
                        self.db.update_result_path(job_id, result_path)
                except Exception as e:
                    self._logger.error(
                        f"Failed to insert job {job_id} into database: {e}"
                    )
                    raise

            # we only add the job to the in-memory storage if the database insert was successful
            self._jobs[job_id] = job
            self._logger.info(f"Job {job_id} created.")

        job._task = asyncio.create_task(self._process_job(job, request, task_function))

        return job.to_public_dict()

    def _get_job_if_authorized(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[Job]:
        """Retrieve a job by ID, returning None if org_id is set and doesn't match."""
        job = self._jobs.get(job_id)
        if job is None or (org_id is not None and job.org_id != org_id):
            return None
        return job

    def get_all_jobs_metadata(
        self, job_type: JobTypes = None, org_id: Optional[str] = None
    ) -> list[dict]:
        """Get metadata for all jobs, optionally filtered by job type and org."""
        with self._jobs_lock:
            return [
                job.to_public_dict()
                for job in self._jobs.values()
                if (job_type is None or job.job_type == job_type.value)
                and (org_id is None or job.org_id == org_id)
            ]

    def get_job_metadata(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[dict]:
        """Get job metadata (public fields only)."""
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job:
                return job.to_public_dict()
            return None

    def get_job_result_path(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[str]:
        """Get job result path if completed."""
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job:
                if job.job_type != JobTypes.TRAINING.value and job.is_terminal():
                    return job.result_path if job.is_completed() else None
                else:
                    return job.result_path
            return None

    def get_job_metrics(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[list]:
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job:
                return job.job_metrics
        return None

    def get_job_logs(self, job_id: str, org_id: Optional[str] = None) -> Optional[list]:
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job:
                return job.job_logs
        return None

    def get_job_checkpoints(
        self, job_id: str, org_id: Optional[str] = None
    ) -> Optional[list]:
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job:
                return job.job_checkpoints
        return None

    def cancel_job(self, job_id: str, org_id: Optional[str] = None) -> bool:
        """Cancel job, cancel if in progress, and return cancellation confirmation."""
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if not job:
                self._logger.warning(f"Cancel failed: Job {job_id} not found.")
                return None

            if job.is_terminal():
                self._logger.warning(
                    f"Cancel failed: Job {job_id} is already {job.status.value}."
                )
                return None

            # if the job is queued, we can cancel it immediately
            if job.status == JobStatus.QUEUED:
                self._cleanup_job(job, force=True)
                job.mark_cancelled()
                self._sync_status_to_db(job)
                self._logger.info(f"Queued job {job_id} cancelled immediately.")
                return job.to_public_dict()

            job.mark_cancelling()
            self._sync_status_to_db(job)

            self._cleanup_job(job)

            self._logger.info(f"Job {job_id} cancellation initiated.")
            return job.to_public_dict()

    async def shutdown(self):
        """Gracefully shutdown job manager and transition active jobs to terminal states."""
        self._logger.info("Shutting down job manager")

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        running_tasks = []
        with self._jobs_lock:
            for job_id in list(self._jobs.keys()):
                job = self._jobs[job_id]

                if not job.is_terminal():
                    self._logger.info(
                        f"Terminating active job {job_id} during shutdown."
                    )

                    job.mark_cancelling()
                    self._sync_status_to_db(job)

                    # force the job to be cancelled without waiting for the runner to handle it, since we are shutting down the server
                    task = self._cleanup_job(job, force=True)
                    if task:
                        running_tasks.append(task)

                # Always remove from memory tracking during shutdown
                self._jobs.pop(job_id)

        if running_tasks:
            await asyncio.gather(*running_tasks, return_exceptions=True)
        self._logger.info("Job manager shutdown complete")

    async def _mark_job_in_progress(self, job: Job):
        if job.start_event:
            while not job.start_event.is_set():
                await asyncio.sleep(0.5)

        job.mark_in_progress()
        self._sync_status_to_db(job)

    async def _process_job(self, job: Job, request: BaseRequest, task_function):
        data_persister = None
        try:
            progress_monitor = asyncio.create_task(self._mark_job_in_progress(job))

            if self.db:
                data_persister = asyncio.create_task(self._persist_job_data_to_db(job))

            result_path = await task_function(request)

            # enforcing result_path to be a string
            if result_path is not None and not isinstance(result_path, str):
                raise TypeError(f"result_path must be str, not {type(result_path)}")
            # for training job types, the path is set on job creation, and needs to be the same as the path returned by the task function
            if job.result_path and job.result_path != result_path:
                raise ValueError(
                    f"The initially set result_path differs from the path returned by the task function ({job.result_path} != {result_path})."
                )

            if job.status == JobStatus.CANCELLING:
                self._logger.info(f"Job {job.id} was cooperatively cancelled by runner")
                job.mark_cancelled()
                self._sync_status_to_db(job)
                return  # we return here to avoid marking the job as completed

            job.mark_completed(result_path=result_path)
            self._sync_status_to_db(job)

        except asyncio.CancelledError:
            self._logger.info(f"Job {job.id} was cancelled")
            if not job.is_terminal():
                job.mark_cancelled()
                self._sync_status_to_db(job)
            self._cleanup_job(job)
            raise
        except Exception as e:
            self._logger.error(f"Job {job.id} failed: {e}")
            job.mark_failed(error_code="processing_error", error_message=str(e))
            self._sync_status_to_db(job)
            if job.cancel_event:
                self._cleanup_job(job)
        finally:
            if not progress_monitor.done():
                progress_monitor.cancel()
            if data_persister and not data_persister.done():
                await data_persister
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
        """Remove old completed/failed/cancelled, stuck in-progress, and stale cancelling jobs."""
        current_time = time.time()
        cutoff_time = current_time - self._settings.job_retention_seconds
        stuck_cutoff_time = current_time - self._settings.job_max_stuck_time_seconds

        jobs_to_remove = []

        with self._jobs_lock:
            for job_id, job in self._jobs.items():
                is_old_terminal = (
                    job.is_terminal()
                    and job.completed_at
                    and job.completed_at < cutoff_time
                )
                is_stuck = (
                    job.is_in_progress() or job.is_cancelling()
                ) and job.created_at < stuck_cutoff_time
                if is_old_terminal or is_stuck:
                    jobs_to_remove.append(job)

        if not jobs_to_remove:
            return

        for job in jobs_to_remove:
            if job.is_in_progress() or job.is_cancelling():
                if job.is_in_progress():
                    self._logger.warning(
                        f"Force-cancelling stuck in-progress job {job.id}"
                    )
                else:
                    self._logger.warning(
                        f"Force-cancelling stale cancelling job {job.id}"
                    )
                self._cleanup_job(job, force=True)
                job.mark_failed(
                    error_code="stale_job",
                    error_message="Job was stuck and force-cancelled by cleanup",
                )
                self._sync_status_to_db(job)
            if job.result_path and isinstance(job.result_path, str):
                try:
                    if os.path.exists(job.result_path):
                        os.remove(job.result_path)
                        self._logger.debug(
                            f"Deleted file for job {job.id}: {job.result_path}"
                        )
                except Exception as e:
                    self._logger.debug(f"Failed to delete file for job {job.id}: {e}")

        # Remove from storage under lock
        with self._jobs_lock:
            for job in jobs_to_remove:
                self._jobs.pop(job.id, None)
                if self.db:
                    try:
                        self.db.delete_job(job.id)
                    except Exception as e:
                        self._logger.error(
                            f"Database deletion failed for job {job.id} during cleanup: {e}"
                        )

            self._logger.info(
                f"Cleaned up {len(jobs_to_remove)} old job(s): {', '.join(job.id for job in jobs_to_remove)}"
            )

    def _cleanup_job(self, job: Job, force: bool = False):
        running_task = None
        if job._task and not job._task.done():
            self._logger.warning(f"Cancelling active job {job.id}")
            if job.cancel_event:
                job.cancel_event.set()
            if not job.cancel_event or force:
                job._task.cancel()
            running_task = job._task

        return running_task

    def _sync_status_to_db(self, job: Job, **overrides):
        if not self.db:
            return
        try:
            self.db.update_job_status(
                job.id,
                job.status.value,
                completed_at=overrides.get("completed_at", job.completed_at),
                result_path=overrides.get("result_path", job.result_path),
                error_message=overrides.get("error_message", job.error),
            )
        except Exception as e:
            self._logger.error(
                f"DB sync failed for job {job.id} to '{job.status.value}': {e}"
            )

    async def _persist_job_data_to_db(self, job: Job):
        if job.job_type != JobTypes.TRAINING.value:
            return
        streams = [
            (self.get_job_metrics(job.id), self._insert_metric, "metric"),
            (self.get_job_checkpoints(job.id), self._insert_checkpoint, "checkpoint"),
            (self.get_job_logs(job.id), self._insert_log, "log"),
        ]
        last_seen = [0] * len(streams)
        failed = [items is None for items, _, _ in streams]
        while True:
            for i, (items, insert_fn, label) in enumerate(streams):
                if failed[i]:
                    continue
                last_seen[i] = self._persist_new_items(
                    job, items, last_seen[i], insert_fn, label
                )
                if last_seen[i] < 0:
                    failed[i] = True
            if job.is_terminal():
                break
            await asyncio.sleep(1.0)

    def _persist_new_items(
        self,
        job: Job,
        items_list: list,
        last_seen: int,
        insert_fn: Callable[[str, Any, int], None],
        item_label: str,
    ) -> int:
        current_len = len(items_list)
        for i in range(last_seen, current_len):
            try:
                insert_fn(job.id, items_list[i], i)
            except IntegrityError:
                self._logger.warning(
                    f"Duplicate {item_label} for job {job.id}: {items_list[i]}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to persist {item_label} for job {job.id}: {e}"
                )
                return -1
        return current_len

    def _insert_metric(self, job_id: str, metric: dict, _index: int):
        self.db.insert_metric(
            job_id=job_id,
            global_step=metric["global_step"],
            epoch=metric["epoch"],
            metric_name=metric["metric_name"],
            value=metric["value"],
            learning_rate=metric.get("learning_rate"),
            timestamp=metric["timestamp"],
        )

    def _insert_checkpoint(self, job_id: str, ckpt: dict, _index: int):
        self.db.insert_checkpoint(
            job_id=job_id,
            checkpoint_id=ckpt["id"],
            step=ckpt["step"],
            epoch=ckpt["epoch"],
            metrics=ckpt.get("metrics", {}),
            created_at=ckpt["created_at"],
        )

    def _insert_log(self, job_id: str, log_entry: dict, index: int):
        self.db.insert_log(
            job_id=job_id,
            log_index=index,
            timestamp=log_entry["timestamp"],
            log_type=log_entry["type"],
            step=log_entry.get("step"),
            message=log_entry["message"],
        )

    def _restore_jobs_from_db(self):
        """
        Restore all jobs from database on server restart.
        Mark stuck jobs as failed or cancelled.
        """
        if not self.db:
            return

        try:
            db_jobs = self.db.get_all_jobs()
            restored_jobs = {}
            for db_job in db_jobs:
                original_status = db_job["status"]

                job = Job(
                    id=db_job["id"],
                    job_type=db_job["job_type"],
                    model=db_job["model"],
                    request_parameters=db_job["request_parameters"],
                    org_id=db_job.get("org_id"),
                    status=JobStatus(original_status),
                    created_at=db_job["created_at"],
                    completed_at=db_job.get("completed_at"),
                    result_path=db_job.get("result_path"),
                    error=db_job.get("error_message"),
                )

                if db_job["job_type"] == JobTypes.TRAINING.value:
                    try:
                        metrics = self.db.get_metrics_flat(job.id)
                        if metrics:
                            job.job_metrics = metrics
                    except Exception as e:
                        self._logger.error(
                            f"Failed to restore metrics for job {job.id}: {e}"
                        )
                    try:
                        checkpoints = self.db.get_checkpoints(job.id)
                        if checkpoints:
                            job.job_checkpoints = checkpoints
                        if job.job_checkpoints:
                            job.job_checkpoints = self._validate_checkpoints_on_disk(
                                job
                            )
                    except Exception as e:
                        self._logger.error(
                            f"Failed to restore checkpoints for job {job.id}: {e}"
                        )
                    try:
                        logs = self.db.get_logs(job.id)
                        if logs:
                            job.job_logs = logs
                    except Exception as e:
                        self._logger.error(
                            f"Failed to restore logs for job {job.id}: {e}"
                        )

                # If job was stuck, mark it as failed or cancelled and sync to database
                if not job.is_terminal():
                    if original_status == "cancelling":
                        job.mark_cancelled()
                    else:
                        job.mark_failed(
                            "server_restart", "Job interrupted by system restart"
                        )

                    # we override the completed_at time with the creation time, since we don't know the time of the system restart
                    job.completed_at = db_job["created_at"]

                    self._logger.warning(
                        f"Job {db_job['id']} was stuck in '{original_status}'. "
                        f"Syncing to '{job.status.value}'."
                    )
                    try:
                        self.db.update_job_status(
                            job.id,
                            job.status.value,
                            completed_at=job.completed_at,
                            error_message=job.error,
                        )
                    except Exception as e:
                        self._logger.error(
                            f"Failed to sync corrected status for job {job.id} to DB: {e}"
                        )
                else:
                    self._logger.debug(
                        f"Restored job {job.id} from database (status: {job.status.value})"
                    )

                restored_jobs[job.id] = job

            with self._jobs_lock:
                self._jobs.update(restored_jobs)
            if restored_jobs:
                self._logger.info(f"Restored {len(restored_jobs)} job(s) from database")

        except Exception as e:
            self._logger.error(f"Failed to restore jobs from database: {e}")

    def _validate_checkpoints_on_disk(self, job: Job) -> list:
        """Filter out checkpoints whose directories no longer exist on disk."""
        if not job.job_checkpoints:
            # no checkpoints to validate
            return job.job_checkpoints
        if not job.result_path:
            self._logger.warning(
                f"Job {job.id} has checkpoints but no result_path, clearing checkpoints"
            )
            return []
        try:
            existing_entries = set(os.listdir(job.result_path))
        except FileNotFoundError:
            self._logger.warning(
                f"Result path {job.result_path} for job {job.id} not found on disk, clearing all checkpoints"
            )
            return []
        valid_checkpoints = []
        for ckpt in job.job_checkpoints:
            if ckpt["id"] in existing_entries:
                valid_checkpoints.append(ckpt)
            else:
                self._logger.warning(
                    f"Checkpoint '{ckpt['id']}' for job {job.id} not found on disk, removing from restored data"
                )
        return valid_checkpoints


_job_manager_instance: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance
