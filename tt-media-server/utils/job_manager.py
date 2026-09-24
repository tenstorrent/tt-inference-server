# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Event
from pathlib import Path
from sqlite3 import IntegrityError
from threading import Lock
from typing import Any, Callable, Dict, Optional

from config.constants import (
    JobTypes,
    adapters_root,
    job_database_path,
    merged_models_root,
)
from config.settings import get_settings
from domain.base_request import BaseRequest
from fastapi import HTTPException
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_503_SERVICE_UNAVAILABLE

from utils.logger import TTLogger

TASK_QUEUE_FULL_DETAIL = "Task queue is full. Please try again later."
MAX_JOBS_REACHED_DETAIL = "Maximum job limit reached"
AUTO_CLEANUP_EXEMPT_JOB_TYPES = frozenset(
    {
        JobTypes.TRAINING.value,
        JobTypes.ADAPTER_MERGE.value,
    }
)


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
    adapter_merge_job_ids: set[str] = field(default_factory=set)
    local_progress_time: Optional[float] = None
    _task: Callable = None
    _progress_tracker: Any = None
    start_event: Optional[Event] = None
    cancel_event: Optional[Event] = None
    job_metrics: list = field(default_factory=list)
    job_logs: list = field(default_factory=list)
    job_checkpoints: list = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = int(time.time())
        if self.local_progress_time is None:
            self.local_progress_time = time.monotonic()

    def mark_in_progress(self):
        self.status = JobStatus.IN_PROGRESS
        if self._progress_tracker is not None:
            self.touch_progress()

    def touch_progress(self) -> float:
        """Record progress locally and in the shared worker heartbeat."""
        self.local_progress_time = time.monotonic()
        if self._progress_tracker is not None:
            self._progress_tracker.value = self.local_progress_time
        return self.local_progress_time

    def progress_time(self) -> float:
        """Return worker progress when shared, otherwise local job progress."""
        if self._progress_tracker is not None:
            return float(self._progress_tracker.value)
        return self.local_progress_time

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
        if self.job_type == JobTypes.TRAINING.value:
            data["adapter_merge_job_ids"] = sorted(self.adapter_merge_job_ids)
        if self.org_id:
            data["org_id"] = self.org_id
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
        self._deleting_job_ids: set[str] = set()
        self._jobs_lock = Lock()

        self.db = None

        if self._settings.enable_job_persistence:
            from utils.job_database import JobDatabase

            self.db = JobDatabase(db_path=Path(job_database_path()))
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
        progress_tracker: Any = None,
        org_id: Optional[str] = None,
    ) -> dict:
        """Create job, start processing in background, and return initial job metadata."""
        with self._jobs_lock:
            self._enforceAdmissionLimits()
            request_parameters = request.model_dump(mode="json")

            job = Job(
                id=job_id,
                job_type=job_type.value,
                model=model,
                request_parameters=request_parameters,
                org_id=org_id,
                _progress_tracker=progress_tracker,
            )

            parent_job = None
            if job_type == JobTypes.ADAPTER_MERGE:
                source_job_id = request_parameters.get("source_job_id")
                if not source_job_id:
                    raise ValueError("Adapter merge jobs must provide source_job_id")
                parent_job = self._get_job_if_authorized(source_job_id, org_id)
                if parent_job is None or parent_job.job_type != JobTypes.TRAINING.value:
                    raise ValueError(f"Training job '{source_job_id}' not found")

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
            if parent_job is not None:
                parent_job.adapter_merge_job_ids.add(job.id)
            self._logger.info(f"Job {job_id} created.")

        job._task = asyncio.create_task(self._process_job(job, request, task_function))

        return job.to_public_dict()

    def _enforceAdmissionLimits(self) -> None:
        """Reject new work when outstanding jobs or stored records are full."""
        activeCount = sum(1 for job in self._jobs.values() if not job.is_terminal())
        if activeCount >= self._settings.max_queue_size:
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail=TASK_QUEUE_FULL_DETAIL,
            )
        if len(self._jobs) >= self._settings.max_jobs:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail=MAX_JOBS_REACHED_DETAIL,
            )

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

    def delete_job(
        self,
        job_id: str,
        org_id: Optional[str] = None,
    ) -> bool:
        """Delete a terminal job, its child merge jobs, and their result artifacts."""
        with self._jobs_lock:
            job = self._get_job_if_authorized(job_id, org_id)
            if job is None:
                return False

            jobs_to_delete = self._collect_jobs_for_manual_deletion(job)
            self._deleting_job_ids.update(
                job_to_delete.id for job_to_delete in jobs_to_delete
            )
            for job_to_delete in jobs_to_delete:
                self._jobs.pop(job_to_delete.id, None)

        try:
            self._delete_jobs_and_results(jobs_to_delete)
        except Exception:
            with self._jobs_lock:
                for job_to_restore in jobs_to_delete:
                    self._jobs.setdefault(job_to_restore.id, job_to_restore)
                    self._deleting_job_ids.discard(job_to_restore.id)
            raise

        with self._jobs_lock:
            for deleted_job in jobs_to_delete:
                self._deleting_job_ids.discard(deleted_job.id)
                if deleted_job.job_type != JobTypes.ADAPTER_MERGE.value:
                    continue
                source_job_id = deleted_job.request_parameters.get("source_job_id")
                parent_job = self._jobs.get(source_job_id)
                if parent_job is not None:
                    parent_job.adapter_merge_job_ids.discard(deleted_job.id)

        self._logger.info(
            f"Manually deleted {len(jobs_to_delete)} job(s): "
            f"{', '.join(job.id for job in jobs_to_delete)}"
        )
        return True

    def _collect_jobs_for_manual_deletion(self, job: Job) -> list[Job]:
        """Collect the job and its merge children while ``_jobs_lock`` is held."""
        jobs_to_delete = [job]
        if job.job_type == JobTypes.TRAINING.value:
            for child_job_id in sorted(job.adapter_merge_job_ids):
                if child_job_id in self._deleting_job_ids:
                    raise ValueError(
                        f"Adapter merge job '{child_job_id}' is being deleted"
                    )
                child_job = self._jobs.get(child_job_id)
                if child_job is not None:
                    jobs_to_delete.append(child_job)

        for job_to_delete in jobs_to_delete:
            if job_to_delete.id in self._deleting_job_ids:
                raise ValueError(f"Job '{job_to_delete.id}' is being deleted")
            if not job_to_delete.is_terminal():
                raise ValueError(
                    f"Only terminal jobs can be deleted; job "
                    f"'{job_to_delete.id}' is {job_to_delete.status.value}"
                )
        return jobs_to_delete

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
        retention_cutoff = time.time() - self._settings.job_retention_seconds
        progress_cutoff = time.monotonic() - self._settings.job_max_stuck_time_seconds

        jobs_to_remove = []
        stuck_jobs = []

        with self._jobs_lock:
            for job in self._jobs.values():
                is_old_terminal = (
                    job.is_terminal()
                    and job.job_type not in AUTO_CLEANUP_EXEMPT_JOB_TYPES
                    and job.completed_at
                    and job.completed_at < retention_cutoff
                )
                if is_old_terminal:
                    jobs_to_remove.append(job)
                elif self._is_job_stuck(job, progress_cutoff):
                    stuck_jobs.append(job)

        if not jobs_to_remove and not stuck_jobs:
            return

        for job in stuck_jobs:
            # Progress and completion happen outside _jobs_lock. Claim the job as
            # failed only after re-reading both immediately before cancellation.
            with self._jobs_lock:
                current_job = self._jobs.get(job.id)
                if current_job is not job:
                    continue
                latest_progress_cutoff = (
                    time.monotonic() - self._settings.job_max_stuck_time_seconds
                )
                if not self._is_job_stuck(job, latest_progress_cutoff):
                    continue
                was_in_progress = job.is_in_progress()
                job.mark_failed(
                    error_code="stale_job",
                    error_message=(
                        "Job made no progress and was force-cancelled by cleanup"
                    ),
                )

            if was_in_progress:
                self._logger.warning(f"Force-cancelling stuck in-progress job {job.id}")
            else:
                self._logger.warning(f"Force-cancelling stale cancelling job {job.id}")
            self._cleanup_job(job, force=True)
            self._sync_status_to_db(job)

        removed_jobs = []
        with self._jobs_lock:
            for job in jobs_to_remove:
                current_job = self._jobs.get(job.id)
                if current_job is not job:
                    continue
                self._jobs.pop(job.id, None)
                removed_jobs.append(job)

        deleted_jobs = []
        for job in removed_jobs:
            try:
                self._delete_jobs_and_results([job])
            except Exception as e:
                self._logger.error(
                    f"Deletion failed for job {job.id} during cleanup: {e}"
                )
                with self._jobs_lock:
                    self._jobs.setdefault(job.id, job)
                continue
            deleted_jobs.append(job)

        if deleted_jobs:
            self._logger.info(
                f"Deleted {len(deleted_jobs)} old job(s): "
                f"{', '.join(job.id for job in deleted_jobs)}"
            )

    def _is_job_stuck(self, job: Job, progress_cutoff: float) -> bool:
        if not (job.is_in_progress() or job.is_cancelling()):
            return False
        return job.progress_time() < progress_cutoff

    def _validate_result_path_for_deletion(self, job: Job) -> Optional[str]:
        if not job.result_path or not isinstance(job.result_path, str):
            return None

        result_roots = {
            JobTypes.TRAINING.value: adapters_root,
            JobTypes.ADAPTER_MERGE.value: merged_models_root,
        }
        root_factory = result_roots.get(job.job_type)
        if root_factory is not None:
            path = os.path.realpath(job.result_path)
            root = os.path.realpath(root_factory())
            if path == root or os.path.commonpath([root, path]) != root:
                raise ValueError(f"Refusing to delete result outside {root}: {path}")
            return job.result_path

        if os.path.islink(job.result_path) or os.path.isfile(job.result_path):
            return job.result_path
        return None

    @staticmethod
    def _delete_result_path(result_path: Optional[str]) -> None:
        if result_path is None:
            return
        if os.path.islink(result_path) or os.path.isfile(result_path):
            os.remove(result_path)
        elif os.path.isdir(result_path):
            shutil.rmtree(result_path)

    def _delete_jobs_and_results(self, jobs: list[Job]) -> None:
        result_paths = [self._validate_result_path_for_deletion(job) for job in jobs]
        if self.db:
            with self.db.job_deletion_transaction([job.id for job in jobs]):
                for result_path in result_paths:
                    self._delete_result_path(result_path)
        else:
            for result_path in result_paths:
                self._delete_result_path(result_path)

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

            for job in restored_jobs.values():
                if job.job_type != JobTypes.ADAPTER_MERGE.value:
                    continue
                source_job_id = job.request_parameters.get("source_job_id")
                parent_job = restored_jobs.get(source_job_id)
                if (
                    parent_job is not None
                    and parent_job.job_type == JobTypes.TRAINING.value
                    and parent_job.org_id == job.org_id
                ):
                    parent_job.adapter_merge_job_ids.add(job.id)

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
