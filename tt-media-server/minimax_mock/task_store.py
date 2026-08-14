# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Thread-safe in-memory task storage for the MiniMax mock."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import RLock

from minimax_mock.fixture_resolver import ResolvedFixture
from minimax_mock.schemas import VideoGenerationRequest

TASK_RETENTION_SECONDS = 7 * 24 * 60 * 60


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskDeleteAction(str, Enum):
    CANCELLED = "cancelled"
    DELETED = "deleted"


@dataclass(frozen=True)
class TaskDeleteResult:
    task_id: str
    action: TaskDeleteAction


class TaskActionNotAllowedError(RuntimeError):
    def __init__(self, task_id: str, status: TaskStatus) -> None:
        super().__init__(
            f"task {task_id} cannot be cancelled or deleted while {status.value}"
        )
        self.task_id = task_id
        self.status = status


@dataclass(frozen=True)
class TaskRecord:
    id: str
    request: VideoGenerationRequest
    fixture: ResolvedFixture
    status: TaskStatus
    created_at: int
    updated_at: int


@dataclass
class _StoredTask:
    id: str
    request: VideoGenerationRequest
    fixture: ResolvedFixture
    status: TaskStatus
    created_at: int
    updated_at: int
    created_wall_time: float
    created_monotonic: float


class TaskStore:
    """Store mock tasks and derive their lifecycle from fixture timings.

    State transitions are calculated lazily when a task is read. This preserves
    asynchronous API behavior without creating one background coroutine per
    fixture-backed task.
    """

    def __init__(
        self,
        *,
        retention_seconds: int = TASK_RETENTION_SECONDS,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        task_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")

        self._retention_seconds = retention_seconds
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._task_id_factory = task_id_factory or _new_task_id
        self._tasks: dict[str, _StoredTask] = {}
        self._lock = RLock()

    def create(
        self,
        request: VideoGenerationRequest,
        fixture: ResolvedFixture,
    ) -> TaskRecord:
        wall_now = self._wall_clock()
        monotonic_now = self._monotonic_clock()

        with self._lock:
            self._remove_expired(wall_now)
            task_id = self._unique_task_id()
            created_at = int(wall_now)
            task = _StoredTask(
                id=task_id,
                request=request.model_copy(deep=True),
                fixture=fixture,
                status=TaskStatus.QUEUED,
                created_at=created_at,
                updated_at=created_at,
                created_wall_time=wall_now,
                created_monotonic=monotonic_now,
            )
            self._tasks[task_id] = task
            self._refresh_status(task, wall_now, monotonic_now)
            return self._snapshot(task)

    def get(self, task_id: str) -> TaskRecord | None:
        wall_now = self._wall_clock()
        monotonic_now = self._monotonic_clock()

        with self._lock:
            self._remove_expired(wall_now)
            task = self._tasks.get(task_id)
            if task is None:
                return None
            self._refresh_status(task, wall_now, monotonic_now)
            return self._snapshot(task)

    def list_tasks(self) -> list[TaskRecord]:
        wall_now = self._wall_clock()
        monotonic_now = self._monotonic_clock()

        with self._lock:
            self._remove_expired(wall_now)
            tasks = list(reversed(self._tasks.values()))
            for task in tasks:
                self._refresh_status(task, wall_now, monotonic_now)
            return [self._snapshot(task) for task in tasks]

    def cancel_or_delete(self, task_id: str) -> TaskDeleteResult | None:
        wall_now = self._wall_clock()
        monotonic_now = self._monotonic_clock()

        with self._lock:
            self._remove_expired(wall_now)
            task = self._tasks.get(task_id)
            if task is None:
                return None
            self._refresh_status(task, wall_now, monotonic_now)

            if task.status is TaskStatus.QUEUED:
                task.status = TaskStatus.CANCELLED
                task.updated_at = int(wall_now)
                return TaskDeleteResult(
                    task_id=task.id,
                    action=TaskDeleteAction.CANCELLED,
                )

            if task.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
                del self._tasks[task.id]
                return TaskDeleteResult(
                    task_id=task.id,
                    action=TaskDeleteAction.DELETED,
                )

            raise TaskActionNotAllowedError(task.id, task.status)

    def __len__(self) -> int:
        with self._lock:
            self._remove_expired(self._wall_clock())
            return len(self._tasks)

    def _unique_task_id(self) -> str:
        for _ in range(100):
            task_id = self._task_id_factory()
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("task_id_factory must return a non-empty string")
            if task_id not in self._tasks:
                return task_id
        raise RuntimeError("could not allocate a unique task_id")

    def _remove_expired(self, wall_now: float) -> None:
        cutoff = wall_now - self._retention_seconds
        expired_ids = [
            task_id
            for task_id, task in self._tasks.items()
            if task.created_wall_time < cutoff
        ]
        for task_id in expired_ids:
            del self._tasks[task_id]

    @staticmethod
    def _snapshot(task: _StoredTask) -> TaskRecord:
        return TaskRecord(
            id=task.id,
            request=task.request.model_copy(deep=True),
            fixture=task.fixture,
            status=task.status,
            created_at=task.created_at,
            updated_at=task.updated_at,
        )

    @staticmethod
    def _desired_status(task: _StoredTask, monotonic_now: float) -> TaskStatus:
        if task.status is TaskStatus.CANCELLED:
            return TaskStatus.CANCELLED

        elapsed_ms = max(0.0, monotonic_now - task.created_monotonic) * 1000
        manifest = task.fixture.manifest

        if elapsed_ms < manifest.queued_for_ms:
            return TaskStatus.QUEUED
        if elapsed_ms < manifest.queued_for_ms + manifest.running_for_ms:
            return TaskStatus.RUNNING
        return TaskStatus(manifest.terminal_status)

    def _refresh_status(
        self,
        task: _StoredTask,
        wall_now: float,
        monotonic_now: float,
    ) -> None:
        desired_status = self._desired_status(task, monotonic_now)
        if desired_status is task.status:
            return
        task.status = desired_status
        task.updated_at = int(wall_now)


def _new_task_id() -> str:
    return str(100_000_000_000_000 + secrets.randbelow(900_000_000_000_000))
