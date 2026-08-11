"""Tests for MiniMax mock task storage and lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from minimax_mock.fixture_resolver import FixtureCatalog
from minimax_mock.schemas import VideoGenerationRequest
from minimax_mock.task_store import (
    TaskActionNotAllowedError,
    TaskDeleteAction,
    TaskStatus,
    TaskStore,
)


@dataclass
class _FakeClock:
    wall: float = 1_700_000_000.0
    monotonic: float = 100.0

    def wall_time(self) -> float:
        return self.wall

    def monotonic_time(self) -> float:
        return self.monotonic

    def advance(self, seconds: float) -> None:
        self.wall += seconds
        self.monotonic += seconds


def _request() -> VideoGenerationRequest:
    return VideoGenerationRequest.model_validate(
        {
            "model": "MiniMax-H3",
            "content": [{"type": "text", "text": "A city at sunrise."}],
            "resolution": "2K",
            "duration": 5,
            "ratio": "16:9",
        }
    )


def _store(
    clock: _FakeClock,
    *,
    retention_seconds: int = 7 * 24 * 60 * 60,
    ids=("100000000000001",),
) -> TaskStore:
    task_ids = iter(ids)
    return TaskStore(
        retention_seconds=retention_seconds,
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        task_id_factory=lambda: next(task_ids),
    )


def test_successful_task_transitions_from_queued_to_running_to_succeeded():
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(request)

    created = store.create(request, fixture)
    assert created.status is TaskStatus.QUEUED

    clock.advance(fixture.manifest.queued_for_ms / 1000)
    assert store.get(created.id).status is TaskStatus.RUNNING

    clock.advance(fixture.manifest.running_for_ms / 1000)
    completed = store.get(created.id)
    assert completed.status is TaskStatus.SUCCEEDED
    assert completed.fixture.manifest.name == "t2v-success"


def test_failed_fixture_transitions_to_failed():
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(request, scenario_name="generation-failed")

    created = store.create(request, fixture)
    clock.advance(
        (fixture.manifest.queued_for_ms + fixture.manifest.running_for_ms) / 1000
    )

    failed = store.get(created.id)
    assert failed.status is TaskStatus.FAILED
    assert failed.fixture.manifest.error.code == "1026"


def test_store_keeps_an_independent_request_snapshot():
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(request)

    created = store.create(request, fixture)
    request.content[0].text = "Changed after submission."
    created.request.content[0].text = "Changed returned snapshot."

    stored = store.get(created.id)
    assert stored.request.content[0].text == "A city at sunrise."


def test_store_retries_task_id_collisions():
    clock = _FakeClock()
    store = _store(
        clock,
        ids=("100000000000001", "100000000000001", "100000000000002"),
    )
    request = _request()
    fixture = FixtureCatalog().resolve(request)

    first = store.create(request, fixture)
    second = store.create(request, fixture)

    assert first.id == "100000000000001"
    assert second.id == "100000000000002"
    assert len(store) == 2


def test_list_tasks_returns_newest_first_and_refreshes_statuses():
    clock = _FakeClock()
    store = _store(
        clock,
        ids=("100000000000001", "100000000000002"),
    )
    request = _request()
    fixture = FixtureCatalog().resolve(request)

    first = store.create(request, fixture)
    clock.advance(fixture.manifest.running_for_ms / 1000)
    second = store.create(request, fixture)
    clock.advance(fixture.manifest.queued_for_ms / 1000)

    tasks = store.list_tasks()

    assert [task.id for task in tasks] == [second.id, first.id]
    assert tasks[0].status is TaskStatus.RUNNING
    assert tasks[1].status is TaskStatus.SUCCEEDED


def test_store_removes_tasks_after_retention_window():
    clock = _FakeClock()
    store = _store(clock, retention_seconds=10)
    request = _request()
    fixture = FixtureCatalog().resolve(request)
    created = store.create(request, fixture)

    clock.advance(10)
    assert store.get(created.id) is not None

    clock.advance(0.001)
    assert store.get(created.id) is None
    assert len(store) == 0


def test_store_returns_none_for_unknown_task():
    clock = _FakeClock()
    store = _store(clock)

    assert store.get("does-not-exist") is None


def test_store_cancels_queued_task_and_keeps_it_cancelled():
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(request)
    created = store.create(request, fixture)

    result = store.cancel_or_delete(created.id)

    assert result.action is TaskDeleteAction.CANCELLED
    assert store.get(created.id).status is TaskStatus.CANCELLED

    clock.advance(60)
    assert store.get(created.id).status is TaskStatus.CANCELLED
    with pytest.raises(TaskActionNotAllowedError) as exc_info:
        store.cancel_or_delete(created.id)
    assert exc_info.value.status is TaskStatus.CANCELLED


def test_store_rejects_cancellation_while_task_is_running():
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(request)
    created = store.create(request, fixture)
    clock.advance(fixture.manifest.queued_for_ms / 1000)

    with pytest.raises(TaskActionNotAllowedError) as exc_info:
        store.cancel_or_delete(created.id)

    assert exc_info.value.status is TaskStatus.RUNNING
    assert store.get(created.id).status is TaskStatus.RUNNING


@pytest.mark.parametrize("scenario_name", [None, "generation-failed"])
def test_store_deletes_succeeded_or_failed_task(scenario_name):
    clock = _FakeClock()
    store = _store(clock)
    request = _request()
    fixture = FixtureCatalog().resolve(
        request,
        **({"scenario_name": scenario_name} if scenario_name else {}),
    )
    created = store.create(request, fixture)
    clock.advance(
        (fixture.manifest.queued_for_ms + fixture.manifest.running_for_ms) / 1000
    )

    result = store.cancel_or_delete(created.id)

    assert result.action is TaskDeleteAction.DELETED
    assert store.get(created.id) is None
    assert len(store) == 0


def test_store_cancel_or_delete_returns_none_for_unknown_task():
    clock = _FakeClock()
    store = _store(clock)

    assert store.cancel_or_delete("does-not-exist") is None
