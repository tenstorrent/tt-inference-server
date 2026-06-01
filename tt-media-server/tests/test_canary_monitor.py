# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for the model-agnostic canary monitor.

The monitor's async methods are driven directly (no real workers, no sleeps)
via a fake scheduler whose task_queue synchronously delivers a configurable
probe outcome into the registered result queue, mirroring what
result_listener does in production.
"""

import asyncio
from types import SimpleNamespace

import pytest

from config.constants import (
    CANARY_DEEP_TASK_ID,
    CANARY_TASK_ID,
    CanaryProbeRequest,
)
from health_monitoring.canary_monitor import CanaryMonitor, CanaryState
from tt_model_runners.base_device_runner import BaseDeviceRunner

PAST = -10_000.0


def _settings(**overrides):
    base = dict(
        model_runner="sp_runner",
        canary_enabled=True,
        canary_gate_readiness=False,
        canary_wait_seconds=5.0,
        canary_probe_timeout_seconds=0.05,
        canary_tick_seconds=0.01,
        canary_dead_misses=3,
        canary_startup_grace_seconds=30.0,
        canary_deep_probe_enabled=False,
        canary_deep_every_n=12,
        canary_deep_probe_timeout_seconds=0.05,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeTaskQueue:
    """Synchronously delivers a probe outcome to the registered result queue.

    behavior:
      "success"     -> True
      "false"       -> False
      "error"       -> Exception payload (mirrors error_listener routing)
      "timeout"     -> no delivery (probe must time out)
      "full"        -> raise on put (queue full)
      "arrive_real" -> a real request lands during the probe; canary stalls
    """

    def __init__(self, scheduler, behavior):
        self.scheduler = scheduler
        self.behavior = behavior
        self.items = []

    def put(self, request, timeout=None):
        self.items.append(request)
        if not isinstance(request, CanaryProbeRequest):
            return
        if self.behavior == "full":
            raise RuntimeError("task queue full")
        # Route to whichever id the probe registered (shallow or deep), mirroring
        # how the device worker echoes the probe's own _task_id.
        result_queue = self.scheduler.result_queues.get(request._task_id)
        if self.behavior == "success":
            result_queue.put_nowait(True)
        elif self.behavior == "false":
            result_queue.put_nowait(False)
        elif self.behavior == "error":
            result_queue.put_nowait(Exception("boom"))
        elif self.behavior == "arrive_real":
            self.scheduler.result_queues["real-task-123"] = asyncio.Queue()
        # "timeout": deliver nothing

    def qsize(self):
        return len(self.items)


class _FakeScheduler:
    def __init__(self, behavior="timeout"):
        self.result_queues = {}
        self.is_ready = True
        self.task_queue = _FakeTaskQueue(self, behavior)

    def check_is_model_ready(self):
        return True

    def get_worker_info(self):
        return {}


def _make_monitor(behavior="timeout", in_grace=False, **settings_overrides):
    scheduler = _FakeScheduler(behavior)
    monitor = CanaryMonitor(scheduler, settings=_settings(**settings_overrides))
    # Bypass startup grace and the idle dwell unless a test wants them.
    import time

    monitor._started_at = time.monotonic() + (1000.0 if in_grace else PAST)
    monitor._last_activity = time.monotonic() + PAST
    return monitor


def test_default_health_check_returns_true():
    # The cheap default must never falsely fail; it ignores self.
    assert BaseDeviceRunner.health_check(None) is True


def test_canary_probe_request_defaults():
    req = CanaryProbeRequest()
    assert req._task_id == CANARY_TASK_ID
    assert req.stream is False
    assert req.deep is False


def test_idle_probe_marks_dead_after_misses():
    monitor = _make_monitor(behavior="timeout", canary_dead_misses=3)
    for _ in range(3):
        asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.DEAD
    assert monitor.is_alive() is False


def test_suspect_before_dead():
    monitor = _make_monitor(behavior="timeout", canary_dead_misses=3)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.SUSPECT
    assert monitor.is_alive() is True


def test_error_payload_counts_as_miss():
    monitor = _make_monitor(behavior="error", canary_dead_misses=1)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.DEAD


def test_probe_success_keeps_healthy():
    monitor = _make_monitor(behavior="success")
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor.is_alive() is True


def test_state_recovers_on_success():
    monitor = _make_monitor(behavior="success")
    monitor._state = CanaryState.DEAD
    monitor._misses["shallow"] = 5
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor._misses["shallow"] == 0


def test_activity_resets_canary_timer():
    monitor = _make_monitor(behavior="timeout")
    monitor._state = CanaryState.SUSPECT
    monitor._misses["deep"] = 2
    monitor.scheduler.result_queues["real-task"] = asyncio.Queue()
    asyncio.run(monitor._tick())
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor._misses == {"shallow": 0, "deep": 0}
    # Busy means no probe was ever submitted.
    assert monitor.scheduler.task_queue.qsize() == 0


def test_busy_does_not_false_positive_when_idle_elapsed():
    monitor = _make_monitor(behavior="timeout")
    monitor.scheduler.result_queues["real-task"] = asyncio.Queue()
    asyncio.run(monitor._tick())
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor.scheduler.task_queue.qsize() == 0


def test_request_arrives_during_probe_window_recovers():
    # Real request lands mid-probe and the canary stalls/times out: in-flight
    # real work is liveness, so we abort to HEALTHY and do NOT count a miss.
    monitor = _make_monitor(behavior="arrive_real", canary_dead_misses=2)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor._misses == {"shallow": 0, "deep": 0}


def test_no_double_canary_inflight():
    monitor = _make_monitor(behavior="success")
    sentinel = asyncio.Queue()
    monitor.scheduler.result_queues[CANARY_TASK_ID] = sentinel
    result = asyncio.run(monitor._submit_probe(deep=False))
    assert result == (None, None)
    # The pre-existing in-flight probe queue must not be clobbered or dropped.
    assert monitor.scheduler.result_queues[CANARY_TASK_ID] is sentinel


def test_deep_probe_disabled_by_default():
    # With the deep probe off, every probe is shallow regardless of count.
    monitor = _make_monitor(behavior="success", canary_deep_probe_enabled=False)
    for _ in range(5):
        asyncio.run(monitor._probe_once())
    ids = [req._task_id for req in monitor.scheduler.task_queue.items]
    assert all(t == CANARY_TASK_ID for t in ids)


def test_deep_probe_cadence_uses_deep_task_id():
    # every_n=3 → probes 3 and 6 are deep, the rest shallow.
    monitor = _make_monitor(
        behavior="success", canary_deep_probe_enabled=True, canary_deep_every_n=3
    )
    for _ in range(6):
        asyncio.run(monitor._probe_once())
    ids = [req._task_id for req in monitor.scheduler.task_queue.items]
    deep_positions = [i + 1 for i, t in enumerate(ids) if t == CANARY_DEEP_TASK_ID]
    assert deep_positions == [3, 6]
    assert monitor.current_state == CanaryState.HEALTHY


def test_deep_probe_carries_deep_flag():
    monitor = _make_monitor(
        behavior="success", canary_deep_probe_enabled=True, canary_deep_every_n=1
    )
    asyncio.run(monitor._probe_once())
    probe = monitor.scheduler.task_queue.items[-1]
    assert probe._task_id == CANARY_DEEP_TASK_ID
    assert probe.deep is True


def test_shallow_success_does_not_clear_deep_miss_streak():
    # The masking gap: a live host (shallow ok) must not whitewash a dying
    # device (deep failing). Interleaving a shallow success between deep misses
    # must still reach DEAD.
    monitor = _make_monitor(canary_dead_misses=2)
    monitor._on_probe_miss(deep=True)
    assert monitor.current_state == CanaryState.SUSPECT
    monitor._on_probe_success(latency=0.01, deep=False)  # host still alive
    assert monitor.current_state == CanaryState.SUSPECT  # deep streak survives
    assert monitor._misses == {"shallow": 0, "deep": 1}
    monitor._on_probe_miss(deep=True)
    assert monitor.current_state == CanaryState.DEAD
    assert monitor.is_alive() is False


def test_deep_recovers_on_deep_success():
    monitor = _make_monitor(canary_dead_misses=2)
    monitor._on_probe_miss(deep=True)
    monitor._on_probe_miss(deep=True)
    assert monitor.current_state == CanaryState.DEAD
    monitor._on_probe_success(latency=0.01, deep=True)
    assert monitor.current_state == CanaryState.HEALTHY
    assert monitor._misses["deep"] == 0


def test_tiers_are_independent():
    # A shallow miss and a deep miss accrue separately.
    monitor = _make_monitor(canary_dead_misses=3)
    monitor._on_probe_miss(deep=False)
    monitor._on_probe_miss(deep=True)
    assert monitor._misses == {"shallow": 1, "deep": 1}
    assert monitor.current_state == CanaryState.SUSPECT


def test_deep_probe_timeout_is_independent(monkeypatch):
    # A deep probe that times out still counts as a miss, using the deep budget.
    monitor = _make_monitor(
        behavior="timeout",
        canary_deep_probe_enabled=True,
        canary_deep_every_n=1,
        canary_deep_probe_timeout_seconds=0.02,
        canary_dead_misses=1,
    )
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == CanaryState.DEAD
    assert monitor.scheduler.task_queue.items[-1]._task_id == CANARY_DEEP_TASK_ID


def test_queue_full_is_skip_not_miss():
    monitor = _make_monitor(behavior="full")
    asyncio.run(monitor._probe_once())
    assert monitor._misses == {"shallow": 0, "deep": 0}
    assert monitor.current_state != CanaryState.SUSPECT


def test_startup_grace_blocks_probe():
    monitor = _make_monitor(behavior="success", in_grace=True)
    asyncio.run(monitor._tick())
    assert monitor.scheduler.task_queue.qsize() == 0
    assert monitor.current_state == CanaryState.STARTING


@pytest.mark.parametrize("gate,expect_raise", [(True, True), (False, False)])
def test_check_is_model_ready_gating(monkeypatch, gate, expect_raise):
    from fastapi import HTTPException

    import model_services.base_service as base_service_module
    from model_services.base_service import BaseService

    monkeypatch.setattr(base_service_module.settings, "canary_gate_readiness", gate)

    dead_monitor = SimpleNamespace(
        is_alive=lambda: False, current_state=CanaryState.DEAD
    )
    scheduler = _FakeScheduler()
    scheduler.canary_monitor = dead_monitor
    fake_self = SimpleNamespace(scheduler=scheduler)

    if expect_raise:
        with pytest.raises(HTTPException) as exc:
            BaseService.check_is_model_ready(fake_self)
        assert exc.value.status_code == 503
    else:
        status = BaseService.check_is_model_ready(fake_self)
        assert status["canary_alive"] is False
        assert status["canary_state"] == CanaryState.DEAD.value
