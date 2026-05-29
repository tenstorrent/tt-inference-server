# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for the model-agnostic gas monitor.

The monitor's async methods are driven directly (no real workers, no sleeps)
via a fake scheduler whose task_queue synchronously delivers a configurable
probe outcome into the registered result queue, mirroring what
result_listener does in production.
"""

import asyncio
from types import SimpleNamespace

import pytest

from config.constants import GAS_PROBE_TASK_ID, GasProbeRequest
from health_monitoring.gas_monitor import GasMonitor, GasMonitorState
from tt_model_runners.base_device_runner import BaseDeviceRunner

PAST = -10_000.0


def _settings(**overrides):
    base = dict(
        model_runner="sp_runner",
        gas_monitor_enabled=True,
        gas_monitor_gate_readiness=False,
        gas_monitor_wait_seconds=5.0,
        gas_monitor_probe_timeout_seconds=0.05,
        gas_monitor_tick_seconds=0.01,
        gas_monitor_dead_misses=3,
        gas_monitor_startup_grace_seconds=30.0,
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
      "arrive_real" -> a real request lands during the probe; gas probe stalls
    """

    def __init__(self, scheduler, behavior):
        self.scheduler = scheduler
        self.behavior = behavior
        self.items = []

    def put(self, request, timeout=None):
        self.items.append(request)
        if not isinstance(request, GasProbeRequest):
            return
        if self.behavior == "full":
            raise RuntimeError("task queue full")
        result_queue = self.scheduler.result_queues.get(GAS_PROBE_TASK_ID)
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
    monitor = GasMonitor(scheduler, settings=_settings(**settings_overrides))
    # Bypass startup grace and the idle dwell unless a test wants them.
    import time

    monitor._started_at = time.monotonic() + (1000.0 if in_grace else PAST)
    monitor._last_activity = time.monotonic() + PAST
    return monitor


def test_default_health_check_returns_true():
    # The cheap default must never falsely fail; it ignores self.
    assert BaseDeviceRunner.health_check(None) is True


def test_gas_probe_request_defaults():
    req = GasProbeRequest()
    assert req._task_id == GAS_PROBE_TASK_ID
    assert req.stream is False


def test_idle_probe_marks_dead_after_misses():
    monitor = _make_monitor(behavior="timeout", gas_monitor_dead_misses=3)
    for _ in range(3):
        asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.DEAD
    assert monitor.is_alive() is False


def test_suspect_before_dead():
    monitor = _make_monitor(behavior="timeout", gas_monitor_dead_misses=3)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.SUSPECT
    assert monitor.is_alive() is True


def test_error_payload_counts_as_miss():
    monitor = _make_monitor(behavior="error", gas_monitor_dead_misses=1)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.DEAD


def test_probe_success_keeps_healthy():
    monitor = _make_monitor(behavior="success")
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.HEALTHY
    assert monitor.is_alive() is True


def test_state_recovers_on_success():
    monitor = _make_monitor(behavior="success")
    monitor._state = GasMonitorState.DEAD
    monitor._consecutive_misses = 5
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.HEALTHY
    assert monitor._consecutive_misses == 0


def test_activity_resets_gas_monitor_timer():
    monitor = _make_monitor(behavior="timeout")
    monitor._state = GasMonitorState.SUSPECT
    monitor._consecutive_misses = 2
    monitor.scheduler.result_queues["real-task"] = asyncio.Queue()
    asyncio.run(monitor._tick())
    assert monitor.current_state == GasMonitorState.HEALTHY
    assert monitor._consecutive_misses == 0
    # Busy means no probe was ever submitted.
    assert monitor.scheduler.task_queue.qsize() == 0


def test_busy_does_not_false_positive_when_idle_elapsed():
    monitor = _make_monitor(behavior="timeout")
    monitor.scheduler.result_queues["real-task"] = asyncio.Queue()
    asyncio.run(monitor._tick())
    assert monitor.current_state == GasMonitorState.HEALTHY
    assert monitor.scheduler.task_queue.qsize() == 0


def test_request_arrives_during_probe_window_recovers():
    # Real request lands mid-probe and the gas probe stalls/times out: in-flight
    # real work is liveness, so we abort to HEALTHY and do NOT count a miss.
    monitor = _make_monitor(behavior="arrive_real", gas_monitor_dead_misses=2)
    asyncio.run(monitor._probe_once())
    assert monitor.current_state == GasMonitorState.HEALTHY
    assert monitor._consecutive_misses == 0


def test_no_double_gas_probe_inflight():
    monitor = _make_monitor(behavior="success")
    sentinel = asyncio.Queue()
    monitor.scheduler.result_queues[GAS_PROBE_TASK_ID] = sentinel
    result = asyncio.run(monitor._submit_probe())
    assert result == (None, None)
    # The pre-existing in-flight probe queue must not be clobbered or dropped.
    assert monitor.scheduler.result_queues[GAS_PROBE_TASK_ID] is sentinel


def test_queue_full_is_skip_not_miss():
    monitor = _make_monitor(behavior="full")
    asyncio.run(monitor._probe_once())
    assert monitor._consecutive_misses == 0
    assert monitor.current_state != GasMonitorState.SUSPECT


def test_startup_grace_blocks_probe():
    monitor = _make_monitor(behavior="success", in_grace=True)
    asyncio.run(monitor._tick())
    assert monitor.scheduler.task_queue.qsize() == 0
    assert monitor.current_state == GasMonitorState.STARTING


@pytest.mark.parametrize("gate,expect_raise", [(True, True), (False, False)])
def test_check_is_model_ready_gating(monkeypatch, gate, expect_raise):
    from fastapi import HTTPException

    import model_services.base_service as base_service_module
    from model_services.base_service import BaseService

    monkeypatch.setattr(
        base_service_module.settings, "gas_monitor_gate_readiness", gate
    )

    dead_monitor = SimpleNamespace(
        is_alive=lambda: False, current_state=GasMonitorState.DEAD
    )
    scheduler = _FakeScheduler()
    scheduler.gas_monitor = dead_monitor
    fake_self = SimpleNamespace(scheduler=scheduler)

    if expect_raise:
        with pytest.raises(HTTPException) as exc:
            BaseService.check_is_model_ready(fake_self)
        assert exc.value.status_code == 503
    else:
        status = BaseService.check_is_model_ready(fake_self)
        assert status["gas_monitor_alive"] is False
        assert status["gas_monitor_state"] == GasMonitorState.DEAD.value
