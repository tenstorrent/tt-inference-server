# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Model-agnostic gas monitor.

Named after the electronic gas monitor that replaced the coal-mine CANARY: when
no real request is in flight the monitor submits a synthetic probe through the
normal scheduler request path and asks the runner's ``health_check()`` hook
whether the model can still serve.

"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from config.constants import GAS_PROBE_TASK_ID, GasProbeRequest
from config.settings import get_settings
from telemetry.telemetry_client import (
    gas_monitor_failures_total,
    gas_monitor_last_success_timestamp,
    gas_monitor_probe_latency_seconds,
    gas_monitor_state,
)
from utils.logger import TTLogger


class GasMonitorState(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    PROBING = "probing"
    SUSPECT = "suspect"
    DEAD = "dead"


class GasMonitor:
    def __init__(self, scheduler, settings=None):
        self.scheduler = scheduler
        self.settings = settings or get_settings()
        self.logger = TTLogger()
        self._model_type = self.settings.model_runner
        self._state = GasMonitorState.STARTING
        self._consecutive_misses = 0
        self._last_activity = time.monotonic()
        self._started_at: float | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def current_state(self) -> GasMonitorState:
        return self._state

    def is_alive(self) -> bool:
        """Report liveness. Whether this gates /health is decided by
        check_is_model_ready via settings.gas_monitor_gate_readiness."""
        return self._state != GasMonitorState.DEAD

    def start(self) -> None:
        if not self.settings.gas_monitor_enabled:
            self.logger.info("Gas monitor disabled; monitor not started")
            return
        if self._running:
            return
        self._running = True
        self._started_at = time.monotonic()
        self._last_activity = time.monotonic()
        self._set_state(GasMonitorState.STARTING)
        self._task = asyncio.create_task(self._run())
        self.logger.info(
            f"Gas monitor started "
            f"(gate_readiness={self.settings.gas_monitor_gate_readiness})"
        )

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    def _in_flight_count(self) -> int:
        """Real (non-probe) requests currently awaiting a result."""
        return sum(
            1
            for key in list(self.scheduler.result_queues.keys())
            if key != GAS_PROBE_TASK_ID
        )

    def _in_startup_grace(self) -> bool:
        if self._started_at is None:
            return True
        elapsed = time.monotonic() - self._started_at
        return elapsed < self.settings.gas_monitor_startup_grace_seconds

    async def _run(self) -> None:
        tick = max(0.05, self.settings.gas_monitor_tick_seconds)
        try:
            while self._running and self.scheduler.is_ready:
                await asyncio.sleep(tick)
                await self._tick()
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("Gas monitor stopped")

    async def _tick(self) -> None:
        if self._in_startup_grace():
            return
        # Busy = alive: any in-flight real request resets the timer and clears
        # a pending suspicion, so we never probe (or stay suspicious) under load.
        if self._in_flight_count() > 0:
            self._on_real_activity()
            return
        idle_for = time.monotonic() - self._last_activity
        if idle_for < self.settings.gas_monitor_wait_seconds:
            return
        await self._probe_once()

    def _on_real_activity(self) -> None:
        self._last_activity = time.monotonic()
        self._consecutive_misses = 0
        if self._state != GasMonitorState.HEALTHY:
            self._set_state(GasMonitorState.HEALTHY)

    async def _probe_once(self) -> None:
        result, latency = await self._submit_probe()
        # A real request that arrived during the probe is proof of liveness:
        # abort without counting a miss.
        if self._in_flight_count() > 0:
            self._on_real_activity()
            return
        if result is None:
            return  # skipped (queue full / probe already in flight): not a miss
        if result:
            self._on_probe_success(latency)
        else:
            self._on_probe_miss()

    def _on_probe_success(self, latency: float | None) -> None:
        self._consecutive_misses = 0
        self._last_activity = time.monotonic()
        recovered = self._state == GasMonitorState.DEAD
        self._set_state(GasMonitorState.HEALTHY)
        if recovered:
            self.logger.info("Gas monitor: model recovered, back to HEALTHY")
        gas_monitor_last_success_timestamp.labels(model_type=self._model_type).set(
            time.time()
        )
        if latency is not None:
            gas_monitor_probe_latency_seconds.labels(
                model_type=self._model_type
            ).observe(latency)

    def _on_probe_miss(self) -> None:
        self._consecutive_misses += 1
        gas_monitor_failures_total.labels(model_type=self._model_type).inc()
        dead_misses = self.settings.gas_monitor_dead_misses
        if self._consecutive_misses >= dead_misses:
            if self._state != GasMonitorState.DEAD:
                self.logger.error(
                    f"Gas monitor: model declared DEAD after "
                    f"{self._consecutive_misses} consecutive misses"
                )
            self._set_state(GasMonitorState.DEAD)
        else:
            self._set_state(GasMonitorState.SUSPECT)
            self.logger.warning(
                f"Gas monitor miss {self._consecutive_misses}/{dead_misses}"
            )

    async def _submit_probe(self) -> tuple[bool | None, float | None]:
        """Round-trip one gas probe through the scheduler path.

        Returns ``(True, latency)`` on success, ``(False, None)`` on a miss
        (timeout / error / falsy), and ``(None, None)`` when the probe was
        skipped (queue full or one already in flight) and must not count.
        """
        if GAS_PROBE_TASK_ID in self.scheduler.result_queues:
            return None, None
        # Register the result queue BEFORE enqueueing so a fast worker response
        # can never race ahead of routing and get dropped by result_listener.
        result_queue: asyncio.Queue = asyncio.Queue()
        self.scheduler.result_queues[GAS_PROBE_TASK_ID] = result_queue
        timeout_s = self.settings.gas_monitor_probe_timeout_seconds
        try:
            if not self._enqueue_probe():
                return None, None
            self._set_state(GasMonitorState.PROBING)
            start = time.monotonic()
            result = await asyncio.wait_for(result_queue.get(), timeout=timeout_s)
            if isinstance(result, Exception):
                self.logger.warning(f"Gas probe returned error: {result}")
                return False, None
            return (result is True), (time.monotonic() - start)
        except asyncio.TimeoutError:
            self.logger.warning(f"Gas probe timed out after {timeout_s:.1f}s")
            return False, None
        finally:
            self.scheduler.result_queues.pop(GAS_PROBE_TASK_ID, None)

    def _enqueue_probe(self) -> bool:
        """Register the gas probe on the task queue.

        We deliberately bypass scheduler.process_request(): it calls
        check_is_model_ready(), which the gas monitor feeds, so routing through
        it would be circular. A full queue means the system is busy, not dead,
        so we skip rather than count a miss.
        """
        try:
            self.scheduler.task_queue.put(GasProbeRequest(), timeout=1.0)
            return True
        except Exception:
            self.logger.warning("Gas monitor: task_queue full; skipping probe")
            return False

    def _set_state(self, state: GasMonitorState) -> None:
        self._state = state
        for candidate in GasMonitorState:
            gas_monitor_state.labels(
                model_type=self._model_type, state=candidate.value
            ).set(1.0 if candidate == state else 0.0)
