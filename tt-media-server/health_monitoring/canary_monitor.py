# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Model-agnostic canary monitor.

Named after the coal-mine canary: when no real request is in flight the monitor
submits a synthetic probe through the normal scheduler request path and asks the
runner's ``health_check()`` hook whether the model can still serve.

"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from config.constants import (
    CANARY_DEEP_TASK_ID,
    CANARY_TASK_ID,
    CANARY_TASK_IDS,
    CanaryProbeRequest,
)
from config.settings import get_settings
from telemetry.telemetry_client import (
    canary_failures_total,
    canary_last_success_timestamp,
    canary_probe_latency_seconds,
    canary_state,
)
from utils.logger import TTLogger

# Probe tiers. Misses are tracked per tier so a shallow (host-liveness) success
# can never clear a deep (device-liveness) miss streak — they answer different
# questions, and conflating them masks a live-host/dead-device failure.
_SHALLOW = "shallow"
_DEEP = "deep"


class CanaryState(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    PROBING = "probing"
    SUSPECT = "suspect"
    DEAD = "dead"


class CanaryMonitor:
    def __init__(self, scheduler, settings=None):
        self.scheduler = scheduler
        self.settings = settings or get_settings()
        self.logger = TTLogger()
        self._model_type = self.settings.model_runner
        self._state = CanaryState.STARTING
        # Per-tier consecutive miss counters; each reset only by a success of
        # its own tier (or by real traffic, which proves both at once).
        self._misses = {_SHALLOW: 0, _DEEP: 0}
        self._last_activity = time.monotonic()
        self._started_at: float | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        # Counts probes actually issued; drives the deep-probe cadence.
        self._probe_count = 0

    @property
    def current_state(self) -> CanaryState:
        return self._state

    def is_alive(self) -> bool:
        """Report liveness. Whether this gates /health is decided by
        check_is_model_ready via settings.canary_gate_readiness."""
        return self._state != CanaryState.DEAD

    def start(self) -> None:
        if not self.settings.canary_enabled:
            self.logger.info("Canary monitor disabled; monitor not started")
            return
        if self._running:
            return
        self._running = True
        self._started_at = time.monotonic()
        self._last_activity = time.monotonic()
        self._set_state(CanaryState.STARTING)
        self._task = asyncio.create_task(self._run())
        self.logger.info(
            f"Canary monitor started "
            f"(gate_readiness={self.settings.canary_gate_readiness})"
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
            if key not in CANARY_TASK_IDS
        )

    def _in_startup_grace(self) -> bool:
        if self._started_at is None:
            return True
        elapsed = time.monotonic() - self._started_at
        return elapsed < self.settings.canary_startup_grace_seconds

    async def _run(self) -> None:
        tick = max(0.05, self.settings.canary_tick_seconds)
        try:
            while self._running and self.scheduler.is_ready:
                await asyncio.sleep(tick)
                await self._tick()
        except asyncio.CancelledError:
            pass
        finally:
            self.logger.info("Canary monitor stopped")

    async def _tick(self) -> None:
        if self._in_startup_grace():
            return
        # Busy = alive: any in-flight real request resets the timer and clears
        # a pending suspicion, so we never probe (or stay suspicious) under load.
        if self._in_flight_count() > 0:
            self._on_real_activity()
            return
        idle_for = time.monotonic() - self._last_activity
        if idle_for < self.settings.canary_wait_seconds:
            return
        await self._probe_once()

    def _on_real_activity(self) -> None:
        # A real request runs the full forward across all ranks, so it proves
        # both host and device liveness: clear every tier.
        self._last_activity = time.monotonic()
        self._misses = {_SHALLOW: 0, _DEEP: 0}
        if self._state != CanaryState.HEALTHY:
            self._set_state(CanaryState.HEALTHY)

    def _should_deep_probe(self) -> bool:
        """Fire a deep (device-depth) probe on every Nth issued probe.

        Most probes stay cheap host collectives for fast hang detection; the
        deep one periodically certifies the device can still compute.
        """
        if not self.settings.canary_deep_probe_enabled:
            return False
        every_n = max(1, self.settings.canary_deep_every_n)
        return self._probe_count % every_n == 0

    async def _probe_once(self) -> None:
        self._probe_count += 1
        deep = self._should_deep_probe()
        result, latency = await self._submit_probe(deep)
        # A real request that arrived during the probe is proof of liveness:
        # abort without counting a miss.
        if self._in_flight_count() > 0:
            self._on_real_activity()
            return
        if result is None:
            return  # skipped (queue full / probe already in flight): not a miss
        if result:
            self._on_probe_success(latency, deep)
        else:
            self._on_probe_miss(deep)

    @staticmethod
    def _depth_label(deep: bool) -> str:
        return _DEEP if deep else _SHALLOW

    def _severity_state(self) -> CanaryState:
        """Derive state from the worst tier: a dead tier dominates a healthy
        one, so a passing shallow probe can't whitewash a failing deep streak.
        """
        worst = max(self._misses.values())
        if worst >= self.settings.canary_dead_misses:
            return CanaryState.DEAD
        if worst > 0:
            return CanaryState.SUSPECT
        return CanaryState.HEALTHY

    def _on_probe_success(self, latency: float | None, deep: bool) -> None:
        label = self._depth_label(deep)
        was_dead = self._state == CanaryState.DEAD
        self._misses[label] = 0
        self._last_activity = time.monotonic()
        self._set_state(self._severity_state())
        if was_dead and self._state != CanaryState.DEAD:
            self.logger.info(
                f"Canary monitor: {label} probe recovered, "
                f"state now {self._state.value}"
            )
        canary_last_success_timestamp.labels(model_type=self._model_type).set(
            time.time()
        )
        if latency is not None:
            canary_probe_latency_seconds.labels(
                model_type=self._model_type, depth=label
            ).observe(latency)

    def _on_probe_miss(self, deep: bool) -> None:
        label = self._depth_label(deep)
        self._misses[label] += 1
        canary_failures_total.labels(model_type=self._model_type, depth=label).inc()
        dead_misses = self.settings.canary_dead_misses
        count = self._misses[label]
        was_dead = self._state == CanaryState.DEAD
        self._set_state(self._severity_state())
        if self._state == CanaryState.DEAD and not was_dead:
            self.logger.error(
                f"Canary monitor: model declared DEAD after {count} "
                f"consecutive {label} misses"
            )
        elif self._state == CanaryState.SUSPECT:
            self.logger.warning(f"Canary {label} miss {count}/{dead_misses}")

    async def _submit_probe(self, deep: bool) -> tuple[bool | None, float | None]:
        """Round-trip one canary probe through the scheduler path.

        Returns ``(True, latency)`` on success, ``(False, None)`` on a miss
        (timeout / error / falsy), and ``(None, None)`` when the probe was
        skipped (queue full or one already in flight) and must not count.
        """
        task_id = CANARY_DEEP_TASK_ID if deep else CANARY_TASK_ID
        if task_id in self.scheduler.result_queues:
            return None, None
        # Register the result queue BEFORE enqueueing so a fast worker response
        # can never race ahead of routing and get dropped by result_listener.
        result_queue: asyncio.Queue = asyncio.Queue()
        self.scheduler.result_queues[task_id] = result_queue
        timeout_s = (
            self.settings.canary_deep_probe_timeout_seconds
            if deep
            else self.settings.canary_probe_timeout_seconds
        )
        try:
            if not self._enqueue_probe(task_id, deep):
                return None, None
            self._set_state(CanaryState.PROBING)
            start = time.monotonic()
            result = await asyncio.wait_for(result_queue.get(), timeout=timeout_s)
            if isinstance(result, Exception):
                self.logger.warning(f"Canary probe returned error: {result}")
                return False, None
            return (result is True), (time.monotonic() - start)
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Canary {self._depth_label(deep)} probe timed out "
                f"after {timeout_s:.1f}s"
            )
            return False, None
        finally:
            self.scheduler.result_queues.pop(task_id, None)

    def _enqueue_probe(self, task_id: str, deep: bool) -> bool:
        """Register the canary probe on the task queue.

        We deliberately bypass scheduler.process_request(): it calls
        check_is_model_ready(), which the canary monitor feeds, so routing
        through it would be circular. A full queue means the system is busy, not
        dead, so we skip rather than count a miss.
        """
        try:
            self.scheduler.task_queue.put(
                CanaryProbeRequest(_task_id=task_id, deep=deep), timeout=1.0
            )
            return True
        except Exception:
            self.logger.warning("Canary monitor: task_queue full; skipping probe")
            return False

    def _set_state(self, state: CanaryState) -> None:
        self._state = state
        for candidate in CanaryState:
            canary_state.labels(model_type=self._model_type, state=candidate.value).set(
                1.0 if candidate == state else 0.0
            )
