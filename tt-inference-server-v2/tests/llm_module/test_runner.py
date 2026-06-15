# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``llm_module.runner.LLMPerformanceRunner`` orchestration.

Drives the runner with a fake driver + fake server controller so the
sweep flow (health gate, trace capture, per-point health check, parse)
is exercised without a real inference server or perf tool.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import requests

from llm_module import DriverResult, LLMPerformanceRunner, RunnerResult
from llm_module.config import DriverContext, LLMRunConfig, ServerConnection
from report_module.schema import Block


def _cfg(isl=128, osl=64, n=4) -> LLMRunConfig:
    return LLMRunConfig(isl=isl, osl=osl, max_concurrency=n, num_prompts=8)


_SERVER = ServerConnection(base_url="localhost", service_port=8000, model="m")
_CTX = DriverContext(output_dir=Path("/tmp/out"), device="n300")


class FakeDriver:
    name = "fake"

    def __init__(self, outcomes: List[DriverResult]):
        self._outcomes = list(outcomes)
        self.run_calls: List[LLMRunConfig] = []
        self.parsed_devices: List[str] = []

    def run(self, config, server, context) -> DriverResult:
        self.run_calls.append(config)
        return self._outcomes.pop(0)

    def parse(self, raw, *, device: str = "") -> Block:
        self.parsed_devices.append(device)
        return Block(kind="benchmarks", data=dict(raw))


class FakeController:
    def __init__(
        self,
        *,
        healthy: bool = True,
        health_status: int = 200,
        health_exc: Optional[Exception] = None,
        capture_exc: Optional[Exception] = None,
    ):
        self._healthy = healthy
        self._health_status = health_status
        self._health_exc = health_exc
        self._capture_exc = capture_exc
        self.wait_calls = 0
        self.capture_args: List = []

    def wait_for_healthy(self, timeout=None, interval=10) -> bool:
        self.wait_calls += 1
        return self._healthy

    def get_health(self):
        if self._health_exc is not None:
            raise self._health_exc
        return SimpleNamespace(status_code=self._health_status)

    def capture_traces(self, context_lens, timeout=None) -> None:
        if self._capture_exc is not None:
            raise self._capture_exc
        self.capture_args.append(list(context_lens))


def _ok(raw=None) -> DriverResult:
    return DriverResult(return_code=0, raw=raw or {"ttft": 1.0}, raw_path=None)


def _runner(driver, controller=None) -> LLMPerformanceRunner:
    return LLMPerformanceRunner(
        driver, controller, inter_run_sleep_s=0.0
    )


def test_zero_configs_returns_empty_without_touching_server():
    controller = FakeController()
    result = _runner(FakeDriver([]), controller).run([], _SERVER, _CTX)
    assert result.blocks == []
    assert controller.wait_calls == 0
    assert result.ok is False  # no return codes -> not ok


def test_happy_path_runs_every_point_and_parses_blocks():
    driver = FakeDriver([_ok(), _ok()])
    controller = FakeController()
    result = _runner(driver, controller).run([_cfg(), _cfg(osl=128)], _SERVER, _CTX)
    assert len(driver.run_calls) == 2
    assert len(result.blocks) == 2
    assert result.return_codes == [0, 0]
    assert result.ok is True
    # device flows through to parse.
    assert driver.parsed_devices == ["n300", "n300"]


def test_trace_capture_gets_unique_sorted_context_lengths():
    driver = FakeDriver([_ok(), _ok(), _ok()])
    controller = FakeController()
    configs = [_cfg(isl=128, osl=64), _cfg(isl=128, osl=64), _cfg(isl=256, osl=64)]
    _runner(driver, controller).run(configs, _SERVER, _CTX)
    assert controller.capture_args == [[(128, 64), (256, 64)]]


def test_skip_trace_capture_skips_warmup():
    controller = FakeController()
    _runner(FakeDriver([_ok()]), controller).run(
        [_cfg()], _SERVER, _CTX, skip_trace_capture=True
    )
    assert controller.capture_args == []


def test_capture_trace_failure_is_non_fatal():
    driver = FakeDriver([_ok()])
    controller = FakeController(capture_exc=RuntimeError("trace boom"))
    result = _runner(driver, controller).run([_cfg()], _SERVER, _CTX)
    assert len(result.blocks) == 1
    assert result.ok is True


def test_unhealthy_server_aborts_before_running_driver():
    driver = FakeDriver([_ok()])
    controller = FakeController(healthy=False)
    result = _runner(driver, controller).run([_cfg()], _SERVER, _CTX)
    assert driver.run_calls == []
    assert result.return_codes == [1]
    assert result.ok is False


def test_no_controller_skips_health_gate_but_runs_driver():
    driver = FakeDriver([_ok()])
    result = LLMPerformanceRunner(driver, None, inter_run_sleep_s=0.0).run(
        [_cfg()], _SERVER, _CTX
    )
    assert len(driver.run_calls) == 1
    assert result.ok is True


def test_nonzero_driver_exit_skips_parse_and_marks_not_ok():
    driver = FakeDriver([DriverResult(return_code=2, raw=None, raw_path=None), _ok()])
    result = _runner(driver, FakeController()).run([_cfg(), _cfg()], _SERVER, _CTX)
    # First point failed (no block), second succeeded.
    assert len(result.blocks) == 1
    assert result.return_codes == [2, 0]
    assert result.ok is False


def test_zero_exit_but_no_raw_records_parse_failure():
    driver = FakeDriver([DriverResult(return_code=0, raw=None, raw_path=None)])
    result = _runner(driver, FakeController()).run([_cfg()], _SERVER, _CTX)
    assert result.blocks == []
    assert result.parse_failures == [1]
    assert result.ok is False


def test_mid_sweep_unhealthy_status_aborts():
    driver = FakeDriver([_ok(), _ok()])
    controller = FakeController(health_status=503)
    result = _runner(driver, controller).run([_cfg(), _cfg()], _SERVER, _CTX)
    # get_health returns 503 before the first point -> abort immediately.
    assert driver.run_calls == []
    assert result.return_codes == [1]


def test_health_check_request_exception_aborts():
    driver = FakeDriver([_ok()])
    controller = FakeController(
        health_exc=requests.exceptions.ConnectionError("down")
    )
    result = _runner(driver, controller).run([_cfg()], _SERVER, _CTX)
    assert driver.run_calls == []
    assert result.return_codes == [1]


class TestRunnerResultOk:
    def test_ok_requires_codes_all_zero_and_no_parse_failures(self):
        assert RunnerResult(return_codes=[0, 0]).ok is True
        assert RunnerResult(return_codes=[]).ok is False
        assert RunnerResult(return_codes=[0, 1]).ok is False
        assert RunnerResult(return_codes=[0], parse_failures=[1]).ok is False
