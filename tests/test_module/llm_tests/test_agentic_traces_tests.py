# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the agentic trace-replay orchestrator.

The driver is stubbed throughout: what is under test is the sweep loop's
bookkeeping. The failure paths matter most, because each one is a way a broken
run could otherwise be reported as a clean pass:

* a model with no config, or an unknown trace-source name, must not run,
* each run is dispatched to the driver for its trace source,
* a failed run must record a non-zero code even if a sibling run succeeded,
* the per-run subprocess timeout must exceed the planned wall-clock window.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_module.agentic_traces import estimated_run_seconds
from llm_module.drivers.aiperf_agentic_traces import AgenticTracesDriverResult
from test_module.llm_tests import agentic_traces_tests
from test_module.llm_tests.agentic_traces_tests import run_agentic_traces
from workflow_module.blocks_sink import get_default_accumulator

KIMI_MODEL_ID = "id_blaze_Kimi-K2.7-Code_super_cluster"


@pytest.fixture(autouse=True)
def _clean_accumulator():
    """Keep accept_blocks() out of sibling tests' report state."""
    get_default_accumulator().clear()
    yield
    get_default_accumulator().clear()


def _ctx(model_id: str = KIMI_MODEL_ID, tmp_path=None):
    ctx = MagicMock()
    ctx.model_spec.model_id = model_id
    ctx.model_spec.model_name = "Kimi-K2.7-Code"
    ctx.model_spec.hf_model_repo = "moonshotai/Kimi-K2.7-Code"
    ctx.model_spec.metadata = {"tokenizer_trust_remote_code": True}
    ctx.model_spec.device_model_spec.max_context = 262144
    ctx.device.name = "super_cluster"
    ctx.server_host = "http://localhost"
    ctx.server_port = 8000
    ctx.output_path = str(tmp_path) if tmp_path else "/tmp/agentic_traces_test"
    return ctx


def _ok_payload(label: str = "run") -> dict:
    return {
        "model_id": "moonshotai/Kimi-K2.7-Code",
        "label": label,
        "date": "20260727-120000",
        "completed": 100,
        "error_request_count": 0,
        "metadata": {"inferencex_git_ref": "abc123"},
    }


def _driver_returning(*outcomes):
    """Patch the AIPerf driver so each run() call returns the next outcome.

    Also stubs the swo-bench driver with a driver that raises if called, so an
    inferencex-narrowed sweep that accidentally dispatched a swarmone run would
    fail loudly rather than silently shell out.
    """
    driver = MagicMock()
    driver.run.side_effect = list(outcomes)
    swo_driver = MagicMock()
    swo_driver.run.side_effect = AssertionError(
        "swo-bench driver called in an inferencex-only sweep"
    )
    patcher = patch.multiple(
        agentic_traces_tests,
        AIPerfAgenticTracesDriver=MagicMock(return_value=driver),
        SwoBenchAgenticTracesDriver=MagicMock(return_value=swo_driver),
    )
    return patcher, driver


def _swo_driver_returning(*outcomes):
    """Patch the swo-bench driver; stub the AIPerf driver to fail if called."""
    swo_driver = MagicMock()
    swo_driver.run.side_effect = list(outcomes)
    aiperf_driver = MagicMock()
    aiperf_driver.run.side_effect = AssertionError(
        "AIPerf driver called in a swarmone-only sweep"
    )
    patcher = patch.multiple(
        agentic_traces_tests,
        AIPerfAgenticTracesDriver=MagicMock(return_value=aiperf_driver),
        SwoBenchAgenticTracesDriver=MagicMock(return_value=swo_driver),
    )
    return patcher, swo_driver


class TestPlanningFailures:
    def test_unconfigured_model_does_not_run(self, tmp_path):
        patcher, driver = _driver_returning()
        with patcher:
            result = run_agentic_traces(_ctx(model_id="id_nope", tmp_path=tmp_path))

        assert result.return_codes == [1]
        assert result.blocks == []
        driver.run.assert_not_called()

    def test_unknown_trace_source_name_does_not_run(self, tmp_path):
        patcher, driver = _driver_returning()
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path), trace_sources="not_a_source"
            )

        assert result.return_codes == [1]
        driver.run.assert_not_called()

    def test_unhealthy_server_aborts_before_any_run(self, tmp_path):
        controller = MagicMock()
        controller.wait_for_healthy.return_value = False
        patcher, driver = _driver_returning()
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path), server_controller=controller
            )

        assert result.return_codes == [1]
        driver.run.assert_not_called()


class TestSweepLoop:
    def test_successful_run_emits_a_block(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, driver = _driver_returning(outcome)
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                inter_run_sleep_s=0,
            )

        assert result.return_codes == [0]
        assert result.ok
        assert [b.kind for b in result.blocks] == ["agentic_traces"]
        assert len(get_default_accumulator().blocks) == 1

    def test_failed_run_records_its_code_and_emits_no_block(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=124, payload=None, raw_path=None
        )
        patcher, _ = _driver_returning(outcome)
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                inter_run_sleep_s=0,
            )

        assert result.return_codes == [124]
        assert result.blocks == []
        assert not result.ok

    def test_timeout_exceeds_the_planned_window(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, driver = _driver_returning(outcome)
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                inter_run_sleep_s=0,
            )

        run, _server, context = driver.run.call_args[0]
        planned = run.benchmark_duration + run.warmup_grace_period
        assert context.per_run_timeout_s > planned

    def test_git_ref_override_flows_into_the_run(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, driver = _driver_returning(outcome)
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                git_ref_override="cafebabe",
                inter_run_sleep_s=0,
            )

        run = driver.run.call_args[0][0]
        assert run.metadata["inferencex_git_ref"] == "cafebabe"

    def test_auth_token_reaches_the_server_connection(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, driver = _driver_returning(outcome)
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                auth_token="tok123",
                inter_run_sleep_s=0,
            )

        server = driver.run.call_args[0][1]
        assert server.auth_token == "tok123"
        assert server.url_with_port == "http://localhost:8000"


class TestTraceSourceDispatch:
    def test_swarmone_dispatches_to_the_swo_driver(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, swo_driver = _swo_driver_returning(outcome)
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="swarmone",
                inter_run_sleep_s=0,
            )

        assert result.return_codes == [0]
        assert [b.kind for b in result.blocks] == ["agentic_traces"]
        swo_driver.run.assert_called_once()
        run = swo_driver.run.call_args[0][0]
        assert run.trace_source.value == "swarmone"

    def _both_drivers(self):
        aiperf_driver = MagicMock()
        aiperf_driver.run.return_value = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload("aiperf"), raw_path=None
        )
        swo_driver = MagicMock()
        swo_driver.run.return_value = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload("swo"), raw_path=None
        )
        patcher = patch.multiple(
            agentic_traces_tests,
            AIPerfAgenticTracesDriver=MagicMock(return_value=aiperf_driver),
            SwoBenchAgenticTracesDriver=MagicMock(return_value=swo_driver),
        )
        return patcher, aiperf_driver, swo_driver

    def test_default_sweep_skips_the_opt_in_swarmone_source(self, tmp_path):
        """Without --agentic-traces-sources the sweep must stay as it was
        before SwarmOne was configured: InferenceX only, no license needed."""
        patcher, aiperf_driver, swo_driver = self._both_drivers()
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path), mode="ci", inter_run_sleep_s=0
            )

        assert result.return_codes == [0]
        aiperf_driver.run.assert_called_once()
        swo_driver.run.assert_not_called()

    def test_naming_both_sources_runs_both(self, tmp_path):
        patcher, aiperf_driver, swo_driver = self._both_drivers()
        with patcher:
            result = run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx,swarmone",
                inter_run_sleep_s=0,
            )

        assert result.return_codes == [0, 0]
        aiperf_driver.run.assert_called_once()
        swo_driver.run.assert_called_once()

    def test_swo_driver_gets_a_generous_timeout(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, swo_driver = _swo_driver_returning(outcome)
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="swarmone",
                inter_run_sleep_s=0,
            )

        _run, _server, context = swo_driver.run.call_args[0]
        # SwarmOne CI timeout floor is 1h plus headroom.
        assert context.per_run_timeout_s >= 3600

    def test_each_run_gets_its_own_timeout(self, tmp_path):
        """A sweep-wide maximum would hand SwarmOne's multi-hour allowance to
        the InferenceX run, leaving a hung one parked on the cluster for it."""
        patcher, aiperf_driver, swo_driver = self._both_drivers()
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx,swarmone",
                inter_run_sleep_s=0,
            )

        aiperf_run, _, aiperf_context = aiperf_driver.run.call_args[0]
        swo_run, _, swo_context = swo_driver.run.call_args[0]
        headroom = agentic_traces_tests._TIMEOUT_HEADROOM_SECONDS
        assert aiperf_context.per_run_timeout_s == (
            estimated_run_seconds(aiperf_run) + headroom
        )
        assert swo_context.per_run_timeout_s == (
            estimated_run_seconds(swo_run) + headroom
        )
        assert aiperf_context.per_run_timeout_s < swo_context.per_run_timeout_s

    def test_artifacts_land_under_the_output_path(self, tmp_path):
        outcome = AgenticTracesDriverResult(
            return_code=0, payload=_ok_payload(), raw_path=None
        )
        patcher, _ = _driver_returning(outcome)
        with patcher:
            run_agentic_traces(
                _ctx(tmp_path=tmp_path),
                mode="ci",
                trace_sources="inferencex_agentx",
                inter_run_sleep_s=0,
            )

        # The orchestrator creates the output root; the (mocked) driver would
        # create its own per-run artifact subtree.
        assert (tmp_path / "agentic_traces").is_dir()
