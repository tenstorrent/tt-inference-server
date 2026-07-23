# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ServingBenchWorkflow registration, run_tasks, the runner, and presets."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from test_module.serving_bench.presets import preset_env_for_mode
from test_module.serving_bench.runner import (
    ServingBenchResult,
    available_suites,
    run_serving_bench,
)
from workflow_module.execution import OrchestratorMetadata, ServingBenchOptions
from workflow_module.workflows import (
    ServingBenchWorkflow,
    WORKFLOW_REGISTRY,
    get_workflow_class,
)


def _make_ctx():
    ctx = MagicMock()
    ctx.model_spec.model_name = "test-llm"
    ctx.device.name = "gpu"
    ctx.service_port = 8000
    ctx.output_path = "/tmp/test_output"
    ctx.base_url = "http://localhost:8000"
    ctx.runtime_config = None
    return ctx


class TestServingBenchWorkflowRegistry:
    def test_registered(self):
        assert "serving_bench" in WORKFLOW_REGISTRY
        assert WORKFLOW_REGISTRY["serving_bench"] is ServingBenchWorkflow

    def test_get_workflow_class(self):
        assert get_workflow_class("serving_bench") is ServingBenchWorkflow

    def test_name(self):
        assert ServingBenchWorkflow.name == "serving_bench"


class TestServingBenchWorkflowRunTasks:
    def _make_workflow(self, suites=None):
        metadata = OrchestratorMetadata(
            serving_bench=ServingBenchOptions(suites=suites)
        )
        return ServingBenchWorkflow(_make_ctx(), orchestrator_metadata=metadata)

    def test_one_outcome_per_suite(self):
        wf = self._make_workflow(suites="benchmark,agentic_bench")
        results = [
            ServingBenchResult("benchmark", 0, 1.0),
            ServingBenchResult("agentic_bench", 7, 2.0),
        ]
        with patch(
            "test_module.serving_bench.runner.run_serving_bench", return_value=results
        ) as mock_run:
            outcomes = wf.run_tasks()

        mock_run.assert_called_once_with(wf.ctx, suites="benchmark,agentic_bench")
        assert [o.task_type for o in outcomes] == [
            "serving_bench:benchmark",
            "serving_bench:agentic_bench",
        ]
        assert [o.exit_code for o in outcomes] == [0, 7]
        assert all(o.block_kind == "serving_bench" for o in outcomes)

    def test_runner_raises_returns_failed_outcome(self):
        wf = self._make_workflow(suites="nope")
        with patch(
            "test_module.serving_bench.runner.run_serving_bench",
            side_effect=ValueError("unknown suite"),
        ):
            outcomes = wf.run_tasks()

        assert len(outcomes) == 1
        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind is None

    def test_no_metadata_runs_all(self):
        wf = ServingBenchWorkflow(_make_ctx())
        with patch(
            "test_module.serving_bench.runner.run_serving_bench",
            return_value=[ServingBenchResult("benchmark", 0, 1.0)],
        ) as mock_run:
            wf.run_tasks()

        mock_run.assert_called_once_with(wf.ctx, suites=None)


class TestServingBenchRunner:
    def test_available_suites_finds_all(self):
        suites = available_suites()
        assert "benchmark" in suites
        assert "_helpers" not in suites
        for suite in suites:
            assert not suite.startswith("_")

    def test_unknown_suite_raises(self):
        with pytest.raises(ValueError, match="Unknown serving-bench suite"):
            run_serving_bench(_make_ctx(), suites="not_a_suite")

    def test_hyphenated_names_normalized(self, tmp_path):
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.serving_bench.runner.subprocess.run", return_value=completed
        ) as mock_run:
            results = run_serving_bench(ctx, suites="agentic-bench")

        assert results == [
            ServingBenchResult("agentic_bench", 0, results[0].elapsed_seconds)
        ]
        cmd = mock_run.call_args[0][0]
        assert "agentic_bench" in cmd

    def test_target_is_ctx_base_url(self, tmp_path):
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        ctx.base_url = "http://remote:9000"
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.serving_bench.runner.subprocess.run", return_value=completed
        ) as mock_run:
            run_serving_bench(ctx, suites="benchmark")

        cmd = mock_run.call_args[0][0]
        target = cmd[cmd.index("--target") + 1]
        assert target == "http://remote:9000"

    def test_limit_mode_preset_injected_into_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DURATION", raising=False)
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        ctx.runtime_config = MagicMock(limit_samples_mode="smoke-test")
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.serving_bench.runner.subprocess.run", return_value=completed
        ) as mock_run:
            run_serving_bench(ctx, suites="benchmark")

        env = mock_run.call_args.kwargs["env"]
        assert env["DURATION"] == "30"
        assert env["TARGET_CONCURRENCY"] == "2"

    def test_caller_env_wins_over_preset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DURATION", "999")
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        ctx.runtime_config = MagicMock(limit_samples_mode="smoke-test")
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.serving_bench.runner.subprocess.run", return_value=completed
        ) as mock_run:
            run_serving_bench(ctx, suites="benchmark")

        env = mock_run.call_args.kwargs["env"]
        assert env["DURATION"] == "999"


class TestPresets:
    def test_smoke_test_short_soak(self):
        env = preset_env_for_mode("smoke-test")
        assert env["DURATION"] == "30"

    def test_ci_nightly_full_soak(self):
        env = preset_env_for_mode("ci-nightly")
        assert env["DURATION"] == "3600"

    def test_none_and_unknown_return_empty(self):
        assert preset_env_for_mode(None) == {}
        assert preset_env_for_mode("ci-long") == {}
        assert preset_env_for_mode("bogus") == {}
