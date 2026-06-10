# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ExaboxWorkflow registration, run_tasks behaviour, and the runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from test_module.exabox.runner import ExaboxResult, available_tests, run_exabox
from workflow_module.execution import ExaboxOptions, OrchestratorMetadata
from workflow_module.workflows import (
    ExaboxWorkflow,
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


class TestExaboxWorkflowRegistry:
    def test_registered(self):
        assert "exabox" in WORKFLOW_REGISTRY
        assert WORKFLOW_REGISTRY["exabox"] is ExaboxWorkflow

    def test_get_workflow_class(self):
        assert get_workflow_class("exabox") is ExaboxWorkflow

    def test_name(self):
        assert ExaboxWorkflow.name == "exabox"


class TestExaboxWorkflowRunTasks:
    def _make_workflow(self, tests=None):
        metadata = OrchestratorMetadata(exabox=ExaboxOptions(tests=tests))
        return ExaboxWorkflow(_make_ctx(), orchestrator_metadata=metadata)

    def test_one_outcome_per_suite(self):
        wf = self._make_workflow(tests="benchmark,summarize_bench")
        results = [
            ExaboxResult("benchmark", 0, 1.0),
            ExaboxResult("summarize_bench", 7, 2.0),
        ]
        with patch(
            "test_module.exabox.runner.run_exabox", return_value=results
        ) as mock_run:
            outcomes = wf.run_tasks()

        mock_run.assert_called_once_with(wf.ctx, tests="benchmark,summarize_bench")
        assert [o.task_type for o in outcomes] == [
            "exabox:benchmark",
            "exabox:summarize_bench",
        ]
        assert [o.exit_code for o in outcomes] == [0, 7]
        assert all(o.block_kind == "exabox" for o in outcomes)

    def test_runner_raises_returns_failed_outcome(self):
        wf = self._make_workflow(tests="nope")
        with patch(
            "test_module.exabox.runner.run_exabox",
            side_effect=ValueError("unknown test"),
        ):
            outcomes = wf.run_tasks()

        assert len(outcomes) == 1
        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind is None

    def test_no_metadata_runs_all(self):
        wf = ExaboxWorkflow(_make_ctx())
        with patch(
            "test_module.exabox.runner.run_exabox",
            return_value=[ExaboxResult("benchmark", 0, 1.0)],
        ) as mock_run:
            wf.run_tasks()

        mock_run.assert_called_once_with(wf.ctx, tests=None)


class TestExaboxRunner:
    def test_available_tests_finds_all_suites(self):
        suites = available_tests()
        assert "benchmark" in suites
        assert "_helpers" not in suites
        for suite in suites:
            assert not suite.startswith("_")

    def test_unknown_test_raises(self):
        with pytest.raises(ValueError, match="Unknown exabox test"):
            run_exabox(_make_ctx(), tests="not_a_suite")

    def test_hyphenated_names_normalized(self, tmp_path):
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.exabox.runner.subprocess.run", return_value=completed
        ) as mock_run:
            results = run_exabox(ctx, tests="long-context-bench")

        assert results == [
            ExaboxResult("long_context_bench", 0, results[0].elapsed_seconds)
        ]
        cmd = mock_run.call_args[0][0]
        assert "long_context_bench" in cmd

    def test_target_is_ctx_base_url(self, tmp_path):
        ctx = _make_ctx()
        ctx.output_path = str(tmp_path)
        ctx.base_url = "http://remote:9000"
        completed = MagicMock(returncode=0)
        with patch(
            "test_module.exabox.runner.subprocess.run", return_value=completed
        ) as mock_run:
            run_exabox(ctx, tests="benchmark")

        cmd = mock_run.call_args[0][0]
        target = cmd[cmd.index("--target") + 1]
        assert target == "http://remote:9000"
