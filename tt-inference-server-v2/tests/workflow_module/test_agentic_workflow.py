# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for AgenticWorkflow registration and run_tasks behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from report_module.schema import Block
from test_module import MediaTaskType
from workflow_module.workflows import (
    AgenticWorkflow,
    WORKFLOW_REGISTRY,
    get_workflow_class,
)
from workflow_module.execution import TaskOutcome


def _fake_block() -> Block:
    return Block(
        kind="evals",
        task_type="llm",
        title="Agentic Eval — test_task",
        data={"success": True, "accuracy_check": 1, "accuracy": 0.6},
    )


def _make_ctx():
    ctx = MagicMock()
    ctx.model_spec.model_name = "test-llm"
    ctx.device.name = "gpu"
    ctx.service_port = 8000
    ctx.output_path = "/tmp/test_output"
    ctx.all_params.tasks = []
    return ctx


class TestAgenticWorkflowRegistry:
    def test_registered(self):
        assert "agentic" in WORKFLOW_REGISTRY
        assert WORKFLOW_REGISTRY["agentic"] is AgenticWorkflow

    def test_get_workflow_class(self):
        assert get_workflow_class("agentic") is AgenticWorkflow

    def test_task_types_is_evaluation(self):
        assert AgenticWorkflow.task_types == (MediaTaskType.EVALUATION,)

    def test_name(self):
        assert AgenticWorkflow.name == "agentic"


class TestAgenticWorkflowRunTasks:
    def _make_workflow(self, ctx=None):
        if ctx is None:
            ctx = _make_ctx()
        return AgenticWorkflow(ctx)

    def test_success_returns_single_task_outcome(self):
        wf = self._make_workflow()
        blocks = [_fake_block(), _fake_block()]
        with patch(
            "test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval",
            return_value=blocks,
        ):
            outcomes = wf.run_tasks()

        assert len(outcomes) == 1
        assert outcomes[0].exit_code == 0
        assert outcomes[0].block_kind == "evals"
        assert outcomes[0].task_type == "evaluation"

    def test_runner_raises_returns_failed_outcome(self):
        wf = self._make_workflow()
        with patch(
            "test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval",
            side_effect=RuntimeError("harness failed"),
        ):
            outcomes = wf.run_tasks()

        assert len(outcomes) == 1
        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind is None

    def test_runner_returns_empty_list_gives_failed_outcome(self):
        wf = self._make_workflow()
        with patch(
            "test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval",
            return_value=[],
        ):
            outcomes = wf.run_tasks()

        assert len(outcomes) == 1
        assert outcomes[0].exit_code == 1
