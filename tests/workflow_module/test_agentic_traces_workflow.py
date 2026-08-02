# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for AgenticTracesWorkflow registration, options wiring, and outcomes.

The wiring worth pinning down is that every CLI-level option reaches the
orchestrator (a dropped flag would silently run the wrong benchmark) and that a
partial sweep failure cannot report success.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_module.runner import RunnerResult
from report_module.schema import Block
from workflow_module.execution import AgenticTracesOptions, OrchestratorMetadata
from workflow_module.workflows import (
    AgenticTracesWorkflow,
    WORKFLOW_REGISTRY,
    get_workflow_class,
)

_ORCHESTRATOR = "test_module.llm_tests.agentic_traces_tests.run_agentic_traces"


def _make_ctx():
    ctx = MagicMock()
    ctx.model_spec.model_name = "Kimi-K2.7-Code"
    ctx.model_spec.model_id = "id_tt-transformers_Kimi-K2.7-Code_super_cluster"
    ctx.device.name = "super_cluster"
    ctx.service_port = 8000
    ctx.output_path = "/tmp/test_output"
    ctx.runtime_config = None
    return ctx


def _result(*, blocks=1, return_codes=(0,)) -> RunnerResult:
    return RunnerResult(
        blocks=[Block(kind="agentic_traces", data={}) for _ in range(blocks)],
        return_codes=list(return_codes),
    )


class TestRegistry:
    def test_registered(self):
        assert WORKFLOW_REGISTRY["agentic_traces"] is AgenticTracesWorkflow

    def test_get_workflow_class(self):
        assert get_workflow_class("agentic_traces") is AgenticTracesWorkflow

    def test_bypasses_the_media_task_dispatcher(self):
        assert AgenticTracesWorkflow.task_types == ()


class TestRunTasks:
    def _workflow(self, **option_kwargs):
        metadata = OrchestratorMetadata(
            agentic_traces=AgenticTracesOptions(**option_kwargs)
        )
        return AgenticTracesWorkflow(_make_ctx(), orchestrator_metadata=metadata)

    def test_every_option_reaches_the_orchestrator(self):
        wf = self._workflow(
            mode="ci",
            trace_sources="inferencex_agentx",
            duration_override=1200,
            git_ref_override="cafebabe",
            auth_token="tok",
            venv_python="/venv/bin/python",
        )
        with patch(_ORCHESTRATOR, return_value=_result()) as mock_run:
            outcomes = wf.run_tasks()

        kwargs = mock_run.call_args.kwargs
        assert kwargs["mode"] == "ci"
        assert kwargs["trace_sources"] == "inferencex_agentx"
        assert kwargs["duration_override"] == 1200
        assert kwargs["git_ref_override"] == "cafebabe"
        assert kwargs["auth_token"] == "tok"
        assert str(kwargs["venv_python"]) == "/venv/bin/python"
        assert [o.exit_code for o in outcomes] == [0]
        assert outcomes[0].task_type == "agentic_traces"
        assert outcomes[0].block_kind == "agentic_traces"

    def test_defaults_to_full_mode_without_metadata(self):
        wf = AgenticTracesWorkflow(_make_ctx())
        with patch(_ORCHESTRATOR, return_value=_result()) as mock_run:
            wf.run_tasks()

        kwargs = mock_run.call_args.kwargs
        assert kwargs["mode"] == "full"
        # venv_python stays None so the driver falls back to sys.executable, the
        # interpreter the launcher already re-exec'd into.
        assert kwargs["venv_python"] is None

    def test_partial_failure_fails_the_task(self):
        wf = self._workflow(mode="full")
        with patch(_ORCHESTRATOR, return_value=_result(return_codes=(0, 1))):
            outcomes = wf.run_tasks()

        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind == "agentic_traces"

    def test_no_blocks_fails_the_task(self):
        wf = self._workflow(mode="full")
        with patch(_ORCHESTRATOR, return_value=_result(blocks=0, return_codes=(1,))):
            outcomes = wf.run_tasks()

        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind is None

    def test_orchestrator_raising_fails_the_task(self):
        wf = self._workflow(mode="full")
        with patch(_ORCHESTRATOR, side_effect=RuntimeError("clone failed")):
            outcomes = wf.run_tasks()

        assert outcomes[0].exit_code == 1
        assert outcomes[0].block_kind is None
