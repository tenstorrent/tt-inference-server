# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ReleaseWorkflow child selection + delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from types import SimpleNamespace

from workflows.workflow_types import ModelType, WorkflowVenvType
from workflow_module.execution import OrchestratorMetadata, TaskOutcome
from workflow_module.workflows import (
    ReleaseWorkflow,
    get_workflow_class,
)


def _make_ctx(model_type, *, agentic=False):
    ctx = MagicMock()
    ctx.model_spec.model_name = "test-model"
    ctx.model_spec.model_type = model_type
    ctx.device.name = "gpu"
    ctx.service_port = 8000
    ctx.output_path = "/tmp/test_output"
    tasks = []
    if agentic:
        tasks.append(
            SimpleNamespace(workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC)
        )
    ctx.all_params = SimpleNamespace(tasks=tasks)
    return ctx


def _fake_child(name):
    """A registry entry that records construction and returns one outcome."""
    instance = MagicMock()
    instance.run_tasks.return_value = [TaskOutcome(name, 0, 0.0, name)]
    return MagicMock(return_value=instance), instance


def _run_release(ctx):
    evals_cls, evals_inst = _fake_child("evals")
    bench_cls, bench_inst = _fake_child("benchmarks")
    spec_cls, spec_inst = _fake_child("spec_tests")
    agentic_cls, agentic_inst = _fake_child("agentic")
    acc = MagicMock()
    meta = OrchestratorMetadata(server_mode="local")
    registry = {
        "evals": evals_cls,
        "benchmarks": bench_cls,
        "spec_tests": spec_cls,
        "agentic": agentic_cls,
    }
    with patch.dict(
        "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
    ):
        wf = ReleaseWorkflow(ctx, accumulator=acc, orchestrator_metadata=meta)
        outcomes = wf.run_tasks()
    return (
        outcomes,
        {
            "evals": evals_cls,
            "benchmarks": bench_cls,
            "spec_tests": spec_cls,
            "agentic": agentic_cls,
        },
        acc,
        meta,
    )


class TestReleaseWorkflowRegistry:
    def test_registered(self):
        assert get_workflow_class("release") is ReleaseWorkflow

    def test_llm_children_drops_spec_tests(self):
        assert ReleaseWorkflow.llm_children == ("evals", "benchmarks")
        assert ReleaseWorkflow.children == ("evals", "benchmarks", "spec_tests")


class TestReleaseWorkflowChildSelection:
    def test_llm_without_agentic_runs_evals_and_benchmarks_only(self):
        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.LLM))
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_not_called()
        classes["agentic"].assert_not_called()
        assert [o.task_type for o in outcomes] == ["evals", "benchmarks"]

    def test_llm_with_agentic_task_runs_agentic_child(self):
        outcomes, classes, _acc, _meta = _run_release(
            _make_ctx(ModelType.LLM, agentic=True)
        )
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_not_called()
        classes["agentic"].assert_called_once()
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "agentic",
        ]

    def test_media_runs_full_suite(self):
        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.IMAGE))
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_called_once()
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
        ]

    def test_children_share_accumulator_and_metadata(self):
        ctx = _make_ctx(ModelType.LLM)
        _outcomes, classes, acc, meta = _run_release(ctx)
        for name in ("evals", "benchmarks"):
            _args, kwargs = classes[name].call_args
            assert kwargs["accumulator"] is acc
            assert kwargs["orchestrator_metadata"] is meta

    def test_child_run_tasks_is_invoked(self):
        ctx = _make_ctx(ModelType.LLM)
        evals_cls, evals_inst = _fake_child("evals")
        bench_cls, bench_inst = _fake_child("benchmarks")
        registry = {"evals": evals_cls, "benchmarks": bench_cls}
        with patch.dict(
            "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
        ):
            ReleaseWorkflow(ctx, accumulator=MagicMock()).run_tasks()
        evals_inst.run_tasks.assert_called_once()
        bench_inst.run_tasks.assert_called_once()
