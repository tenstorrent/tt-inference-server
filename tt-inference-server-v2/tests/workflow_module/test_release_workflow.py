# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ReleaseWorkflow child selection + delegation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from workflows.workflow_types import ModelType, WorkflowVenvType
from workflow_module.execution import OrchestratorMetadata, TaskOutcome
from workflow_module.workflows import (
    ReleaseWorkflow,
    get_workflow_class,
)

_RELEASE_LLM_CHILDREN = ("evals", "benchmarks", "spec_tests", "agentic")


def _agentic_task():
    return SimpleNamespace(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
    )


def _make_ctx(model_type, tasks=()):
    ctx = MagicMock()
    ctx.model_spec.model_name = "test-model"
    ctx.model_spec.model_type = model_type
    ctx.device.name = "gpu"
    ctx.service_port = 8000
    ctx.output_path = "/tmp/test_output"
    ctx.all_params.tasks = list(tasks)
    return ctx


def _fake_child(name):
    """A registry entry that records construction and returns one outcome."""
    instance = MagicMock()
    instance.run_tasks.return_value = [TaskOutcome(name, 0, 0.0, name)]
    cls = MagicMock(return_value=instance)
    cls.is_applicable.return_value = True
    return cls, instance


def _run_release(ctx):
    registry = {}
    classes = {}
    for name in _RELEASE_LLM_CHILDREN:
        cls, _inst = _fake_child(name)
        registry[name] = cls
        classes[name] = cls
    acc = MagicMock()
    meta = OrchestratorMetadata(server_mode="local")
    with patch.dict(
        "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
    ):
        wf = ReleaseWorkflow(ctx, accumulator=acc, orchestrator_metadata=meta)
        outcomes = wf.run_tasks()
    return outcomes, classes, acc, meta


class TestReleaseWorkflowRegistry:
    def test_registered(self):
        assert get_workflow_class("release") is ReleaseWorkflow

    def test_llm_children_include_agentic(self):
        assert ReleaseWorkflow.llm_children == _RELEASE_LLM_CHILDREN
        assert ReleaseWorkflow.children == ("evals", "benchmarks", "spec_tests")


class TestReleaseWorkflowChildSelection:
    def test_llm_runs_full_suite(self):
        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.LLM))
        for name in _RELEASE_LLM_CHILDREN:
            classes[name].assert_called_once()
        assert [o.task_type for o in outcomes] == list(_RELEASE_LLM_CHILDREN)

    def test_media_runs_full_suite_without_agentic(self):
        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.IMAGE))
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_called_once()
        classes["agentic"].assert_not_called()
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
        ]

    def test_inapplicable_child_is_skipped(self):
        ctx = _make_ctx(ModelType.LLM)
        registry = {}
        classes = {}
        for name in _RELEASE_LLM_CHILDREN:
            cls, _inst = _fake_child(name)
            registry[name] = cls
            classes[name] = cls
        classes["agentic"].is_applicable.return_value = False
        with patch.dict(
            "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
        ):
            outcomes = ReleaseWorkflow(ctx, accumulator=MagicMock()).run_tasks()
        classes["agentic"].assert_not_called()
        assert [o.task_type for o in outcomes] == ["evals", "benchmarks", "spec_tests"]

    def test_agentic_applicability_follows_eval_tasks(self):
        from workflow_module.workflows import AgenticWorkflow

        with_tasks = _make_ctx(ModelType.LLM, tasks=[_agentic_task()])
        without_tasks = _make_ctx(ModelType.LLM)
        assert AgenticWorkflow.is_applicable(with_tasks)
        assert not AgenticWorkflow.is_applicable(without_tasks)

    def test_children_share_accumulator_and_metadata(self):
        ctx = _make_ctx(ModelType.LLM)
        _outcomes, classes, acc, meta = _run_release(ctx)
        for name in ("evals", "benchmarks", "agentic"):
            _args, kwargs = classes[name].call_args
            assert kwargs["accumulator"] is acc
            assert kwargs["orchestrator_metadata"] is meta

    def test_child_run_tasks_is_invoked(self):
        ctx = _make_ctx(ModelType.LLM)
        registry = {}
        instances = {}
        for name in _RELEASE_LLM_CHILDREN:
            cls, inst = _fake_child(name)
            registry[name] = cls
            instances[name] = inst
        with patch.dict(
            "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
        ):
            ReleaseWorkflow(ctx, accumulator=MagicMock()).run_tasks()
        for inst in instances.values():
            inst.run_tasks.assert_called_once()
