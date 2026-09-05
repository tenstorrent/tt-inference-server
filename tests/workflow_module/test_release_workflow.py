# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ReleaseWorkflow child selection + delegation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from types import SimpleNamespace

from workflows.workflow_types import ModelType, WorkflowVenvType
from workflow_module.execution import (
    AgenticTracesOptions,
    LLMBenchOptions,
    OrchestratorMetadata,
    PrefixCacheOptions,
    SpecDecodeOptions,
    TaskOutcome,
)
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
        tasks.append(SimpleNamespace(workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC))
    ctx.all_params = SimpleNamespace(tasks=tasks)
    return ctx


def _fake_child(name):
    """A registry entry that records construction and returns one outcome."""
    instance = MagicMock()
    instance.run_tasks.return_value = [TaskOutcome(name, 0, 0.0, name)]
    return MagicMock(return_value=instance), instance


def _run_release(ctx, meta=None):
    evals_cls, evals_inst = _fake_child("evals")
    bench_cls, bench_inst = _fake_child("benchmarks")
    spec_cls, spec_inst = _fake_child("spec_tests")
    agentic_cls, agentic_inst = _fake_child("agentic")
    traces_cls, traces_inst = _fake_child("agentic_traces")
    acc = MagicMock()
    meta = meta or OrchestratorMetadata(server_mode="local")
    registry = {
        "evals": evals_cls,
        "benchmarks": bench_cls,
        "spec_tests": spec_cls,
        "agentic": agentic_cls,
        "agentic_traces": traces_cls,
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
            "agentic_traces": traces_cls,
        },
        acc,
        meta,
    )


class TestReleaseWorkflowRegistry:
    def test_registered(self):
        assert get_workflow_class("release") is ReleaseWorkflow

    def test_llm_children_include_spec_tests(self):
        assert ReleaseWorkflow.llm_children == ("evals", "benchmarks", "spec_tests")
        assert ReleaseWorkflow.children == ("evals", "benchmarks", "spec_tests")


class TestReleaseWorkflowChildSelection:
    def test_llm_without_agentic_runs_full_suite(self):
        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.LLM))
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_called_once()
        classes["agentic"].assert_not_called()
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
        ]

    def test_llm_with_agentic_task_runs_agentic_child(self):
        outcomes, classes, _acc, _meta = _run_release(
            _make_ctx(ModelType.LLM, agentic=True)
        )
        classes["evals"].assert_called_once()
        classes["benchmarks"].assert_called_once()
        classes["spec_tests"].assert_called_once()
        classes["agentic"].assert_called_once()
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
            "agentic",
        ]

    def test_llm_omits_agentic_child_when_all_tasks_exceed_context(self):
        ctx = _make_ctx(ModelType.LLM, agentic=True)
        task = ctx.all_params.tasks[0]
        task.task_name = "swe_bench_verified"
        task.agentic_eval_config = None
        task.swebench_eval_config = SimpleNamespace(
            max_input_tokens=160 * 1024,
            max_output_tokens=32 * 1024,
        )
        task.min_context_required = None
        ctx.model_spec.device_model_spec = SimpleNamespace(max_context=32 * 1024)
        ctx.runtime_config = SimpleNamespace(agentic_benchmark=None)

        outcomes, classes, _acc, _meta = _run_release(ctx)

        classes["agentic"].assert_not_called()
        assert [outcome.task_type for outcome in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
        ]

    def test_agentic_traces_child_is_opt_in(self):
        """Without --agentic-traces the options are None, so no child runs: a
        full-mode replay is roughly an hour of profiling per configured run."""
        _outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.LLM))
        classes["agentic_traces"].assert_not_called()

    def test_agentic_traces_runs_last_when_opted_in(self):
        meta = OrchestratorMetadata(
            server_mode="local",
            agentic_traces=AgenticTracesOptions(mode="ci"),
        )

        outcomes, classes, acc, _meta = _run_release(_make_ctx(ModelType.LLM), meta)

        classes["agentic_traces"].assert_called_once()
        # Last, so a failure in the cheaper children surfaces without waiting.
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "spec_tests",
            "agentic_traces",
        ]
        # Shares the accumulator, so its Blocks land in the one release report.
        kwargs = classes["agentic_traces"].call_args.kwargs
        assert kwargs["accumulator"] is acc
        assert kwargs["orchestrator_metadata"].agentic_traces is meta.agentic_traces

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
        spec_cls, spec_inst = _fake_child("spec_tests")
        registry = {
            "evals": evals_cls,
            "benchmarks": bench_cls,
            "spec_tests": spec_cls,
        }
        with patch.dict(
            "workflow_module.workflows.WORKFLOW_REGISTRY", registry, clear=False
        ):
            ReleaseWorkflow(ctx, accumulator=MagicMock()).run_tasks()
        evals_inst.run_tasks.assert_called_once()
        bench_inst.run_tasks.assert_called_once()
        spec_inst.run_tasks.assert_called_once()

    def test_llm_release_expands_optional_benchmark_sweeps(self):
        prefix_opts = PrefixCacheOptions(preset="ci")
        spec_opts = SpecDecodeOptions(preset="ci")
        bench_opts = LLMBenchOptions(tools="vllm")
        meta = OrchestratorMetadata(
            server_mode="local",
            prefix_cache=prefix_opts,
            spec_decode=spec_opts,
            llm_bench=bench_opts,
        )

        outcomes, classes, _acc, _meta = _run_release(_make_ctx(ModelType.LLM), meta)

        classes["evals"].assert_called_once()
        assert classes["benchmarks"].call_count == 3
        assert classes["agentic"].call_count == 0
        assert [o.task_type for o in outcomes] == [
            "evals",
            "benchmarks",
            "benchmarks",
            "benchmarks",
            "spec_tests",
        ]

        bench_metas = [
            call.kwargs["orchestrator_metadata"]
            for call in classes["benchmarks"].call_args_list
        ]
        standard_meta, prefix_meta, spec_meta = bench_metas

        assert standard_meta.prefix_cache is None
        assert standard_meta.spec_decode is None
        assert standard_meta.llm_bench is bench_opts

        assert prefix_meta.prefix_cache is prefix_opts
        assert prefix_meta.spec_decode is None
        assert prefix_meta.llm_bench is None

        assert spec_meta.prefix_cache is None
        assert spec_meta.spec_decode is spec_opts
        assert spec_meta.llm_bench is None
