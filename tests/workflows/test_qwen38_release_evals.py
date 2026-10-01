# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Qwen3.8 Terminal-Bench performance experiment configuration."""

from reference_config.evals.eval_config import _eval_config_map
from workflows.workflow_types import EvalLimitMode


def test_evals_provisions_agentic_dependencies(monkeypatch):
    from types import SimpleNamespace
    from workflows import workflow_dispatch as dispatch
    from workflows.workflow_types import ModelType, WorkflowType, WorkflowVenvType

    spec = SimpleNamespace(model_type=ModelType.LLM, hf_model_repo="Qwen/Qwen3.8-27B")
    monkeypatch.setattr(dispatch, "_llm_eval_venv_types", lambda *args: [])
    monkeypatch.setattr(dispatch, "_llm_release_includes_agentic", lambda *args: True)
    dependencies = dispatch._engine_dependency_venv_types(spec, WorkflowType.EVALS)
    assert WorkflowVenvType.EVALS_AGENTIC in dependencies


def test_combined_evals_reports_three_suites_and_rejects_missing(monkeypatch):
    from types import SimpleNamespace
    from report_module.schema import Block
    from workflow_module.blocks_sink import BlockAccumulator
    from workflow_module.execution import TaskOutcome
    from workflow_module.workflows import EvalsWorkflow, AgenticWorkflow
    from workflows.workflow_types import (
        ModelType,
        DeviceTypes,
        WorkflowVenvType,
    )
    import llm_module.eval_configs as configs

    tasks = [
        SimpleNamespace(
            task_name="standard_probe",
            workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
        ),
        SimpleNamespace(
            task_name="agentic_probe",
            workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        ),
    ]
    ctx = SimpleNamespace(
        model_spec=SimpleNamespace(model_type=ModelType.LLM),
        runtime_config=SimpleNamespace(workflow="evals"),
        all_params=SimpleNamespace(tasks=tasks),
        device=DeviceTypes.P300X2,
    )
    monkeypatch.setattr(configs, "get_llm_eval_tasks", lambda *args: tasks[:1])

    def standard(self):
        self.accumulator.accept(
            [Block(kind="evals", data={}, targets={"task_name": tasks[0].task_name})]
        )
        return TaskOutcome("evaluation", 0, 0.0, "evals")

    def agentic(self):
        self.accumulator.accept(
            [
                Block(kind="evals", data={}, targets={"task_name": t.task_name})
                for t in tasks[1:]
            ]
        )
        return [TaskOutcome("evaluation", 0, 0.0, "evals")]

    monkeypatch.setattr(EvalsWorkflow, "_run_llm_eval_task", standard)
    monkeypatch.setattr(AgenticWorkflow, "run_tasks", agentic)
    accumulator = BlockAccumulator()
    outcomes = EvalsWorkflow(ctx, accumulator=accumulator).run_tasks()
    assert all(o.succeeded for o in outcomes)
    assert [b.targets["task_name"] for b in accumulator.blocks] == [
        t.task_name for t in tasks
    ]

    monkeypatch.setattr(AgenticWorkflow, "run_tasks", lambda self: [])
    outcomes = EvalsWorkflow(ctx, accumulator=BlockAccumulator()).run_tasks()
    assert any(o.task_type == "missing_evals" and not o.succeeded for o in outcomes)

    # Release has a separate agentic child; do not execute it twice.
    ctx.runtime_config.workflow = "release"
    monkeypatch.setattr(
        AgenticWorkflow,
        "run_tasks",
        lambda self: (_ for _ in ()).throw(AssertionError("duplicate agentic")),
    )
    assert len(EvalsWorkflow(ctx, accumulator=BlockAccumulator()).run_tasks()) == 1


def test_qwen38_b8_evals_enforce_complete_acceptance():
    from pathlib import Path

    import yaml

    catalog = Path(__file__).resolve().parents[2] / "workflows/model_specs/dev/llm.yaml"
    templates = yaml.safe_load(catalog.read_text())["templates"]
    template = next(t for t in templates if t["impl"] == "qwen38_autoport_b8")
    assert template["status"] == "COMPLETE"


def test_qwen38_perf_branch_runs_only_terminal_bench():
    tasks = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    assert [task.task_name for task in tasks] == ["terminal_bench_2_1"]


def test_qwen38_terminal_perf_cohort_budget_and_qemu_pin():
    (terminal,) = _eval_config_map["Qwen/Qwen3.8-27B"].tasks
    terminal_config = terminal.agentic_eval_config
    assert terminal_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/break-filter-js-from-html",
        "terminal-bench/cobol-modernization",
        "terminal-bench/compile-compcert",
        "terminal-bench/feal-differential-cryptanalysis",
        "terminal-bench/qemu-startup",
        "terminal-bench/caffe-cifar-10",
        "terminal-bench/password-recovery",
        "terminal-bench/portfolio-optimization",
        "terminal-bench/hf-model-inference",
        "terminal-bench/financial-document-processor",
    ]
    assert terminal_config.task_overrides["terminal-bench/qemu-startup"] == {
        "path": "tasks/qemu-startup",
        "git_url": "https://github.com/mvasiljevicTT/terminal-bench-2-1.git",
        "git_commit_id": "a355fc6aaeaf62ba94b6cab023e179c7e440c651",
    }
    assert terminal_config.agent_timeout_sec == 6 * 60 * 60
    assert terminal_config.agent_kwargs["model_info"]["max_output_tokens"] == 16 * 1024
    assert terminal_config.agent_kwargs["llm_kwargs"]["max_tokens"] == 16 * 1024
    assert terminal_config.agent_kwargs["llm_kwargs"]["extra_body"] == {
        "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": "medium"},
    }
    assert terminal_config.agent_kwargs["temperature"] == 1.0
    assert terminal_config.n_concurrent_trials == 5
