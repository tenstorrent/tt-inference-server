# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for v2 agentic eval parser, drivers, and bridge helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from llm_module import DriverContext, ServerConnection
from llm_module.drivers.agentic import (
    build_swebench_config,
    build_terminal_bench_config,
    resolve_instance_ids,
    resolve_n_tasks,
    resolve_task_names,
)
from llm_module.parsers.agentic import AgenticEvalParser, compute_accuracy_check
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks
from workflows.workflow_types import EvalLimitMode, ReportCheckTypes, WorkflowVenvType


@dataclass
class FakeScore:
    published_score: float = 0.5
    published_score_ref: str = "https://example.com"
    gpu_reference_score: Optional[float] = 0.45
    tolerance: float = 0.05


@dataclass
class FakeTerminalBenchConfig:
    dataset: str = "terminal-bench/terminal-bench-2"
    agent: str = "terminus-2"
    model: Optional[str] = None
    n_concurrent_trials: int = 5
    n_attempts: int = 1
    n_tasks: Optional[int] = 89
    task_names: List[str] = field(default_factory=list)
    exclude_task_names: List[str] = field(default_factory=list)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    environment_type: str = "docker"
    override_cpus: Optional[int] = 16
    override_memory_mb: Optional[int] = 48 * 1024
    timeout_multiplier: Optional[float] = None
    agent_timeout_sec: Optional[float] = 3 * 60 * 60
    quiet: bool = True
    yes: bool = True
    task_names_map: Dict[EvalLimitMode, List[str]] = field(default_factory=dict)


@dataclass
class FakeSWEbenchConfig:
    dataset_name: str = "SWE-bench/SWE-bench_Verified"
    sweagent_subset: str = "verified"
    dataset_split: str = "test"
    agent_backend: str = "mini-swe-agent"
    model: Optional[str] = None
    n_concurrent_trials: int = 5
    max_workers: int = 8
    n_tasks: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.95
    max_input_tokens: int = 200 * 1024
    max_output_tokens: Optional[int] = 32 * 1024
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)
    sweagent_config: str = "config/default.yaml"
    mini_config: str = "swebench.yaml"
    mini_model_class: str = "litellm"
    mini_environment_class: str = "docker"
    swebench_timeout_sec: Optional[int] = None
    shuffle: bool = True
    random_delay_multiplier: float = 0.3
    instance_ids_map: Dict[EvalLimitMode, List[str]] = field(default_factory=dict)


def _runtime(limit_samples_mode: Optional[str] = None):
    return SimpleNamespace(limit_samples_mode=limit_samples_mode)


def _terminal_task(**overrides):
    task = SimpleNamespace(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        score=FakeScore(),
        agentic_eval_config=FakeTerminalBenchConfig(),
        swebench_eval_config=None,
        limit_samples_map={EvalLimitMode.SMOKE_TEST: 5},
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def _swebench_task(**overrides):
    task = SimpleNamespace(
        task_name="swe_bench_verified",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        score=FakeScore(),
        agentic_eval_config=None,
        swebench_eval_config=FakeSWEbenchConfig(),
        limit_samples_map={EvalLimitMode.SMOKE_TEST: 5},
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def _server():
    return ServerConnection(
        base_url="http://127.0.0.1",
        service_port=8000,
        model="Qwen/Qwen3.6-27B",
    )


def _driver_context():
    return DriverContext(output_dir=Path("/tmp/out"), device="N150")


HARBOR_RESULT_FIXTURE = {
    "stats": {
        "evals": {
            "terminal_bench_2": {
                "metrics": [{"mean": 0.62, "std": 0.05}],
                "n_trials": 89,
                "reward_stats": {
                    "reward": {
                        "1.0": ["task-a", "task-b", "task-c"],
                        "0.0": ["task-d"],
                    }
                },
                "pass_at_k": {"1": 0.62},
            }
        }
    },
}


class TestAgenticParser:
    def test_parse_harbor_result_to_evals_block(self):
        parser = AgenticEvalParser(task_name="terminal_bench_2", score=FakeScore())
        block = parser.parse(HARBOR_RESULT_FIXTURE, device="N150")

        assert block.kind == "evals"
        assert block.task_type == "llm"
        assert block.targets["task_name"] == "terminal_bench_2"
        assert abs(block.data["accuracy"] - 0.62) < 1e-9
        assert block.data["n_trials"] == 89
        assert block.data["n_resolved"] == 3
        assert block.data["accuracy_check"] == ReportCheckTypes.PASS

    def test_failure_block_uses_failing_accuracy_check(self):
        parser = AgenticEvalParser(task_name="terminal_bench_2", score=FakeScore())
        block = parser.failure_block(return_code=7, device="N150")

        assert block.kind == "evals"
        assert block.data == {
            "success": False,
            "accuracy_check": 3,
            "subprocess_rc": 7,
        }

    def test_compute_accuracy_check_boundaries(self):
        score = FakeScore(gpu_reference_score=50.0, tolerance=0.05)

        assert compute_accuracy_check({"accuracy": 0.49}, score) == ReportCheckTypes.PASS
        assert compute_accuracy_check({"accuracy": 0.40}, score) == ReportCheckTypes.FAIL
        assert compute_accuracy_check({}, score) == ReportCheckTypes.NA
        assert compute_accuracy_check({"accuracy": 0.90}, None) == ReportCheckTypes.NA

    def test_compute_accuracy_check_preserves_percent_accuracy(self):
        score = FakeScore(gpu_reference_score=50.0, tolerance=0.05)

        assert compute_accuracy_check({"accuracy": 49.0}, score) == ReportCheckTypes.PASS


class TestAgenticDriverConfigMapping:
    def test_terminal_bench_config_uses_limit_mode_task_names_and_n_tasks(self):
        task = _terminal_task()
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["terminal-bench/caffe-cifar-10"]
        }

        cfg = build_terminal_bench_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime("ci-nightly"),
            n_tasks=resolve_n_tasks(task, _runtime("smoke-test")),
        )

        assert cfg.n_tasks == 5
        assert cfg.task_names == ["terminal-bench/caffe-cifar-10"]
        assert cfg.jobs_dir == Path("/tmp/out/eval_Qwen__Qwen3.6-27B/agentic")
        assert cfg.model_name == "openai/Qwen/Qwen3.6-27B"

    def test_swebench_config_uses_limit_mode_instance_ids_and_n_tasks(self):
        task = _swebench_task()
        task.swebench_eval_config.instance_ids_map = {
            EvalLimitMode.CI_NIGHTLY: ["django__django-11299"]
        }

        cfg = build_swebench_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime("ci-nightly"),
            n_tasks=resolve_n_tasks(task, _runtime("smoke-test")),
        )

        assert cfg.n_tasks == 5
        assert cfg.instance_ids == ["django__django-11299"]
        assert cfg.output_dir == Path(
            "/tmp/out/eval_Qwen__Qwen3.6-27B/agentic/swe_bench_verified"
        )
        assert cfg.model_name == "openai/Qwen/Qwen3.6-27B"


class TestAgenticLimitResolution:
    def test_fractional_agentic_limits_become_one_task(self):
        task = _terminal_task(limit_samples_map={EvalLimitMode.CI_COMMIT: 0.01})

        assert resolve_n_tasks(task, _runtime("ci-commit")) == 1

    def test_zero_limit_means_skip(self):
        task = _terminal_task(limit_samples_map={EvalLimitMode.CI_COMMIT: 0})

        assert resolve_n_tasks(task, _runtime("ci-commit")) == 0

    def test_default_task_names_and_instance_ids(self):
        terminal = _terminal_task()
        terminal.agentic_eval_config.task_names = ["default-task"]
        swe = _swebench_task()

        assert resolve_task_names(terminal, None) == ["default-task"]
        assert resolve_instance_ids(swe, None) == []


class TestSelectAgenticTasks:
    def _ctx_with_tasks(self, tasks):
        ctx = MagicMock()
        ctx.all_params.tasks = tasks
        ctx.model_spec.model_name = "test-llm"
        return ctx

    def test_returns_only_agentic_tasks(self):
        t1 = _terminal_task()
        t2 = _swebench_task()
        ctx = self._ctx_with_tasks([t1, t2])

        assert _select_agentic_tasks(ctx) == [t1, t2]

    def test_empty_task_list_returns_empty(self):
        assert _select_agentic_tasks(self._ctx_with_tasks([])) == []

    def test_mixed_tasks_raises(self):
        t_agentic = _terminal_task()
        t_other = _terminal_task(
            task_name="mmlu",
            workflow_venv_type=WorkflowVenvType.EVALS_META,
        )
        ctx = self._ctx_with_tasks([t_agentic, t_other])

        try:
            _select_agentic_tasks(ctx)
        except RuntimeError as exc:
            assert "non-agentic tasks" in str(exc)
        else:
            raise AssertionError("Expected mixed agentic task selection to fail")


class TestAgenticBridge:
    def test_bridge_delegates_to_driver_and_accepts_blocks(self):
        from test_module.llm_tests.agentic_eval_tests import run_llm_agentic_eval

        ctx = MagicMock()
        ctx.all_params.tasks = [_terminal_task()]
        ctx.model_spec.model_name = "test-llm"
        ctx.model_spec.hf_model_repo = "Qwen/Qwen3.6-27B"
        ctx.device.name = "N150"
        ctx.service_port = 8000
        ctx.output_path = "/tmp/out"
        ctx.runtime_config = _runtime("smoke-test")

        block = AgenticEvalParser(
            task_name="terminal_bench_2",
            score=FakeScore(),
        ).parse(HARBOR_RESULT_FIXTURE, device="N150")
        driver = MagicMock()
        driver.name = "terminal_bench"
        driver.run.return_value.return_code = 0
        driver.run.return_value.raw = HARBOR_RESULT_FIXTURE
        driver.parse.return_value = block

        with patch("test_module.llm_tests.agentic_eval_tests._require_openai_server"), patch(
            "test_module.llm_tests.agentic_eval_tests.make_agentic_driver",
            return_value=driver,
        ), patch("test_module.llm_tests.agentic_eval_tests.accept_blocks") as accept:
            blocks = run_llm_agentic_eval(ctx)

        assert blocks == [block]
        driver.run.assert_called_once()
        driver.parse.assert_called_once_with(HARBOR_RESULT_FIXTURE, device="N150")
        accept.assert_called_once()
