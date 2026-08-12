# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Tests for v2 agentic eval parser, drivers, and bridge helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field

import pytest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from llm_module import DriverContext, ServerConnection
from llm_module.drivers.agentic import (
    build_harbor_config,
    resolve_n_tasks,
    resolve_task_names,
)
from llm_module.agentic.harbor import _run_with_timeout, run as run_harbor
from llm_module.parsers.agentic import (
    AgenticEvalParser,
    compute_accuracy_check,
    extract_harbor_metrics,
)
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks
from workflows.workflow_types import EvalLimitMode, ReportCheckTypes, WorkflowVenvType


@dataclass
class FakeScore:
    published_score: float = 0.5
    published_score_ref: str = "https://example.com"
    gpu_reference_score: Optional[float] = 0.45
    tolerance: float = 0.05


@dataclass
class FakeHarborConfig:
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
    agent_setup_timeout_multiplier: Optional[float] = None
    quiet: bool = True
    yes: bool = True
    task_names_map: Dict[EvalLimitMode, List[str]] = field(default_factory=dict)
    agent_import_path: Optional[str] = None
    agent_env: Dict[str, str] = field(default_factory=dict)
    environment_env: Dict[str, str] = field(default_factory=dict)
    verifier_env: Dict[str, str] = field(default_factory=dict)
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)
    harbor_timeout_sec: Optional[float] = None


def _runtime(limit_samples_mode: Optional[str] = None):
    return SimpleNamespace(limit_samples_mode=limit_samples_mode)


def _harbor_task(**overrides):
    task = SimpleNamespace(
        task_name="terminal_bench_2",
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        score=FakeScore(),
        agentic_eval_config=FakeHarborConfig(),
        limit_samples_map={EvalLimitMode.SMOKE_TEST: 5},
    )
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


def _swebench_task(**overrides):
    """SWE-bench is just another Harbor eval: the registry dataset plus the
    mini-swe-agent agent, with instance ids as ordinary Harbor task names."""
    task = _harbor_task(
        task_name="swe_bench_verified",
        agentic_eval_config=FakeHarborConfig(
            dataset="swebench-verified",
            agent="mini-swe-agent",
            n_tasks=None,
            override_cpus=4,
            override_memory_mb=8 * 1024,
        ),
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
        assert block.title == "LLM Eval — terminal_bench_2"
        assert block.targets["task_name"] == "terminal_bench_2"
        assert block.targets["n_trials"] == 89
        assert block.targets["n_resolved"] == 3
        assert abs(block.data["score"] - 62.0) < 1e-9
        assert block.data["task_name"] == "terminal_bench_2"
        assert block.data["published_score"] == 0.5
        assert block.data["gpu_reference_score"] == 0.45
        assert abs(block.data["ratio_to_published"] - (62.0 / 0.5)) < 1e-9
        assert block.data["accuracy_check"] == ReportCheckTypes.PASS
        assert "success" not in block.data
        assert "accuracy" not in block.data

    def test_mean_seconds_per_task_from_harbor_timing(self):
        raw = {
            **HARBOR_RESULT_FIXTURE,
            "started_at": "2026-06-30T07:00:00.000000",
            "finished_at": "2026-06-30T08:29:00.000000",
            "n_total_trials": 89,
        }
        block = AgenticEvalParser(
            task_name="terminal_bench_2", score=FakeScore()
        ).parse(raw, device="N150")

        # 89 minutes (5340s) across 89 trials -> 60s/task.
        assert abs(block.data["mean_seconds_per_task"] - 60.0) < 1e-6

    def test_mean_seconds_per_task_falls_back_to_n_trials(self):
        raw = {
            **HARBOR_RESULT_FIXTURE,
            "started_at": "2026-06-30T07:00:00",
            "finished_at": "2026-06-30T07:01:29",
        }
        metrics = extract_harbor_metrics(raw)

        # 89 seconds across the summary's 89 n_trials -> 1s/task.
        assert abs(metrics["mean_seconds_per_task"] - 1.0) < 1e-6

    def test_mean_seconds_per_task_absent_without_timing(self):
        block = AgenticEvalParser(
            task_name="terminal_bench_2", score=FakeScore()
        ).parse(HARBOR_RESULT_FIXTURE, device="N150")

        assert block.data["mean_seconds_per_task"] is None

    def test_failure_block_uses_failing_accuracy_check(self):
        parser = AgenticEvalParser(task_name="terminal_bench_2", score=FakeScore())
        block = parser.failure_block(return_code=7, device="N150")

        assert block.kind == "evals"
        assert block.title == "LLM Eval — terminal_bench_2"
        assert block.data["task_name"] == "terminal_bench_2"
        assert block.data["score"] is None
        assert block.data["accuracy_check"] == ReportCheckTypes.FAIL
        assert block.data["success"] is False
        assert block.data["subprocess_rc"] == 7

    def test_compute_accuracy_check_boundaries(self):
        score = FakeScore(gpu_reference_score=50.0, tolerance=0.05)

        assert (
            compute_accuracy_check({"accuracy": 0.49}, score) == ReportCheckTypes.PASS
        )
        assert (
            compute_accuracy_check({"accuracy": 0.40}, score) == ReportCheckTypes.FAIL
        )
        assert compute_accuracy_check({}, score) == ReportCheckTypes.NA
        assert compute_accuracy_check({"accuracy": 0.90}, None) == ReportCheckTypes.NA

    def test_compute_accuracy_check_preserves_percent_accuracy(self):
        score = FakeScore(gpu_reference_score=50.0, tolerance=0.05)

        assert (
            compute_accuracy_check({"accuracy": 49.0}, score) == ReportCheckTypes.PASS
        )


class TestAgenticModeReference:
    """Under --ci-mode, agentic scoring compares against the task's CI-subset
    reference (mode_reference_scores) using the sample-count-aware check
    (n_trials = subset size) instead of the full gpu_reference_score."""

    def _score(self):
        from reference_config.evals.eval_config import EvalTaskScore, ModeReferenceScore

        return EvalTaskScore(
            published_score=None,
            published_score_ref="x",
            score_func=lambda *a, **k: 0.0,
            gpu_reference_score=64.80,
            gpu_reference_score_ref="full (500)",
            tolerance=0.05,
            mode_reference_scores={
                EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                    40.0, ref="ci-nightly 5 instances", tolerance=0.10
                )
            },
        )

    def test_subset_20pct_fails_full_but_passes_ci_sample_aware(self):
        # 5-instance subset: 1/5 (20%). threshold = floor(5*0.40*0.9) = 1.
        score = self._score()
        metrics = {"accuracy": 0.20, "n_trials": 5}
        assert compute_accuracy_check(metrics, score, None) == ReportCheckTypes.FAIL
        assert (
            compute_accuracy_check(metrics, score, EvalLimitMode.CI_NIGHTLY)
            == ReportCheckTypes.PASS
        )

    def test_subset_zero_fails_ci(self):
        score = self._score()
        metrics = {"accuracy": 0.0, "n_trials": 5}
        assert (
            compute_accuracy_check(metrics, score, EvalLimitMode.CI_NIGHTLY)
            == ReportCheckTypes.FAIL
        )

    def test_parser_block_labels_subset_reference(self):
        parser = AgenticEvalParser(
            task_name="swe_bench_verified",
            score=self._score(),
            limit_mode=EvalLimitMode.CI_NIGHTLY,
        )
        block = parser.parse(HARBOR_RESULT_FIXTURE, device="N150")

        assert block.targets["gpu_reference_score"] == 40.0
        assert "[CI_NIGHTLY subset]" in block.targets["gpu_reference_score_ref"]


class TestStandardEvalModeReference:
    """Standard lm-eval scoring (_score_one) honors the CI-subset reference."""

    def _gpqa_score(self):
        from reference_config.evals.eval_config import EvalTaskScore, ModeReferenceScore
        from reference_config.evals.eval_utils import score_task_single_key

        return EvalTaskScore(
            published_score=84.3,
            published_score_ref="x",
            score_func=score_task_single_key,
            gpu_reference_score=83.33,
            gpu_reference_score_ref="full (198)",
            tolerance=0.05,
            score_func_kwargs={"result_keys": ["exact_match,none"], "unit": "percent"},
            mode_reference_scores={
                EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
            },
        )

    def test_subset_score_fails_full_but_passes_ci_subset(self):
        from reference_config.evals.eval_config import resolve_eval_reference
        from test_module.llm_tests.llm_eval_tests import _score_one

        score = self._gpqa_score()
        task = SimpleNamespace(score=score, task_name="r1_gpqa_diamond")
        # 28/40 correct = 70%; full ref 83.33 FAIL, subset ref 72.5 PASS.
        results = {"r1_gpqa_diamond": {"exact_match,none": 0.70}}

        ref_full = resolve_eval_reference(score, None)
        _, _, _, ac_full = _score_one(
            task, results, "r1_gpqa_diamond", ref_full, n_total=40
        )
        assert ac_full == ReportCheckTypes.FAIL

        ref_ci = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
        s, _, _, ac_ci = _score_one(
            task, results, "r1_gpqa_diamond", ref_ci, n_total=40
        )
        assert abs(s - 70.0) < 1e-6
        assert ac_ci == ReportCheckTypes.PASS

    def test_collect_sample_counts_reads_effective(self, tmp_path):
        import json as _json
        from test_module.llm_tests.llm_eval_tests import collect_sample_counts

        f = tmp_path / "results_x.json"
        f.write_text(
            _json.dumps(
                {
                    "results": {"r1_gpqa_diamond": {"exact_match,none": 0.7}},
                    "n-samples": {
                        "r1_gpqa_diamond": {"original": 198, "effective": 40}
                    },
                }
            )
        )
        counts = collect_sample_counts([str(f)])
        assert counts == {"r1_gpqa_diamond": 40}


class TestAgenticDriverConfigMapping:
    def test_harbor_config_uses_limit_mode_task_names_and_n_tasks(self):
        task = _harbor_task()
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["terminal-bench/caffe-cifar-10"]
        }

        cfg = build_harbor_config(
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

    def test_harbor_config_preserves_prefixed_server_model_id(self):
        server = ServerConnection(
            base_url="http://127.0.0.1",
            service_port=8000,
            model="openai/gpt-oss-120b",
        )

        cfg = build_harbor_config(_harbor_task(), server, _driver_context())

        assert cfg.model_name == "openai/openai/gpt-oss-120b"

    def test_harbor_config_forwards_harbor_adapter_fields(self):
        task = _harbor_task()
        task.agentic_eval_config.agent_import_path = (
            "adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent"
        )
        task.agentic_eval_config.environment_env = {"TAU2_USER_MODEL": "openai/Qwen"}
        task.agentic_eval_config.verifier_env = {
            "TAU2_NL_ASSERTIONS_MODEL": "openai/Qwen"
        }

        cfg = build_harbor_config(
            task,
            _server(),
            _driver_context(),
        )

        assert (
            cfg.agent_import_path == "adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent"
        )
        assert cfg.environment_env == {"TAU2_USER_MODEL": "openai/Qwen"}
        assert cfg.verifier_env == {"TAU2_NL_ASSERTIONS_MODEL": "openai/Qwen"}

    def test_swebench_config_uses_limit_mode_task_names_and_n_tasks(self):
        # SWE-bench instance ids are plain Harbor task names in the
        # swebench-verified registry dataset, so the CI subset rides the same
        # task_names_map every other agentic eval uses.
        task = _swebench_task()
        task.agentic_eval_config.task_names_map = {
            EvalLimitMode.CI_NIGHTLY: ["django__django-11299"]
        }

        cfg = build_harbor_config(
            task,
            _server(),
            _driver_context(),
            runtime_config=_runtime("ci-nightly"),
            n_tasks=resolve_n_tasks(task, _runtime("smoke-test")),
        )

        assert cfg.n_tasks == 5
        assert cfg.dataset == "swebench-verified"
        assert cfg.agent == "mini-swe-agent"
        assert cfg.task_names == ["django__django-11299"]
        assert cfg.jobs_dir == Path("/tmp/out/eval_Qwen__Qwen3.6-27B/agentic")
        assert cfg.model_name == "openai/Qwen/Qwen3.6-27B"

    def test_harbor_config_forwards_agent_env(self):
        task = _harbor_task()
        task.agentic_eval_config.agent_env = {"OPENAI_BASE_URL": "http://alt/v1"}

        cfg = build_harbor_config(task, _server(), _driver_context())

        assert cfg.agent_env == {"OPENAI_BASE_URL": "http://alt/v1"}


class TestHarborHarness:
    def test_nonzero_return_code_does_not_require_result_file(self, tmp_path):
        task = _harbor_task()
        cfg = build_harbor_config(
            task,
            _server(),
            DriverContext(output_dir=tmp_path, device="N150"),
            n_tasks=1,
        )

        with patch("llm_module.agentic.harbor._run_with_timeout") as run_cmd:
            run_cmd.return_value = 17

            assert run_harbor(cfg) == 17

    def test_harbor_config_includes_adapter_and_env_overrides(self, tmp_path):
        task = _harbor_task()
        task.agentic_eval_config.agent_timeout_sec = None
        task.agentic_eval_config.agent_import_path = (
            "adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent"
        )
        task.agentic_eval_config.environment_env = {
            "TAU2_USER_MODEL": "openai/Qwen/Qwen3.6-27B"
        }
        task.agentic_eval_config.verifier_env = {
            "TAU2_NL_ASSERTIONS_MODEL": "openai/Qwen/Qwen3.6-27B"
        }
        cfg = build_harbor_config(
            task,
            _server(),
            DriverContext(output_dir=tmp_path, device="N150"),
            n_tasks=1,
        )

        with patch("llm_module.agentic.harbor._run_with_timeout") as run_cmd:
            run_cmd.return_value = 17

            assert run_harbor(cfg) == 17

        config_path = cfg.jobs_dir / f"{cfg.task_name}_harbor_config.json"
        harbor_config = json.loads(config_path.read_text())
        assert harbor_config["agents"][0]["import_path"] == (
            "adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent"
        )
        assert "name" not in harbor_config["agents"][0]
        assert harbor_config["environment"]["env"] == {
            "TAU2_USER_MODEL": "openai/Qwen/Qwen3.6-27B"
        }
        assert harbor_config["verifier"]["env"] == {
            "TAU2_NL_ASSERTIONS_MODEL": "openai/Qwen/Qwen3.6-27B"
        }
        run_cmd.assert_called_once()

    def test_environment_kwargs_force_the_config_file_path(self, tmp_path):
        """Cluster knobs have no CLI equivalent, so they must select --config.

        With every other config-file trigger cleared, kwargs alone have to flip
        run() off the flag-based command line -- otherwise the namespace and
        node selector are silently dropped and trials land in the wrong place
        (or on the wrong cluster).
        """
        task = _harbor_task()
        task.agentic_eval_config.agent_timeout_sec = None
        task.agentic_eval_config.environment_type = "kubernetes"
        task.agentic_eval_config.environment_kwargs = {
            "namespace": "harbor-kube-env",
            "image_mode": "prebuilt",
            "node_selector": {"tt-pool": "shield"},
        }
        cfg = build_harbor_config(
            task,
            _server(),
            DriverContext(output_dir=tmp_path, device="N150"),
            n_tasks=1,
        )

        # Nonzero so run() returns before _annotate_result_file, which would
        # otherwise demand a result.json no mocked harbor ever wrote.
        with patch("llm_module.agentic.harbor._run_with_timeout") as run_cmd:
            run_cmd.return_value = 17

            assert run_harbor(cfg) == 17

        cmd = run_cmd.call_args.args[0]
        assert "--config" in cmd

        config_path = cfg.jobs_dir / f"{cfg.task_name}_harbor_config.json"
        harbor_config = json.loads(config_path.read_text())
        assert harbor_config["environment"]["type"] == "kubernetes"
        assert harbor_config["environment"]["kwargs"] == {
            "namespace": "harbor-kube-env",
            "image_mode": "prebuilt",
            "node_selector": {"tt-pool": "shield"},
        }

    def test_harbor_timeout_is_passed_to_the_subprocess(self, tmp_path):
        task = _harbor_task()
        task.agentic_eval_config.harbor_timeout_sec = 7200.0
        cfg = build_harbor_config(
            task,
            _server(),
            DriverContext(output_dir=tmp_path, device="N150"),
            n_tasks=1,
        )

        with patch("llm_module.agentic.harbor._run_with_timeout") as run_cmd:
            run_cmd.return_value = 17

            assert run_harbor(cfg) == 17

        assert run_cmd.call_args.args[1] == 7200.0

    def test_timed_out_harbor_run_returns_124(self, tmp_path):
        """A stuck harbor otherwise hangs to the outer job cap (70h in CI)."""
        task = _harbor_task()
        task.agentic_eval_config.harbor_timeout_sec = 1.0
        cfg = build_harbor_config(
            task,
            _server(),
            DriverContext(output_dir=tmp_path, device="N150"),
            n_tasks=1,
        )

        with patch("llm_module.agentic.harbor._run_with_timeout") as run_cmd:
            run_cmd.side_effect = subprocess.TimeoutExpired(cmd="harbor", timeout=1.0)

            assert run_harbor(cfg) == 124

    def test_overrunning_harbor_gets_sigterm_before_sigkill(self, tmp_path):
        """Harbor must get a chance to delete its sandboxes on timeout.

        ``subprocess.run(timeout=...)`` escalates straight to SIGKILL, which
        harbor cannot trap, silently leaking every live sandbox -- for the
        Kubernetes backend, orphaned pods that hold cluster capacity until
        somebody notices. Drive a real child that traps SIGTERM and records
        that it got to clean up.
        """
        marker = tmp_path / "cleanup-ran"
        child = tmp_path / "traps_sigterm.py"
        child.write_text(
            "import signal, sys, time\n"
            "def on_term(sig, frame):\n"
            f"    open({str(marker)!r}, 'w').write('cleaned up')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, on_term)\n"
            "while True: time.sleep(0.05)\n"
        )

        with pytest.raises(subprocess.TimeoutExpired):
            _run_with_timeout([sys.executable, str(child)], 0.5)

        assert marker.exists(), "harbor was SIGKILLed before it could clean up"


class TestAgenticLimitResolution:
    def test_fractional_agentic_limits_become_one_task(self):
        task = _harbor_task(limit_samples_map={EvalLimitMode.CI_COMMIT: 0.01})

        assert resolve_n_tasks(task, _runtime("ci-commit")) == 1

    def test_zero_limit_means_skip(self):
        task = _harbor_task(limit_samples_map={EvalLimitMode.CI_COMMIT: 0})

        assert resolve_n_tasks(task, _runtime("ci-commit")) == 0

    def test_default_task_names_without_limit_mode(self):
        terminal = _harbor_task()
        terminal.agentic_eval_config.task_names = ["default-task"]
        swe = _swebench_task()

        assert resolve_task_names(terminal, None) == ["default-task"]
        # No task_names and no limit mode -> the whole dataset.
        assert resolve_task_names(swe, None) == []


class TestSelectAgenticTasks:
    def _ctx_with_tasks(self, tasks):
        ctx = MagicMock()
        ctx.all_params.tasks = tasks
        ctx.model_spec.model_name = "test-llm"
        return ctx

    def test_returns_only_agentic_tasks(self):
        t1 = _harbor_task()
        t2 = _swebench_task()
        ctx = self._ctx_with_tasks([t1, t2])

        assert _select_agentic_tasks(ctx) == [t1, t2]

    def test_empty_task_list_returns_empty(self):
        assert _select_agentic_tasks(self._ctx_with_tasks([])) == []

    def test_mixed_tasks_returns_only_agentic(self):
        # Mixed configs (e.g. GPQA + agentic) are normal; the agentic runner
        # selects the agentic tasks and skips the standard ones (which run under
        # --workflow evals). --eval-samples can't be used under --ci-mode, so a
        # hard failure here would block agentic CI runs.
        t_agentic = _harbor_task()
        t_other = _harbor_task(
            task_name="mmlu",
            workflow_venv_type=WorkflowVenvType.EVALS_META,
        )
        ctx = self._ctx_with_tasks([t_agentic, t_other])

        assert _select_agentic_tasks(ctx) == [t_agentic]


class TestAgenticBridge:
    def test_bridge_delegates_to_driver_and_accepts_blocks(self):
        from test_module.llm_tests.agentic_eval_tests import run_llm_agentic_eval

        ctx = MagicMock()
        ctx.all_params.tasks = [_harbor_task()]
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

        with patch(
            "test_module.llm_tests.agentic_eval_tests._require_openai_server"
        ), patch(
            "test_module.llm_tests.agentic_eval_tests.make_agentic_driver",
            return_value=driver,
        ), patch("test_module.llm_tests.agentic_eval_tests.accept_blocks") as accept:
            blocks = run_llm_agentic_eval(ctx)

        assert blocks == [block]
        driver.run.assert_called_once()
        driver.parse.assert_called_once_with(HARBOR_RESULT_FIXTURE, device="N150")
        accept.assert_called_once()
