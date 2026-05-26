# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for agentic_eval_tests runner internals.

These tests cover the pure-Python helper functions (config building,
result parsing, accuracy mapping, task selection) and do NOT require
the EVALS_AGENTIC venv to be active — the harness imports are mocked
at the module level so tests run in the standard v2 venv.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub the v2 agentic harness modules before importing the runner so we
# don't need harbor / swe-agent installed in the test venv.
#
# report.py is pure stdlib — load the real module via importlib so tests
# exercise the actual parsing logic rather than a mock.
# ---------------------------------------------------------------------------

_V2_ROOT = Path(__file__).resolve().parents[3]

# Load the real report module directly from disk (no heavy deps needed).
_report_spec = _importlib_util.spec_from_file_location(
    "llm_module.agentic.report",
    _V2_ROOT / "llm_module" / "agentic" / "report.py",
)
_real_report_module = _importlib_util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(_real_report_module)

_stub_swebench = SimpleNamespace(
    SWEbenchRunConfig=None,
    run=lambda cfg: 0,
)
_stub_terminal_bench = SimpleNamespace(
    TerminalBenchRunConfig=None,
    run=lambda cfg: 0,
)

sys.modules.setdefault("llm_module", MagicMock())
sys.modules.setdefault("llm_module.agentic", MagicMock())
sys.modules["llm_module.agentic.report"] = _real_report_module
sys.modules["llm_module.agentic.swebench"] = _stub_swebench
sys.modules["llm_module.agentic.terminal_bench"] = _stub_terminal_bench


# Now we can safely import the runner helpers. Import only the pure helpers
# that don't trigger the harness imports at module scope.
from test_module.llm_tests.agentic_eval_tests import (  # noqa: E402
    _agentic_output_dir,
    _compute_accuracy_check,
    _parse_harbor_result,
    _select_agentic_tasks,
)


# ---------------------------------------------------------------------------
# Minimal fakes
# ---------------------------------------------------------------------------

def _fake_ctx(model_id="Qwen__Qwen3.6-27B", service_port=8000, output_path="/tmp/out"):
    ctx = MagicMock()
    ctx.model_spec.model_id = model_id
    ctx.model_spec.hf_model_repo = "Qwen/Qwen3.6-27B"
    ctx.service_port = service_port
    ctx.output_path = output_path
    return ctx


def _make_eval_task(task_name, venv_type, has_swebench=False, has_terminal=False):
    from workflows.workflow_types import WorkflowVenvType

    task = MagicMock()
    task.task_name = task_name
    task.workflow_venv_type = venv_type
    task.swebench_eval_config = MagicMock() if has_swebench else None
    task.agentic_eval_config = MagicMock() if has_terminal else None
    task.score = MagicMock()
    task.score.tolerance = 0.05
    task.score.published_score = 0.5
    task.score.gpu_reference_score = 0.45
    task.score.published_score_ref = "https://example.com"
    return task


# ---------------------------------------------------------------------------
# _agentic_output_dir
# ---------------------------------------------------------------------------

class TestAgenticOutputDir:
    def test_path_structure(self):
        ctx = _fake_ctx(model_id="Qwen/Qwen3.6-27B", output_path="/out")
        task = MagicMock()
        task.task_name = "terminal_bench_2"
        result = _agentic_output_dir(ctx, task)
        assert result == Path("/out/eval_Qwen__Qwen3.6-27B/agentic/terminal_bench_2")

    def test_slash_replaced(self):
        ctx = _fake_ctx(model_id="org/model-name", output_path="/out")
        task = MagicMock()
        task.task_name = "swe_bench"
        result = _agentic_output_dir(ctx, task)
        assert "__" in str(result)
        assert "/" not in result.name


# ---------------------------------------------------------------------------
# _parse_harbor_result
# ---------------------------------------------------------------------------

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
    "trial_results": [
        {"verifier_result": {"rewards": {"reward": 1.0}}},
        {"verifier_result": {"rewards": {"reward": 0.0}}},
    ],
}


class TestParseHarborResult:
    def test_happy_path(self):
        metrics = _parse_harbor_result(HARBOR_RESULT_FIXTURE)
        assert "accuracy" in metrics
        assert metrics["accuracy"] == pytest.approx(0.62)
        assert "n_trials" in metrics

    def test_empty_dict_does_not_raise(self):
        metrics = _parse_harbor_result({})
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# _compute_accuracy_check
# ---------------------------------------------------------------------------

class TestComputeAccuracyCheck:
    def _task_with_score(self, published=0.5, gpu_ref=0.45, tol=0.05):
        task = MagicMock()
        task.score.published_score = published
        task.score.gpu_reference_score = gpu_ref
        task.score.tolerance = tol
        return task

    def test_pass_within_tolerance(self):
        task = self._task_with_score(gpu_ref=0.5, tol=0.05)
        assert _compute_accuracy_check({"accuracy": 0.49}, task) == 1

    def test_marginal_between_one_and_two_tol(self):
        task = self._task_with_score(gpu_ref=0.5, tol=0.05)
        # 0.5 * (1 - 0.05) = 0.475; 0.5 * (1 - 0.10) = 0.45
        assert _compute_accuracy_check({"accuracy": 0.46}, task) == 2

    def test_fail_below_two_tol(self):
        task = self._task_with_score(gpu_ref=0.5, tol=0.05)
        assert _compute_accuracy_check({"accuracy": 0.40}, task) == 3

    def test_no_accuracy_returns_marginal(self):
        task = self._task_with_score()
        assert _compute_accuracy_check({}, task) == 2

    def test_no_score_returns_marginal(self):
        task = MagicMock()
        task.score = None
        assert _compute_accuracy_check({"accuracy": 0.9}, task) == 2


# ---------------------------------------------------------------------------
# _select_agentic_tasks
# ---------------------------------------------------------------------------

class TestSelectAgenticTasks:
    def _ctx_with_tasks(self, tasks):
        ctx = MagicMock()
        ctx.all_params.tasks = tasks
        ctx.model_spec.model_name = "test-llm"
        return ctx

    def test_returns_only_agentic_tasks(self):
        from workflows.workflow_types import WorkflowVenvType

        t1 = _make_eval_task("tb2", WorkflowVenvType.EVALS_AGENTIC)
        t2 = _make_eval_task("swe", WorkflowVenvType.EVALS_AGENTIC)
        ctx = self._ctx_with_tasks([t1, t2])
        result = _select_agentic_tasks(ctx)
        assert result == [t1, t2]

    def test_empty_task_list_returns_empty(self):
        ctx = self._ctx_with_tasks([])
        assert _select_agentic_tasks(ctx) == []

    def test_mixed_tasks_raises(self):
        from workflows.workflow_types import WorkflowVenvType

        t_agentic = _make_eval_task("tb2", WorkflowVenvType.EVALS_AGENTIC)
        t_other = _make_eval_task("mmlu", WorkflowVenvType.EVALS_META)
        ctx = self._ctx_with_tasks([t_agentic, t_other])
        with pytest.raises(RuntimeError, match="non-agentic tasks"):
            _select_agentic_tasks(ctx)

    def test_all_non_agentic_returns_empty(self):
        from workflows.workflow_types import WorkflowVenvType

        t = _make_eval_task("mmlu", WorkflowVenvType.EVALS_META)
        ctx = self._ctx_with_tasks([t])
        result = _select_agentic_tasks(ctx)
        assert result == []
