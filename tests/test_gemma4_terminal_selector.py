"""The dedicated probe catalog must reach the production agentic selector."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from reference_config.evals.eval_config import EVAL_CONFIGS
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks


def _context(tasks, selection=None):
    return SimpleNamespace(
        all_params=SimpleNamespace(tasks=tasks),
        model_spec=SimpleNamespace(
            impl=SimpleNamespace(impl_id="gemma4_fresh_autoport")
        ),
        runtime_config=SimpleNamespace(agentic_benchmark=selection),
    )


def test_real_catalog_reaches_production_selector_with_frozen_probe():
    tasks = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    selected = _select_agentic_tasks(_context(tasks))
    assert [task.task_name for task in selected] == ["terminal_bench_2"]
    assert selected == tasks
    config = selected[0].agentic_eval_config
    cohort = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reference_config/evals/gemma4_cohorts.json"
        ).read_text()
    )["terminal"]
    assert cohort["tasks"] == ["build-cython-ext"]
    assert (
        config.dataset_ref
        == cohort["ref"]
        == ("sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b")
    )
    assert config.dataset == "gemma4-isolated-terminal"
    assert config.n_tasks == 1
    assert config.n_concurrent_trials == 32
    assert config.agent_timeout_sec == 18000
    assert config.llm_timeout_sec == 3600
    assert config.agent_kwargs["llm_kwargs"]["timeout"] == 3600
    assert config.agent_kwargs["llm_kwargs"]["max_tokens"] == 65536
    assert config.agent_kwargs["model_info"]["max_output_tokens"] == 65536
    assert config.offline_task_prerequisites and config.offline_apt_fixtures
    assert config.collect_preflight_evidence
    assert config.preflight_min_free_bytes == 512 * 1024**3


@pytest.mark.parametrize("name", ["gemma4_preflight_unscored", "swe_bench_verified"])
def test_probe_selector_rejects_unintended_tasks(name):
    task = replace(EVAL_CONFIGS["google/gemma-4-31B-it"].tasks[0], task_name=name)
    assert _select_agentic_tasks(_context([task])) == []


def test_probe_selector_still_honors_explicit_benchmark_filter():
    tasks = EVAL_CONFIGS["google/gemma-4-31B-it"].tasks
    assert _select_agentic_tasks(_context(tasks, "terminal_bench_2")) == tasks
    with pytest.raises(RuntimeError, match="matched no EVALS_AGENTIC"):
        _select_agentic_tasks(_context(tasks, "swe_bench_verified"))
