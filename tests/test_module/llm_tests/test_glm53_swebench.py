# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from llm_module.agentic.harbor import _get_agent_kwargs
from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers.agentic import build_harbor_config, resolve_n_tasks
from reference_config.evals.eval_config import _eval_config_map
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks


def selected(selection):
    return _select_agentic_tasks(
        SimpleNamespace(
            all_params=_eval_config_map["zai-org/GLM-5.3"],
            runtime_config=SimpleNamespace(agentic_benchmark=selection),
        )
    )


@pytest.mark.parametrize("selection", [None, "", "all", "  "])
def test_default_campaign_still_has_only_terminal_and_banking(selection):
    assert [t.task_name for t in selected(selection)] == [
        "terminal_bench_2_1",
        "tau3_bench_banking",
    ]


@pytest.mark.parametrize("selection", ["swebench", "swe_bench_verified"])
def test_explicit_swe_selection_has_no_other_evals(selection):
    assert [t.task_name for t in selected(selection)] == ["swe_bench_verified"]


def test_other_catalog_tasks_remain_enabled_by_default():
    opt_in = [
        (c.hf_model_repo, t.task_name)
        for c in _eval_config_map.values()
        for t in c.tasks
        if t.requires_explicit_selection
    ]
    assert opt_in == [("zai-org/GLM-5.3", "swe_bench_verified")]


def test_swe_smoke_uses_same_agent_with_one_fixed_task(tmp_path):
    (task,) = selected("swebench")
    runtime = SimpleNamespace(limit_samples_mode="smoke-test")
    config = build_harbor_config(
        task,
        ServerConnection("http://example.test", 8000, "zai-org/GLM-5.3"),
        DriverContext(output_dir=tmp_path, device="super_cluster"),
        runtime_config=runtime,
        n_tasks=resolve_n_tasks(task, runtime),
    )
    assert config.task_names == ["pytest-dev__pytest-5262"]
    assert config.n_tasks == 1
    assert config.dataset == "swebench-verified"
    assert config.model_name == "openai/zai-org/GLM-5.3"
    assert config.n_concurrent_trials == 8 and config.n_attempts == 1
    kwargs = _get_agent_kwargs(config)
    assert "reasoning_effort" not in kwargs  # Preserve the chat API route.
    assert kwargs["config"]["model"]["model_kwargs"]["timeout"] == 600
    assert task.score is None  # No score borrowed from another model.
    assert (
        resolve_n_tasks(task) is None
    )  # Use the complete dataset, as other models do.


def test_swe_reuses_existing_glm52_recipe():
    (task,) = selected("swebench")
    baseline = next(
        t
        for t in _eval_config_map["zai-org/GLM-5.2"].tasks
        if t.task_name == "swe_bench_verified"
    ).agentic_eval_config
    config = task.agentic_eval_config
    for field in (
        "dataset",
        "agent",
        "n_tasks",
        "n_attempts",
        "agent_timeout_sec",
        "agent_kwargs",
        "agent_env",
        "environment_env",
        "verifier_env",
    ):
        assert getattr(config, field) == getattr(baseline, field), field


def test_swe_and_existing_alias_can_be_selected_together():
    assert [t.task_name for t in selected("tb2.1,swebench")] == [
        "terminal_bench_2_1",
        "swe_bench_verified",
    ]
