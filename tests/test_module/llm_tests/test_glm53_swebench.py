# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest

from llm_module.parsers.agentic import AgenticEvalParser
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


def test_swe_reuses_existing_glm52_recipe():
    (task,) = selected("swebench")
    baseline = next(
        t
        for t in _eval_config_map["zai-org/GLM-5.2"].tasks
        if t.task_name == "swe_bench_verified"
    )
    assert task.agentic_eval_config == replace(
        baseline.agentic_eval_config, n_concurrent_trials=8
    )
    assert task.limit_samples_map == baseline.limit_samples_map


def test_swe_report_uses_published_score_without_a_gpu_reference():
    (task,) = selected("swebench")
    block = AgenticEvalParser(task_name=task.task_name, score=task.score).parse({})
    assert block.targets["published_score"] == 95.4
    assert (
        block.targets["published_score_ref"]
        == "https://www.vals.ai/benchmarks/swebench"
    )
    assert block.targets["gpu_reference_score"] is None


def test_swe_and_existing_alias_can_be_selected_together():
    assert [t.task_name for t in selected("tb2.1,swebench")] == [
        "terminal_bench_2_1",
        "swe_bench_verified",
    ]
