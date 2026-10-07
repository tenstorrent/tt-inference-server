#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Quetzal candidate EvalConfigs carry cited published scores.

A task with no published score and no GPU reference grades NA, so a model whose
tasks are all NA can never PASS its Shield evals. These tests pin which tasks
are gradable and that every published score cites a URL.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import _eval_config_map
from reference_config.evals.eval_utils import (
    score_multilevel_keys_mean,
    score_task_single_key,
)

OLL_RESULTS = "https://huggingface.co/datasets/open-llm-leaderboard/results/blob/main/"

# hf_model_repo -> {task_name: published_score or None (task grades NA)}
EXPECTED = {
    "tiiuae/Falcon3-1B-Instruct": {
        "ifeval": 50.46,
        "gpqa_diamond_generative_n_shot": None,
    },
    "tiiuae/Falcon3-3B-Instruct": {
        "ifeval": 65.43,
        "gpqa_diamond_generative_n_shot": None,
    },
    "tiiuae/Falcon3-10B-Instruct": {
        "ifeval": 74.68,
        "gpqa_diamond_generative_n_shot": None,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "r1_aime24": 50.4,
        "r1_gpqa_diamond": 49.0,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "leaderboard_ifeval": 40.85,
        "leaderboard_math_hard": 8.13,
        "mmlu_pro": 35.91,
    },
    "deepcogito/cogito-v1-preview-llama-8B": {"mmlu_pro": 57.77},
    "Qwen/Qwen3-4B-Instruct-2507": {"r1_gpqa_diamond": 62.0, "mmlu_pro": 69.6},
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "leaderboard_ifeval": 40.3,
        "leaderboard_math_hard": 5.22,
        "mmlu_pro": 31.38,
    },
    "01-ai/Yi-1.5-9B-Chat": {
        "leaderboard_ifeval": 55.08,
        "leaderboard_math_hard": 19.85,
        "mmlu_pro": 39.75,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "leaderboard_ifeval": 79.5,
        "leaderboard_math_hard": 60.92,
        "gpqa_diamond_generative_n_shot": None,
        "mmlu_pro": 69.0,
    },
    "Qwen/Qwen2.5-Math-7B": {
        "leaderboard_ifeval": 19.22,
        "leaderboard_math_hard": 27.67,
        "mmlu_pro": 27.18,
    },
    "deepseek-ai/deepseek-coder-1.3b-instruct": {
        "mbpp_instruct": 49.4,
        "humaneval_instruct": 65.2,
    },
    "Qwen/Qwen3-0.6B": {"r1_gpqa_diamond": 27.9, "mmlu_pro": 24.74},
    "Qwen/Qwen3-1.7B": {"r1_gpqa_diamond": 40.1, "mmlu_pro": 36.76},
    "Qwen/Qwen3-4B": {"r1_gpqa_diamond": 55.9, "mmlu_pro": 50.58},
    "Qwen/Qwen3-14B": {"r1_gpqa_diamond": 64.0, "mmlu_pro": 61.03},
    "Qwen/Qwen2.5-14B-Instruct-1M": {
        "leaderboard_ifeval": 84.3,
        "leaderboard_math_hard": 50.72,
        "gpqa_diamond_generative_n_shot": None,
        "mmlu_pro": 63.3,
    },
    "ALLaM-AI/ALLaM-7B-Instruct-preview": {"ifeval": 38.08, "mmlu_pro": 30.4},
    "Qwen/Qwen3.5-27B": {"r1_gpqa_diamond": 85.5},
}

# No authoritative source exists for any task, so every task stays NA.
ALL_NA = ["deepseek-ai/deepseek-math-7b-instruct"]


@pytest.mark.parametrize("repo", ALL_NA)
def test_unsourced_configs_stay_na(repo):
    for task in _eval_config_map[repo].tasks:
        assert task.score.published_score is None
        assert task.score.published_score_ref is None


@pytest.mark.parametrize("repo", sorted(EXPECTED))
def test_tasks_and_published_scores(repo):
    tasks = {t.task_name: t for t in _eval_config_map[repo].tasks}

    assert {name: t.score.published_score for name, t in tasks.items()} == EXPECTED[
        repo
    ]
    assert any(t.score.published_score for t in tasks.values()), (
        f"{repo}: every task grades NA"
    )
    for name, task in tasks.items():
        if task.score.published_score is None:
            assert task.score.published_score_ref is None
        else:
            assert task.score.published_score > 0
            assert task.score.published_score_ref.startswith("https://"), name


@pytest.mark.parametrize("repo", sorted(EXPECTED))
def test_leaderboard_scores_use_raw_metric_keys(repo):
    """OLL citations must use the raw results JSON, scored with the same key."""
    for task in _eval_config_map[repo].tasks:
        score = task.score
        ref = score.published_score_ref or ""
        if not ref.startswith(OLL_RESULTS):
            continue
        # The leaderboard's displayed (normalized) scores live in the contents
        # dataset; only the per-model results JSON holds raw accuracies.
        assert f"{repo}/results_2025-02-13T18-27-04.338360.json" in ref
        keys = score.score_func_kwargs["result_keys"]
        if task.task_name in ("leaderboard_ifeval", "ifeval"):
            assert score.score_func is score_task_single_key
            assert keys[0] == "prompt_level_strict_acc,none"
            assert "prompt_level_strict_acc,none" in ref
        elif task.task_name == "leaderboard_math_hard":
            assert score.score_func is score_multilevel_keys_mean
            assert len(keys) == 7
            assert "mean of results.leaderboard_math_*_hard" in ref
        elif task.task_name == "mmlu_pro":
            assert keys == ["exact_match,custom-extract"]
            assert "leaderboard_mmlu_pro.acc,none" in ref
        else:
            pytest.fail(f"{repo}: unexpected OLL-cited task {task.task_name}")
