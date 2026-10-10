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
    "Qwen/Qwen3.6-35B-A3B": {
        "terminal_bench_2": 51.5,
    },
    "tiiuae/Falcon3-1B-Instruct": {
        "leaderboard_ifeval": 50.46,
        "gpqa_diamond_generative_n_shot": None,
    },
    "tiiuae/Falcon3-3B-Instruct": {
        "leaderboard_ifeval": 65.43,
        "gpqa_diamond_generative_n_shot": None,
    },
    "tiiuae/Falcon3-10B-Instruct": {
        "leaderboard_ifeval": 74.68,
        "gpqa_diamond_generative_n_shot": None,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "r1_aime24": 50.4,
        "r1_gpqa_diamond": 49.0,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "r1_aime24": 72.6,
        "r1_gpqa_diamond": 62.1,
    },
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "leaderboard_ifeval": 40.85,
        "leaderboard_math_hard": 8.13,
    },
    "deepcogito/cogito-v1-preview-llama-8B": {"mmlu_pro": 57.77},
    "Qwen/Qwen3-4B-Instruct-2507": {"r1_gpqa_diamond": 62.0, "mmlu_pro": 69.6},
    "upstage/SOLAR-10.7B-Instruct-v1.0": {
        "leaderboard_ifeval": 40.3,
        "leaderboard_math_hard": 5.22,
    },
    "01-ai/Yi-1.5-9B-Chat": {
        "leaderboard_ifeval": 55.08,
        "leaderboard_math_hard": 19.85,
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
    },
    "Qwen/Qwen2.5-Coder-7B": {
        "leaderboard_ifeval": 27.91,
        "leaderboard_math_hard": 16.7,
    },
    "Qwen/Qwen2.5-Coder-7B-Instruct": {
        "mbpp_instruct": None,
        "humaneval_instruct": 88.4,
    },
    "deepseek-ai/deepseek-coder-1.3b-instruct": {
        "mbpp_instruct": 49.4,
        "humaneval_instruct": 65.2,
    },
    # mmlu_pro NA: the report's MMLU-Pro is the -Base checkpoint, untemplated.
    "Qwen/Qwen3-0.6B": {"r1_gpqa_diamond": 27.9, "mmlu_pro": None},
    "Qwen/Qwen3-1.7B": {"r1_gpqa_diamond": 40.1, "mmlu_pro": None},
    "Qwen/Qwen3-4B": {"r1_gpqa_diamond": 55.9, "mmlu_pro": None},
    "Qwen/Qwen3-14B": {"r1_gpqa_diamond": 64.0, "mmlu_pro": None},
    "Qwen/Qwen2.5-14B-Instruct-1M": {
        "leaderboard_ifeval": 84.3,
        "leaderboard_math_hard": 50.72,
        "gpqa_diamond_generative_n_shot": None,
        "mmlu_pro": 63.3,
    },
    "ALLaM-AI/ALLaM-7B-Instruct-preview": {"ifeval": 38.08, "mmlu_pro": 30.4},
    "Qwen/Qwen3.5-27B": {"r1_gpqa_diamond": 85.5},
    # wave18 Quetzal candidates (Open LLM Leaderboard v2, model card or paper citations).
    "NousResearch/Hermes-3-Llama-3.2-3B": {
        "leaderboard_ifeval": 31.05,
        "leaderboard_math_hard": 3.8,
    },
    "01-ai/Yi-1.5-6B-Chat": {
        "leaderboard_ifeval": 45.47,
        "leaderboard_math_hard": 14.43,
    },
    "Qwen/Qwen2-7B-Instruct": {
        "leaderboard_ifeval": 52.31,
        "leaderboard_math_hard": 24.77,
    },
    "Qwen/Qwen2.5-Math-7B-Instruct": {
        "leaderboard_ifeval": 20.7,
        "leaderboard_math_hard": 56.13,
    },
    "allenai/Llama-3.1-Tulu-3-8B": {
        "leaderboard_ifeval": 79.48,
        "leaderboard_math_hard": 18.84,
    },
    "arcee-ai/Llama-3.1-SuperNova-Lite": {
        "leaderboard_ifeval": 76.89,
        "leaderboard_math_hard": 16.46,
    },
    "01-ai/Yi-1.5-9B": {
        "leaderboard_ifeval": 23.11,
        "leaderboard_math_hard": 10.27,
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "leaderboard_ifeval": 58.78,
        "leaderboard_math_hard": 11.29,
    },
    "Qwen/Qwen2.5-Coder-14B": {
        "leaderboard_ifeval": 27.73,
        "leaderboard_math_hard": 20.43,
    },
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "leaderboard_ifeval": 65.25,
        "leaderboard_math_hard": 30.51,
    },
    "Qwen/Qwen2.5-32B": {
        "leaderboard_ifeval": 35.49,
        "leaderboard_math_hard": 33.25,
    },
    "Qwen/Qwen2.5-Coder-32B": {
        "leaderboard_ifeval": 39.19,
        "leaderboard_math_hard": 28.12,
    },
    "01-ai/Yi-1.5-34B-Chat": {
        "leaderboard_ifeval": 55.27,
        "leaderboard_math_hard": 24.74,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "r1_aime24": 55.5,
        "r1_gpqa_diamond": 49.1,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "r1_aime24": 69.7,
        "r1_gpqa_diamond": 59.1,
    },
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "mbpp_instruct": 65.4,
        "humaneval_instruct": 78.6,
    },
    "Qwen/Qwen3-4B-Thinking-2507": {"r1_gpqa_diamond": 65.8},
}

# Whether each cited Open LLM Leaderboard v2 run applied a chat template: the
# results JSON's ``chat_template`` is non-null (True) or null (False). Those runs
# used fewshot_as_multiturn with the template and no system_instruction, which is
# what the pinned harness does under --apply_chat_template, so matching this flag
# reproduces the reference prompts.
OLL_CHAT_TEMPLATE = {
    "01-ai/Yi-1.5-9B-Chat": True,
    "arcee-ai/Arcee-Spark": True,
    "EleutherAI/pythia-160m": False,
    "EleutherAI/pythia-1b": False,
    "EleutherAI/pythia-410m": False,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": True,
    "huggyllama/llama-7b": False,
    "meta-llama/Llama-3.1-8B": False,
    "meta-llama/Llama-3.2-1B": False,
    "meta-llama/Llama-3.2-3B": False,
    "meta-llama/Meta-Llama-3-8B-Instruct": False,
    "microsoft/phi-1_5": False,
    "microsoft/phi-1": False,
    "Qwen/Qwen1.5-0.5B-Chat": True,
    "Qwen/Qwen1.5-0.5B": False,
    "Qwen/Qwen2-7B": False,
    "Qwen/Qwen2.5-14B-Instruct-1M": True,
    "Qwen/Qwen2.5-32B-Instruct": True,
    "Qwen/Qwen2.5-7B": False,
    "Qwen/Qwen2.5-Coder-7B": True,
    "Qwen/Qwen2.5-Math-7B": True,
    "stabilityai/stablelm-2-1_6b": False,
    "tiiuae/Falcon3-10B-Base": False,
    "tiiuae/Falcon3-10B-Instruct": True,
    "tiiuae/Falcon3-1B-Base": False,
    "tiiuae/Falcon3-1B-Instruct": True,
    "tiiuae/Falcon3-3B-Base": False,
    "tiiuae/Falcon3-3B-Instruct": True,
    "tiiuae/Falcon3-7B-Base": False,
    "TinyLlama/TinyLlama_v1.1": False,
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": False,
    "upstage/SOLAR-10.7B-Instruct-v1.0": True,
    # wave18 Quetzal candidates.
    "01-ai/Yi-1.5-34B-Chat": True,
    "01-ai/Yi-1.5-6B-Chat": True,
    "01-ai/Yi-1.5-9B": False,
    "allenai/Llama-3.1-Tulu-3-8B": True,
    "arcee-ai/Llama-3.1-SuperNova-Lite": True,
    "mistralai/Mistral-Nemo-Instruct-2407": True,
    "NousResearch/Hermes-3-Llama-3.2-3B": True,
    "Qwen/Qwen2-7B-Instruct": True,
    "Qwen/Qwen2.5-32B": False,
    "Qwen/Qwen2.5-Coder-14B": True,
    "Qwen/Qwen2.5-Coder-14B-Instruct": True,
    "Qwen/Qwen2.5-Coder-32B": False,
    "Qwen/Qwen2.5-Math-7B-Instruct": True,
}

# Leaderboard tasks TTIS runs as-is; a citation of any other leaderboard metric
# (notably log-likelihood leaderboard_mmlu_pro) has no generative TTIS twin.
OLL_TASKS = ("leaderboard_ifeval", "leaderboard_math_hard")

# No authoritative source exists for any task, so every task stays NA.
ALL_NA = [
    "deepseek-ai/deepseek-math-7b-instruct",
    # QB2 business set: card scores are not like-for-like, so every task stays NA
    # until a GPU reference is measured.
    "meta-models/Muse-Glimmer-30B",
    "mistralai/Mistral-Small-4-119B-2603",
    "IFM/K2-Horizon-MoVA-36B-A4B",
]


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
        if task.task_name == "leaderboard_ifeval":
            assert score.score_func is score_task_single_key
            assert keys[0] == "prompt_level_strict_acc,none"
            assert "prompt_level_strict_acc,none" in ref
        elif task.task_name == "leaderboard_math_hard":
            assert score.score_func is score_multilevel_keys_mean
            assert len(keys) == 7
            assert "mean of results.leaderboard_math_*_hard" in ref
        else:
            pytest.fail(f"{repo}: unexpected OLL-cited task {task.task_name}")


def _oll_cited_tasks():
    for repo, config in sorted(_eval_config_map.items()):
        for task in config.tasks:
            if task.score and (task.score.published_score_ref or "").startswith(
                OLL_RESULTS
            ):
                yield repo, task


def test_oll_citations_run_the_same_task_and_template():
    """An OLL-cited score is comparable only if TTIS runs the same leaderboard
    task with the same chat-template setting the leaderboard recorded."""
    cited = list(_oll_cited_tasks())
    assert cited
    for repo, task in cited:
        assert task.task_name in OLL_TASKS, (
            f"{repo}: {task.task_name} cites the leaderboard but is not the "
            f"leaderboard's own task"
        )
        assert repo in OLL_CHAT_TEMPLATE, f"{repo}: record its OLL chat_template"
        assert task.apply_chat_template is OLL_CHAT_TEMPLATE[repo], (
            f"{repo}/{task.task_name}: apply_chat_template must match the cited "
            f"run (chat_template recorded: {OLL_CHAT_TEMPLATE[repo]})"
        )


def test_generative_mmlu_pro_never_cites_loglikelihood_mmlu_pro():
    """TTIS mmlu_pro is generative CoT; leaderboard_mmlu_pro is log-likelihood."""
    for repo, config in _eval_config_map.items():
        for task in config.tasks:
            if task.task_name != "mmlu_pro" or not task.score:
                continue
            ref = task.score.published_score_ref or ""
            assert "leaderboard_mmlu_pro" not in ref, repo
            assert "Open LLM Leaderboard" not in ref, repo
            assert not ref.startswith(OLL_RESULTS), repo
