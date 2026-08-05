# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import subprocess
import sys

import pytest

from reference_config.evals.eval_config import (
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.workflow_types import EvalLimitMode


def _eval_tasks():
    return _eval_config_map["google/diffusiongemma-26B-A4B-it"].tasks


def _task(task_name):
    return next(task for task in _eval_tasks() if task.task_name == task_name)


def _gpqa_task():
    return _task("gpqa_diamond_cot_zeroshot")


def test_diffusiongemma_release_evals_are_gpqa_and_terminal_bench_only():
    assert [task.task_name for task in _eval_tasks()] == [
        "gpqa_diamond_cot_zeroshot",
        "terminal_bench_2_1",
    ]

    task = _gpqa_task()

    assert task.use_chat_api is True
    assert task.model_kwargs["max_length"] == 16384
    assert task.gen_kwargs["stream"] == "false"
    assert task.score.score_func_kwargs["result_keys"] == [
        "exact_match,flexible-extract"
    ]


def test_diffusiongemma_gpqa_gpu_reference_requires_more_than_67_percent():
    task = _gpqa_task()
    score = task.score

    assert score.gpu_reference_score == 70.0
    assert score.tolerance == pytest.approx(3 / 70)

    reference = resolve_eval_reference(score, None)
    assert accept_eval_score(reference, 132 / 198 * 100, n_total=198) is False
    assert accept_eval_score(reference, 133 / 198 * 100, n_total=198) is True


def test_diffusiongemma_nightly_gpqa_uses_the_same_quality_gate():
    task = _gpqa_task()
    reference = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)

    # A 5% GPQA subset has approximately ten questions, so its attainable
    # scores jump by ten points: 60% fails and 70% passes the >67% boundary.
    assert reference["is_subset_reference"] is False
    assert accept_eval_score(reference, 60.0, n_total=10) is False
    assert accept_eval_score(reference, 70.0, n_total=10) is True


def test_diffusiongemma_gpqa_reserves_whole_canvas_output_budget():
    task = _gpqa_task()

    assert task.gen_kwargs["max_gen_toks"] == 13824
    assert (
        task.gen_kwargs["max_gen_toks"]
        == (task.model_kwargs["max_length"] - 2432) // 256 * 256
    )


def test_diffusiongemma_gpqa_omits_unsupported_sampling_params():
    task = _gpqa_task()

    for unsupported_key in (
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "logprobs",
        "response_format",
        "bad_words",
    ):
        assert unsupported_key not in task.gen_kwargs


def test_diffusiongemma_terminal_bench_ci_is_small_and_single_request():
    task = _task("terminal_bench_2_1")
    config = task.agentic_eval_config

    assert config.n_concurrent_trials == 1
    assert config.n_attempts == 1
    assert config.task_names_map[EvalLimitMode.CI_NIGHTLY] == [
        "terminal-bench/break-filter-js-from-html",
        "terminal-bench/cobol-modernization",
        "terminal-bench/compile-compcert",
        "terminal-bench/feal-differential-cryptanalysis",
        "terminal-bench/qemu-startup",
    ]
    assert config.agent_kwargs["temperature"] == 1.0
    assert config.agent_kwargs["model_info"] == {
        "max_input_tokens": 192 * 1024,
        "max_output_tokens": 64 * 1024,
    }
    assert config.agent_kwargs["llm_kwargs"] == {
        "top_p": 1.0,
        "max_tokens": 64 * 1024,
        "timeout": 60 * 60,
    }


def test_diffusiongemma_dev_catalog_maps_gpqa_into_release_evals():
    env = {**os.environ, "MODEL_SPECS_ENV": "dev"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from reference_config.evals.eval_config import EVAL_CONFIGS; "
                "assert 'diffusiongemma-26B-A4B-it' in EVAL_CONFIGS"
            ),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
