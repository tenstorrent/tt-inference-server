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


def _gpqa_task():
    config = _eval_config_map["google/diffusiongemma-26B-A4B-it"]
    assert len(config.tasks) == 1
    task = config.tasks[0]
    assert task.task_name == "gpqa_diamond_cot_zeroshot"
    return task


def test_diffusiongemma_release_eval_is_gpqa_only():
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
