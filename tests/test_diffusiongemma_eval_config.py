# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import os
import subprocess
import sys

from reference_config.evals.eval_config import _eval_config_map


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
