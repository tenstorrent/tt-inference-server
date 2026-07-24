# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from evals.eval_config import _eval_config_map


def test_diffusiongemma_gpqa_has_full_thinking_output_budget():
    config = _eval_config_map["google/diffusiongemma-26B-A4B-it"]
    task = next(task for task in config.tasks if task.task_name == "gpqa_diamond_cot_zeroshot")

    assert task.use_chat_api is True
    assert task.model_kwargs["max_length"] == 8192
    assert task.gen_kwargs["max_gen_toks"] == 4096
    assert task.gen_kwargs["stream"] == "false"
    # CoT task exposes strict-match / flexible-extract filter keys (not "none").
    assert task.score.score_func_kwargs["result_keys"] == ["exact_match,flexible-extract"]
