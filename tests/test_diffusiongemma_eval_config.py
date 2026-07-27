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


def test_diffusiongemma_gpqa_omits_inert_sampling_params():
    """DiffusionGemma's TT generator discards per-request SamplingParams.

    Its sampler is an internal per-denoise-step temperature schedule
    (t_max 0.8 -> t_min 0.4), so do_sample/temperature/top_k/top_p are no-ops
    here. They must stay out of gen_kwargs: they misreport the run's sampling
    config, and would override the checkpoint schedule if request-param
    plumbing is ever wired up. See the comment in evals/eval_config.py.
    """
    config = _eval_config_map["google/diffusiongemma-26B-A4B-it"]
    task = next(task for task in config.tasks if task.task_name == "gpqa_diamond_cot_zeroshot")

    for inert_key in ("do_sample", "temperature", "top_k", "top_p"):
        assert inert_key not in task.gen_kwargs, (
            f"{inert_key} is inert for DiffusionGemma -- do not copy it from the gemma-4 EvalTask"
        )
