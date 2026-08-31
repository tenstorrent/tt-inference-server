# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

from llm_module.eval_configs import filter_tasks_by_min_context
from reference_config.evals.eval_config import EVAL_CONFIGS


def _model_spec(max_context):
    return SimpleNamespace(device_model_spec=SimpleNamespace(max_context=max_context))


def test_gemma_long_context_evals_declare_their_unclamped_envelopes():
    tasks = {task.task_name: task for task in EVAL_CONFIGS["gemma-4-31B-it"].tasks}
    assert tasks["r1_gpqa_diamond"].min_context_required == 131072
    assert tasks["terminal_bench_2"].min_context_required == 200 * 1024
    assert tasks["swe_bench_verified"].min_context_required == 200 * 1024


def test_gemma_s4096_skips_incomparable_long_context_evals():
    tasks = EVAL_CONFIGS["gemma-4-31B-it"].tasks
    selected = filter_tasks_by_min_context(tasks, _model_spec(4096))
    assert [task.task_name for task in selected] == ["gsm8k"]
    assert selected[0].gen_kwargs["max_gen_toks"] == 768


def test_gemma_native_200k_envelope_retains_all_declared_evals():
    tasks = EVAL_CONFIGS["gemma-4-31B-it"].tasks
    assert filter_tasks_by_min_context(tasks, _model_spec(200 * 1024)) == tasks
