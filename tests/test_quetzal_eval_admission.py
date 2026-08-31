# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace

import pytest

from llm_module.eval_configs import get_llm_eval_tasks
from reference_config.evals.eval_config import EVAL_CONFIGS
from test_module.llm_tests.agentic_eval_tests import _select_agentic_tasks
from workflow_module.workflows import _has_agentic_tasks


def _context(model_name, hf_model_repo, max_context):
    model_spec = SimpleNamespace(
        model_name=model_name,
        hf_model_repo=hf_model_repo,
        device_model_spec=SimpleNamespace(max_context=max_context),
    )
    return SimpleNamespace(
        model_spec=model_spec,
        all_params=SimpleNamespace(tasks=EVAL_CONFIGS[model_name].tasks),
        runtime_config=SimpleNamespace(
            agentic_benchmark=None,
            external_agentic_contract=None,
        ),
    )


def test_qwen_s8192_selects_only_context_compatible_gsm8k():
    ctx = _context("Qwen3.6-27B", "Qwen/Qwen3.6-27B", 8192)

    assert [task.task_name for task in get_llm_eval_tasks(ctx.model_spec)] == ["gsm8k"]
    assert _has_agentic_tasks(ctx) is False
    assert _select_agentic_tasks(ctx) == []


def test_gemma_s4096_selects_context_compatible_gsm8k_only():
    ctx = _context("gemma-4-31B-it", "google/gemma-4-31B-it", 4096)

    assert [task.task_name for task in get_llm_eval_tasks(ctx.model_spec)] == ["gsm8k"]
    assert _has_agentic_tasks(ctx) is False
    assert _select_agentic_tasks(ctx) == []


def test_gpt_s8192_selects_context_compatible_gsm8k_only():
    ctx = _context("gpt-oss-120b", "openai/gpt-oss-120b", 8192)

    assert [task.task_name for task in get_llm_eval_tasks(ctx.model_spec)] == ["gsm8k"]
    assert _has_agentic_tasks(ctx) is False
    assert _select_agentic_tasks(ctx) == []


def test_explicit_inadmissible_agentic_task_fails_instead_of_clamping():
    ctx = _context("Qwen3.6-27B", "Qwen/Qwen3.6-27B", 8192)
    ctx.runtime_config.agentic_benchmark = "tb2.0"

    with pytest.raises(RuntimeError, match="matched no EVALS_AGENTIC tasks"):
        _select_agentic_tasks(ctx)


def test_explicit_inadmissible_standard_task_fails_instead_of_clamping():
    ctx = _context("gemma-4-31B-it", "google/gemma-4-31B-it", 4096)
    ctx.runtime_config.eval_samples = '{"r1_gpqa_diamond":[0]}'

    with pytest.raises(ValueError, match="none match this model's eval tasks"):
        get_llm_eval_tasks(ctx.model_spec, ctx.runtime_config)
