# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from types import SimpleNamespace

from llm_module.eval_configs import (
    agentic_task_required_context,
    select_agentic_eval_tasks,
)
from reference_config.evals.eval_config import EVAL_CONFIGS
from workflows.workflow_types import EvalLimitMode


def _task():
    config = EVAL_CONFIGS["Llama-3.2-1B-Instruct"]
    return next(
        task for task in config.tasks if task.task_name == "swe_bench_verified_32k"
    )


def _model_spec(max_context):
    return SimpleNamespace(
        device_model_spec=SimpleNamespace(max_context=max_context),
    )


def test_llama_1b_32k_swe_task_has_exact_deterministic_envelope():
    task = _task()
    config = task.swebench_eval_config

    assert config.max_input_tokens == 24 * 1024
    assert config.max_output_tokens == 8 * 1024
    assert agentic_task_required_context(task) == 32 * 1024
    assert config.temperature == 0.0
    assert config.top_p == 1.0
    assert config.completion_kwargs == {"extra_body": {"top_k": 0}}
    assert config.n_concurrent_trials == 1
    assert config.max_workers == 1


def test_llama_1b_32k_swe_task_has_fixed_smoke_subset():
    subsets = _task().swebench_eval_config.instance_ids_map

    assert subsets[EvalLimitMode.SMOKE_TEST] == ["django__django-11299"]


def test_llama_1b_32k_swe_task_is_reachable_only_at_its_full_envelope():
    task = _task()
    runtime = SimpleNamespace(agentic_benchmark=task.task_name)

    assert select_agentic_eval_tasks([task], _model_spec(32 * 1024), runtime) == [task]
    assert select_agentic_eval_tasks([task], _model_spec(32 * 1024 - 1), runtime) == []
