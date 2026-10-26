# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Guards the MiniMax-M3 Terminal-Bench 4.0 eval task."""

from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    TERMINAL_BENCH_4_DATASET_REF,
    TERMINAL_BENCH_4_GPU_TASKS,
)
from workflows.workflow_types import EvalLimitMode


def _tb4_config():
    tasks = ALL_EVAL_CONFIGS["MiniMaxAI/MiniMax-M3"].tasks
    (task,) = [t for t in tasks if t.task_name == "terminal_bench_4"]
    return task.agentic_eval_config


def test_tb4_dataset_is_pinned():
    config = _tb4_config()
    assert config.dataset == "terminal-bench/terminal-bench"
    assert config.dataset_ref == TERMINAL_BENCH_4_DATASET_REF
    assert TERMINAL_BENCH_4_DATASET_REF.startswith("sha256:")


def test_tb4_excludes_gpu_tasks():
    config = _tb4_config()
    assert config.exclude_task_names == TERMINAL_BENCH_4_GPU_TASKS
    assert len(TERMINAL_BENCH_4_GPU_TASKS) == 3
    nightly = config.task_names_map[EvalLimitMode.CI_NIGHTLY]
    assert not set(nightly) & set(TERMINAL_BENCH_4_GPU_TASKS)


def test_tb4_uses_flat_8h_agent_timeout():
    assert _tb4_config().agent_timeout_sec == 8 * 60 * 60


def test_tb4_sampling_matches_m3_rules():
    kwargs = _tb4_config().agent_kwargs
    assert kwargs["llm_kwargs"]["top_p"] == 0.95
    assert kwargs["temperature"] == 1.0
