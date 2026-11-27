#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Terminal-Bench trials get the CPU and memory their task.toml declares.

The Terminal-Bench leaderboard rejects resource overrides, and the published
scores we compare against use the task budgets (TB 2.1 tasks ask for 1-4 CPUs
and 2-8 GB). A Terminal-Bench task overrides them only where its published
score states a different setting. No configured Terminal-Bench task does today:
the Qwen3.6 rows run a five-task Terminal-Bench 2.1 subset with no published
score, so they use the task budgets too.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

# hf_model_repo -> (override_cpus, override_memory_mb) its published score states.
PUBLISHED_OVERRIDES = {}


def _terminal_bench_tasks():
    for model, eval_config in ALL_EVAL_CONFIGS.items():
        for task in eval_config.tasks:
            agentic = getattr(task, "agentic_eval_config", None)
            if agentic is not None and agentic.dataset.startswith("terminal-bench/"):
                yield model, task.task_name, agentic


def test_terminal_bench_tasks_exist():
    assert list(_terminal_bench_tasks())


def test_terminal_bench_uses_the_task_resources_unless_published_otherwise():
    wrong = {
        (model, name): (agentic.override_cpus, agentic.override_memory_mb)
        for model, name, agentic in _terminal_bench_tasks()
        if (agentic.override_cpus, agentic.override_memory_mb)
        != PUBLISHED_OVERRIDES.get(model, (None, None))
    }
    assert not wrong, (
        f"Terminal-Bench resource overrides without a published source: {wrong}"
    )


def test_terminal_bench_caps_cpu_and_leaves_memory_on_auto():
    loose = {
        (model, name): (
            agentic.cpu_enforcement_policy,
            agentic.memory_enforcement_policy,
        )
        for model, name, agentic in _terminal_bench_tasks()
        if (agentic.cpu_enforcement_policy, agentic.memory_enforcement_policy)
        != ("limit", None)
    }
    assert not loose, f"Terminal-Bench trials with another CPU/memory policy: {loose}"
