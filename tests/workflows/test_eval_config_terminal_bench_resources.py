#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Terminal-Bench trials get the CPU and memory their task.toml declares.

The Terminal-Bench leaderboard rejects resource overrides, and the published
scores we compare against use the task budgets (TB 2.1 tasks ask for 1-4 CPUs
and 2-8 GB). A Terminal-Bench task overrides them only where its published
score states a different setting. The Qwen3.6 model cards report Terminal-Bench
2.0 with 32 CPUs and 48 GB RAM; those models also run on QB2, whose 16 CPUs are
the most Docker grants a container there.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

PUBLISHED_OVERRIDES = {
    "Qwen/Qwen3.6-27B": (16, 48 * 1024),
    "Qwen/Qwen3.6-35B-A3B": (16, 48 * 1024),
}


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
