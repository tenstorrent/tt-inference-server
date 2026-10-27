# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3.1-Flash reuses MiniMax-M3's eval tasks, without targets."""

from dataclasses import replace

from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
)
from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

M3 = "MiniMaxAI/MiniMax-M3"
M31_FLASH = "MiniMaxAI/MiniMax-M3.1-Flash"


def _tasks(repo):
    return {t.task_name: t for t in ALL_EVAL_CONFIGS[repo].tasks}


def test_m31_flash_runs_every_m3_task():
    assert _tasks(M31_FLASH).keys() == _tasks(M3).keys()


def test_m31_flash_tasks_match_m3_apart_from_targets_and_concurrency():
    m3 = _tasks(M3)
    for name, task in _tasks(M31_FLASH).items():
        expected = replace(m3[name], score=task.score)
        if task.agentic_eval_config is not None:
            expected = replace(
                expected,
                agentic_eval_config=replace(
                    expected.agentic_eval_config,
                    n_concurrent_trials=task.agentic_eval_config.n_concurrent_trials,
                ),
            )
        assert expected == task, name


def test_m31_flash_terminal_bench_fills_the_deployment():
    # Virtu's P/D deployment serves 62 agentic users without slowing down; M3
    # stays at 8 (its IAD deployment did not).
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    for name in ("terminal_bench_2_1", "terminal_bench_4"):
        assert flash[name].agentic_eval_config.n_concurrent_trials == 62
        assert m3[name].agentic_eval_config.n_concurrent_trials == 8


def test_m31_flash_has_no_targets():
    for name, task in _tasks(M31_FLASH).items():
        score = task.score
        assert score.published_score is None, name
        assert score.gpu_reference_score is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS
