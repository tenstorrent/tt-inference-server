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


def test_m31_flash_agentic_concurrency_fits_kix1():
    # Sized to the KIX1 shield-evals pool (3 x 64 CPUs): trials that don't fit fail
    # Harbor's pod-ready timeout. M3 keeps its IAD-era values.
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    expected = {
        "terminal_bench_2_1": (21, 8),
        "terminal_bench_4": (21, 8),
        "tau3_bench_banking": (32, 4),
        "swe_bench_verified": (62, 8),
    }
    for name, (flash_n, m3_n) in expected.items():
        assert flash[name].agentic_eval_config.n_concurrent_trials == flash_n, name
        assert m3[name].agentic_eval_config.n_concurrent_trials == m3_n, name


def test_m31_flash_has_no_targets():
    for name, task in _tasks(M31_FLASH).items():
        score = task.score
        assert score.published_score is None, name
        assert score.gpu_reference_score is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS
