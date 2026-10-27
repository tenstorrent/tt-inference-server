# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3.1-Flash reuses a subset of MiniMax-M3's eval tasks, plus DeepSWE,
without targets."""

from dataclasses import replace

from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
)
from reference_config.evals.eval_config import (
    ALL_EVAL_CONFIGS,
    DEEPSWE_TEST_WORKER_ENV,
)

M3 = "MiniMaxAI/MiniMax-M3"
M31_FLASH = "MiniMaxAI/MiniMax-M3.1-Flash"
M3_TASKS = {
    "r1_gpqa_diamond",
    "longbench2_generate",
    "terminal_bench_4",
    "automation_bench",
}


def _tasks(repo):
    return {t.task_name: t for t in ALL_EVAL_CONFIGS[repo].tasks}


def test_m31_flash_task_set():
    assert _tasks(M31_FLASH).keys() == M3_TASKS | {"deepswe"}


def test_m31_flash_m3_tasks_match_m3_apart_from_targets():
    m3 = _tasks(M3)
    for name, task in _tasks(M31_FLASH).items():
        if name in M3_TASKS:
            assert replace(m3[name], score=task.score) == task, name


def test_m31_flash_deepswe_reaches_the_server_and_caps_test_workers():
    cfg = _tasks(M31_FLASH)["deepswe"].agentic_eval_config
    assert cfg.dataset == "datacurve/deep-swe"
    assert cfg.allow_agent_endpoint
    assert cfg.environment_env == DEEPSWE_TEST_WORKER_ENV
    assert cfg.verifier_env == DEEPSWE_TEST_WORKER_ENV


def test_m31_flash_has_no_targets():
    for name, task in _tasks(M31_FLASH).items():
        score = task.score
        assert score.published_score is None, name
        assert score.gpu_reference_score is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS
