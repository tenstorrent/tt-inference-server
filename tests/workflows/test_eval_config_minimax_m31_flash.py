# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3.1-Flash reuses a subset of MiniMax-M3's eval tasks, plus DeepSWE,
with the model card's published scores."""

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
CARD = "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash"
M3_TASKS = {
    "r1_gpqa_diamond",
    "longbench2_generate",
    "terminal_bench_4",
    "automation_bench",
}
PUBLISHED = {"automation_bench": 62.8, "terminal_bench_4": 38.1, "deepswe": 73.7}


def _tasks(repo):
    return {t.task_name: t for t in ALL_EVAL_CONFIGS[repo].tasks}


def test_m31_flash_task_set():
    assert _tasks(M31_FLASH).keys() == M3_TASKS | {"deepswe"}


def test_m31_flash_m3_tasks_match_m3_apart_from_targets_and_tb4_agent():
    m3 = _tasks(M3)
    for name, task in _tasks(M31_FLASH).items():
        if name not in M3_TASKS:
            continue
        expected = replace(m3[name], score=task.score)
        if name == "terminal_bench_4":
            expected = replace(
                expected,
                agentic_eval_config=replace(
                    expected.agentic_eval_config,
                    agent=task.agentic_eval_config.agent,
                    agent_kwargs=task.agentic_eval_config.agent_kwargs,
                ),
            )
        elif name == "automation_bench":
            expected = replace(
                expected,
                agentic_eval_config=replace(
                    expected.agentic_eval_config,
                    n_concurrent_trials=56,
                    agent_kwargs={
                        **expected.agentic_eval_config.agent_kwargs,
                        "reasoning_effort": "max",
                    },
                ),
            )
        else:
            expected = replace(
                expected,
                request_body={**expected.request_body, "reasoning_effort": "max"},
            )
        assert expected == task, name


def test_m31_flash_asks_for_max_effort_on_every_task():
    tasks = _tasks(M31_FLASH)
    for name in ("r1_gpqa_diamond", "longbench2_generate", "deepswe"):
        assert tasks[name].request_body["reasoning_effort"] == "max", name
    for name in ("automation_bench", "terminal_bench_4"):
        cfg = tasks[name].agentic_eval_config
        assert cfg.agent_kwargs["reasoning_effort"] == "max", name
    for task in _tasks(M3).values():
        assert "reasoning_effort" not in task.request_body, task.task_name


def test_m31_flash_automation_bench_stays_under_the_cuda_graph_batch():
    m31 = _tasks(M31_FLASH)["automation_bench"].agentic_eval_config
    assert m31.n_concurrent_trials == 56
    assert _tasks(M3)["automation_bench"].agentic_eval_config.n_concurrent_trials == 62


def test_m31_flash_tb4_runs_the_cards_codex():
    cfg = _tasks(M31_FLASH)["terminal_bench_4"].agentic_eval_config
    assert cfg.agent == "codex"
    assert cfg.agent_kwargs["version"] == "0.149.1"


def test_m3_tb4_still_runs_terminus_2():
    assert _tasks(M3)["terminal_bench_4"].agentic_eval_config.agent == "terminus-2"


def test_m31_flash_deepswe_reaches_the_server_and_caps_test_workers():
    cfg = _tasks(M31_FLASH)["deepswe"].agentic_eval_config
    assert cfg.dataset == "datacurve/deep-swe"
    assert cfg.allow_agent_endpoint
    assert cfg.agent_kwargs["version"] == "2.4.5"
    assert cfg.environment_env == DEEPSWE_TEST_WORKER_ENV
    assert cfg.verifier_env == DEEPSWE_TEST_WORKER_ENV


def test_m31_flash_targets_are_the_cards():
    for name, task in _tasks(M31_FLASH).items():
        score = task.score
        assert score.published_score == PUBLISHED.get(name), name
        assert score.published_score_ref == (CARD if name in PUBLISHED else None)
        assert score.gpu_reference_score is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS
