# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""MiniMax-M3.1-Flash reuses MiniMax-M3's eval tasks, without targets, plus DeepSWE v1.1."""

from dataclasses import replace

from reference_config.agentic_traces.agentic_traces_config import (
    AGENTIC_TRACES_CONFIGS,
)
from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

M3 = "MiniMaxAI/MiniMax-M3"
M31_FLASH = "MiniMaxAI/MiniMax-M3.1-Flash"
MIMO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
# Not an M3 task: M3.1-Flash adds it with MiMo's harness settings.
EXTRA = {"deepswe"}


def _tasks(repo):
    return {t.task_name: t for t in ALL_EVAL_CONFIGS[repo].tasks}


def test_m31_flash_runs_every_m3_task():
    assert _tasks(M31_FLASH).keys() == _tasks(M3).keys() | EXTRA


def test_m31_flash_tasks_match_m3_apart_from_targets_and_concurrency():
    m3 = _tasks(M3)
    for name, task in _tasks(M31_FLASH).items():
        if name in EXTRA:
            continue
        expected = replace(m3[name], score=task.score)
        if task.agentic_eval_config is not None:
            cfg = task.agentic_eval_config
            fields = {"n_concurrent_trials": cfg.n_concurrent_trials}
            for name in ("environment_env", "verifier_env"):
                if hasattr(cfg, name):
                    fields[name] = getattr(cfg, name)
            for name in ("override_cpus", "override_memory_mb"):
                if hasattr(cfg, name):
                    fields[name] = getattr(cfg, name)
            expected = replace(
                expected,
                agentic_eval_config=replace(expected.agentic_eval_config, **fields),
            )
        assert expected == task, name


def test_m31_flash_agentic_concurrency_fits_kix1():
    # Sized to the KIX1 shield-evals pool (3 x 64 CPUs): trials that don't fit fail
    # Harbor's pod-ready timeout. M3 keeps its IAD-era values.
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    expected = {
        "terminal_bench_2_1": (62, 8),
        "terminal_bench_4": (40, 8),
        "tau3_bench_banking": (32, 4),
        "swe_bench_verified": (62, 8),
    }
    for name, (flash_n, m3_n) in expected.items():
        assert flash[name].agentic_eval_config.n_concurrent_trials == flash_n, name
        assert m3[name].agentic_eval_config.n_concurrent_trials == m3_n, name


def test_m31_flash_has_no_targets():
    for name, task in _tasks(M31_FLASH).items():
        if name in EXTRA:
            continue
        score = task.score
        assert score.published_score is None, name
        assert score.gpu_reference_score is None, name
        assert not score.mode_reference_scores, name


def test_m31_flash_has_an_agentx_config():
    assert "id_blaze_MiniMax-M3.1-Flash_super_cluster" in AGENTIC_TRACES_CONFIGS


def test_m31_flash_terminal_bench_2_1_uses_task_declared_resources():
    # M3's 8 CPU / 32 GiB override fits 7 trials per KIX1 node; the tasks
    # themselves mostly declare 1 CPU / 2 GiB.
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    cfg = flash["terminal_bench_2_1"].agentic_eval_config
    assert cfg.override_cpus is None and cfg.override_memory_mb is None
    assert m3["terminal_bench_2_1"].agentic_eval_config.override_cpus == 8


def test_m31_flash_tau3_user_sim_and_verifier_call_the_served_m31_name():
    # An M3.1-Flash endpoint serves only MiniMaxAI/MiniMax-M3.1-Flash: tau3's user
    # simulator and NL-assertion verifier must call that name, not M3's (on
    # 2026-10-29 every trial died on the runtime's empty first tool reply).
    flash, m3 = _tasks(M31_FLASH), _tasks(M3)
    assert m3["tau3_bench_banking"].agentic_eval_config.environment_env == {
        "TAU2_USER_MODEL": "openai/MiniMaxAI/MiniMax-M3"
    }
    cfg = flash["tau3_bench_banking"].agentic_eval_config
    assert cfg.environment_env == {
        "TAU2_USER_MODEL": "openai/MiniMaxAI/MiniMax-M3.1-Flash"
    }
    assert cfg.verifier_env == {
        "TAU2_NL_ASSERTIONS_MODEL": "openai/MiniMaxAI/MiniMax-M3.1-Flash"
    }
    for name, task in flash.items():
        cfg = task.agentic_eval_config
        if not hasattr(cfg, "environment_env"):
            continue
        envs = {**cfg.environment_env, **cfg.verifier_env}
        assert "openai/MiniMaxAI/MiniMax-M3" not in envs.values(), name


def test_m31_flash_deepswe_is_mimos_task_scored_against_the_card():
    # Same harness as MiMo's (mini-swe-agent 2.3.0, endpoint-only egress); the card's
    # 73.7 is the published score, and there is no GPU reference yet.
    flash, mimo = _tasks(M31_FLASH)["deepswe"], _tasks(MIMO)["deepswe"]
    assert flash.agentic_eval_config == mimo.agentic_eval_config
    assert flash.agentic_eval_config.allow_agent_endpoint
    assert flash.score.published_score == 73.7
    assert flash.score.gpu_reference_score is None
    assert not flash.score.mode_reference_scores
