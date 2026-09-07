# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Agentic harness binaries resolve from the explicit ``venv_python``.

When agentic runs as a release child it cannot rely on ``sys.executable``
pointing at the EVALS_AGENTIC venv (the engine runs under WORKFLOW_RUN_SCRIPT). These
tests pin that harbor / sweagent / mini-extra / swebench all resolve relative
to the supplied ``venv_python`` instead.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import swebench, terminal_bench

_VENV_PY = Path("/opt/venvs/evals_agentic/bin/python")


def _tb_config(tmp_path, venv_python):
    return terminal_bench.TerminalBenchRunConfig(
        task_name="tb",
        dataset="terminal-bench",
        agent="terminus",
        model_name="m",
        jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs={},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        venv_python=venv_python,
    )


def test_terminal_bench_uses_venv_python(tmp_path):
    config = _tb_config(tmp_path, _VENV_PY)
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        terminal_bench, "run_with_progress", fake_run_with_progress
    ), patch.object(terminal_bench, "_annotate_result_file", lambda *_a, **_k: None):
        rc = terminal_bench.run(config)

    assert rc == 0
    assert captured["cmd"][0] == str(_VENV_PY.parent / "harbor")


def test_terminal_bench_falls_back_to_sys_executable(tmp_path):
    config = _tb_config(tmp_path, None)
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        terminal_bench, "run_with_progress", fake_run_with_progress
    ), patch.object(
        terminal_bench, "_annotate_result_file", lambda *_a, **_k: None
    ), patch.object(terminal_bench.sys, "executable", "/cur/bin/python"):
        terminal_bench.run(config)

    assert captured["cmd"][0] == str(Path("/cur/bin/python").parent / "harbor")


def _swe_config(tmp_path, venv_python, backend="swe-agent"):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    return swebench.SWEbenchRunConfig(
        task_name="swe",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        dataset_split="test",
        sweagent_subset="verified",
        agent_backend=backend,
        model_name="m",
        api_base="http://localhost:8000/v1",
        output_dir=output_dir,
        sweagent_config="config/default.yaml",
        mini_config="mini.yaml",
        mini_model_class="cls",
        mini_environment_class="env",
        n_concurrent_trials=1,
        max_workers=1,
        n_tasks=None,
        temperature=0.0,
        top_p=1.0,
        max_input_tokens=1,
        max_output_tokens=None,
        completion_kwargs={},
        swebench_timeout_sec=None,
        shuffle=False,
        random_delay_multiplier=0.0,
        score_existing_predictions=False,
        venv_python=venv_python,
    )


def test_sweagent_command_uses_venv_python(tmp_path):
    config = _swe_config(tmp_path, _VENV_PY)
    cmd = swebench.build_sweagent_command(
        config, tmp_path / "cfg.yaml", tmp_path / "sweout"
    )
    assert cmd[0] == str(_VENV_PY.parent / "sweagent")


def test_mini_sweagent_command_uses_venv_python(tmp_path):
    config = _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent")
    cmd = swebench.build_mini_sweagent_command(
        config, tmp_path / "cfg.yaml", tmp_path / "miniout"
    )
    assert cmd[0] == str(_VENV_PY.parent / "mini-extra")


def test_swebench_harness_command_uses_venv_python(tmp_path):
    config = _swe_config(tmp_path, _VENV_PY)
    cmd = swebench.build_swebench_harness_command(
        config, tmp_path / "preds.jsonl", "run-1"
    )
    assert cmd[0] == str(_VENV_PY)


def test_swebench_harness_command_falls_back_to_sys_executable(tmp_path):
    config = _swe_config(tmp_path, None)
    with patch.object(swebench.sys, "executable", "/cur/bin/python"):
        cmd = swebench.build_swebench_harness_command(
            config, tmp_path / "preds.jsonl", "run-1"
        )
    assert cmd[0] == "/cur/bin/python"


def test_mini_model_config_writes_default_llm_timeout(tmp_path):
    config = _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent")
    config_path = swebench._write_mini_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (
        model_config["model"]["model_kwargs"]["timeout"]
        == swebench.DEFAULT_LLM_TIMEOUT_SEC
    )


def test_mini_model_config_omits_llm_timeout_when_disabled(tmp_path):
    config = dataclasses.replace(
        _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent"),
        llm_timeout_sec=None,
    )
    config_path = swebench._write_mini_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "timeout" not in model_config["model"]["model_kwargs"]


def test_mini_model_config_completion_kwargs_timeout_wins(tmp_path):
    config = dataclasses.replace(
        _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent"),
        completion_kwargs={"timeout": 5},
    )
    config_path = swebench._write_mini_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert model_config["model"]["model_kwargs"]["timeout"] == 5


def test_sweagent_model_config_writes_default_llm_timeout(tmp_path):
    config = _swe_config(tmp_path, _VENV_PY)
    config_path = swebench._write_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (
        model_config["agent"]["model"]["completion_kwargs"]["timeout"]
        == swebench.DEFAULT_LLM_TIMEOUT_SEC
    )


def test_run_command_returns_124_on_timeout(tmp_path):
    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["x"], timeout=1)

    with patch.object(swebench.subprocess, "run", fake_run):
        rc = swebench._run_command(["x"], cwd=tmp_path, env={}, timeout_s=1)
    assert rc == 124


def test_mini_config_writes_container_timeout(tmp_path):
    config = dataclasses.replace(
        _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent"),
        mini_container_timeout_sec=1234,
    )
    config_path = swebench._write_mini_sweagent_model_config(config)
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert model_config["environment"]["container_timeout"] == "1234s"


def test_mini_agent_run_uses_progress_watchdog(tmp_path):
    config = dataclasses.replace(
        _swe_config(tmp_path, _VENV_PY, backend="mini-swe-agent"),
        n_tasks=5,
        n_concurrent_trials=8,
        mini_container_timeout_sec=7200,
    )
    captured = {}

    def fake_run_with_progress(cmd, **kwargs):
        captured.update(kwargs)
        return 124  # stop run() before the preds lookup

    with patch.object(swebench, "run_with_progress", fake_run_with_progress):
        rc = swebench.run(config)

    assert rc == 124
    assert captured["per_task_budget_s"] == 7200
    assert captured["concurrency"] == 8
    assert captured["label"] == config.task_name
    # No explicit hard override -> watchdog relies on ceiling + stall.
    assert captured["hard_timeout_s"] is None


def _run_swe_agent_capturing_timeout(tmp_path, **overrides):
    config = dataclasses.replace(
        _swe_config(tmp_path, _VENV_PY, backend="swe-agent"),
        n_tasks=5,
        n_concurrent_trials=8,
        mini_container_timeout_sec=7200,
        startup_grace_sec=600,
        **overrides,
    )
    captured = {}

    def fake_run_command(cmd, cwd, env, timeout_s=None):
        captured["timeout_s"] = timeout_s
        return 124

    with patch.object(swebench, "_run_command", fake_run_command):
        rc = swebench.run(config)
    return rc, captured


def test_swe_agent_run_uses_derived_flat_timeout(tmp_path):
    rc, captured = _run_swe_agent_capturing_timeout(
        tmp_path, enforce_agent_deadline=True
    )
    assert rc == 124
    # ceil(5/8)=1 wave -> 1 * budget (no grace folded into the ceiling).
    assert captured["timeout_s"] == 7200


def test_swe_agent_run_is_unbounded_when_not_enforcing(tmp_path):
    # An early kill drops unfinished instances from preds.json and inflates the
    # score, so not enforcing must disable the kill on this backend too.
    _, captured = _run_swe_agent_capturing_timeout(
        tmp_path, enforce_agent_deadline=False
    )
    assert captured["timeout_s"] is None
