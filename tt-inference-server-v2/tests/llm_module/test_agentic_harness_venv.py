# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Agentic harness binaries resolve from the explicit ``venv_python``.

When agentic runs as a release child it cannot rely on ``sys.executable``
pointing at the EVALS_AGENTIC venv (the engine runs under V2_RUN_SCRIPT). These
tests pin that harbor / sweagent / mini-extra / swebench all resolve relative
to the supplied ``venv_python`` instead.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    with patch.object(terminal_bench.subprocess, "run", fake_run), patch.object(
        terminal_bench, "_annotate_result_file", lambda *_a, **_k: None
    ):
        rc = terminal_bench.run(config)

    assert rc == 0
    assert captured["cmd"][0] == str(_VENV_PY.parent / "harbor")


def test_terminal_bench_falls_back_to_sys_executable(tmp_path):
    config = _tb_config(tmp_path, None)
    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    with patch.object(terminal_bench.subprocess, "run", fake_run), patch.object(
        terminal_bench, "_annotate_result_file", lambda *_a, **_k: None
    ), patch.object(terminal_bench.sys, "executable", "/cur/bin/python"):
        terminal_bench.run(config)

    assert captured["cmd"][0] == str(Path("/cur/bin/python").parent / "harbor")


def _swe_config(tmp_path, venv_python, backend="swe-agent"):
    return swebench.SWEbenchRunConfig(
        task_name="swe",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        dataset_split="test",
        sweagent_subset="verified",
        agent_backend=backend,
        model_name="m",
        api_base="http://localhost:8000/v1",
        output_dir=tmp_path / "out",
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
