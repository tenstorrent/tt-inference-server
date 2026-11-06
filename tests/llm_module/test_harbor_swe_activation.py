# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from llm_module.agentic import harbor


def _config(tmp_path, **overrides):
    config = harbor.HarborRunConfig(
        task_name="swe_bench_verified",
        dataset="swebench-verified",
        agent="mini-swe-agent",
        model_name="test-model",
        jobs_dir=tmp_path,
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=2,
        n_attempts=1,
        environment_type="docker",
        agent_kwargs={},
        n_tasks=2,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=21600,
    )
    return replace(config, **overrides)


def test_swe_agent_startup_file_reaches_scoped_harbor_environment(tmp_path):
    payload = json.loads(harbor._write_harbor_config(_config(tmp_path)).read_text())
    assert payload["agents"][0]["env"]["BASH_ENV"] == "/root/.bashrc"
    assert "PATH" not in payload["agents"][0]["env"]
    assert "BASH_ENV" not in payload.get("environment", {}).get("env", {})
    assert "BASH_ENV" not in payload.get("verifier", {}).get("env", {})


def test_explicit_startup_override_and_other_environment_survive(tmp_path):
    env = harbor._get_agent_env(
        _config(tmp_path, agent_env={"BASH_ENV": "/custom/startup", "EXAMPLE": "kept"})
    )
    assert env["BASH_ENV"] == "/custom/startup"
    assert env["EXAMPLE"] == "kept"


@pytest.mark.parametrize(
    "override",
    [
        {"dataset": "terminal-bench/terminal-bench-2-1"},
        {"agent": "terminus-2"},
        {"environment_type": "modal"},
        {"agent_import_path": "custom.Agent"},
    ],
)
def test_other_routes_do_not_gain_swe_activation(tmp_path, override):
    assert "BASH_ENV" not in harbor._get_agent_env(_config(tmp_path, **override))


def test_bash_startup_activation_is_inherited_by_mini_style_child_shell(tmp_path):
    # Exact startup contents extracted from both official smoke image layers.
    startup = tmp_path / ".bashrc"
    startup.write_text(
        "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed\n"
    )
    task_bin = tmp_path / "testbed-bin"
    task_bin.mkdir()
    probe = task_bin / "task-interpreter"
    probe.write_text('#!/bin/sh\nprintf "%s" "$CONDA_DEFAULT_ENV"\n')
    probe.chmod(0o755)
    env = os.environ | {
        "BASH_ENV": str(startup),
        "TASK_BIN": str(task_bin),
        # Substitute only the unavailable image conda implementation; the
        # exact source/activation line and Bash startup mechanism run intact.
        "BASH_FUNC_source%%": '() { test "$1" = /opt/miniconda3/etc/profile.d/conda.sh; }',
        "BASH_FUNC_conda%%": '() { test "$1 $2" = "activate testbed" || return 1; export CONDA_DEFAULT_ENV=testbed; export PATH="$TASK_BIN:$PATH"; }',
    }
    child = tmp_path / "child.py"
    child.write_text(
        "import subprocess\n"
        "subprocess.run('task-interpreter', shell=True, check=True)\n"
    )
    result = subprocess.run(
        ["bash", "-c", 'exec "$1" "$2"', "bash", sys.executable, str(child)],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout == "testbed"
    assert result.stderr == ""
