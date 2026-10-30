# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Harbor trials are capped at their CPU budget on every environment.

Harbor's ``auto`` policy means a hard limit on Docker but only a request on
Kubernetes, where a trial can then burst to the whole node. The run config
carries an explicit CPU policy so both environments apply the same ceiling.
Memory stays on ``auto``: a hard limit on Docker, and on Kubernetes a request
whose ceiling comes from ``memory_limit_multiplier``. Policies reach Harbor
through the config file's ``environment`` block, or the ``--cpus`` /
``--memory`` flags when Harbor runs without a config file.
"""

import json
from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import harbor
from reference_config.evals.eval_config import HarborEvalConfig


def _run_config(tmp_path, environment_type="docker", **kw):
    return harbor.HarborRunConfig(
        task_name="tb",
        dataset="terminal-bench/terminal-bench-2-1",
        agent="terminus-2",
        model_name="m",
        jobs_dir=tmp_path / "jobs",
        api_base="http://localhost:8000/v1",
        n_concurrent_trials=1,
        n_attempts=1,
        environment_type=environment_type,
        agent_kwargs={},
        n_tasks=None,
        override_cpus=None,
        override_memory_mb=None,
        timeout_multiplier=None,
        agent_timeout_sec=None,
        venv_python=Path("/opt/venvs/evals_agentic/bin/python"),
        **kw,
    )


def _environment(path):
    return json.loads(Path(path).read_text())["environment"]


def _cli_command(config):
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        harbor, "run_with_progress", fake_run_with_progress
    ), patch.object(harbor, "_annotate_result_file", lambda *_a, **_k: None):
        harbor.run(config)
    return captured["cmd"]


def test_harbor_eval_config_caps_cpu_and_leaves_memory_on_auto():
    # Memory stays on ``auto`` so Kubernetes can take memory_limit_multiplier
    # (which Harbor applies only under ``auto``); Docker's auto is a hard limit.
    config = HarborEvalConfig(
        dataset="terminal-bench/terminal-bench-2-1", agent="terminus-2"
    )
    assert config.cpu_enforcement_policy == "limit"
    assert config.memory_enforcement_policy is None


def test_policies_reach_the_config_file_environment_block(tmp_path):
    path = harbor._write_harbor_config(
        _run_config(
            tmp_path,
            environment_type="kubernetes",
            cpu_enforcement_policy="limit",
            memory_enforcement_policy="limit",
        )
    )
    environment = _environment(path)
    assert environment["cpu_enforcement_policy"] == "limit"
    assert environment["memory_enforcement_policy"] == "limit"


def test_unset_policies_leave_harbor_on_auto(tmp_path):
    environment = _environment(harbor._write_harbor_config(_run_config(tmp_path)))
    assert "cpu_enforcement_policy" not in environment
    assert "memory_enforcement_policy" not in environment


def test_policies_reach_the_cli_flags(tmp_path):
    config = _run_config(
        tmp_path, cpu_enforcement_policy="limit", memory_enforcement_policy="limit"
    )
    assert harbor._needs_config_file(config) is False
    cmd = _cli_command(config)
    assert cmd[cmd.index("--cpus") + 1] == "limit"
    assert cmd[cmd.index("--memory") + 1] == "limit"


def test_unset_policies_add_no_cli_flags(tmp_path):
    cmd = _cli_command(_run_config(tmp_path))
    assert "--cpus" not in cmd
    assert "--memory" not in cmd
