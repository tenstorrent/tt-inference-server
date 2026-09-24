# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The harbor binary resolves from the explicit ``venv_python``.

When agentic runs as a release child it cannot rely on ``sys.executable``
pointing at the EVALS_AGENTIC venv (the engine runs under WORKFLOW_RUN_SCRIPT).
These tests pin that ``harbor`` resolves relative to the supplied
``venv_python`` instead. Harbor is now the only agentic harness on the host:
the agent (mini-swe-agent) and the SWE-bench grading harness both run inside
the task container, so there are no other binaries to resolve.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_module.agentic import harbor

_VENV_PY = Path("/opt/venvs/evals_agentic/bin/python")


def _harbor_config(tmp_path, venv_python):
    return harbor.HarborRunConfig(
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


def test_harbor_uses_venv_python(tmp_path):
    config = _harbor_config(tmp_path, _VENV_PY)
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        harbor, "run_with_progress", fake_run_with_progress
    ), patch.object(harbor, "_annotate_result_file", lambda *_a, **_k: None):
        rc = harbor.run(config)

    assert rc == 0
    assert captured["cmd"][0] == str(_VENV_PY.parent / "harbor")


def test_harbor_falls_back_to_sys_executable(tmp_path):
    config = _harbor_config(tmp_path, None)
    captured = {}

    def fake_run_with_progress(cmd, *a, **k):
        captured["cmd"] = cmd
        return 0

    with patch.object(
        harbor, "run_with_progress", fake_run_with_progress
    ), patch.object(
        harbor, "_annotate_result_file", lambda *_a, **_k: None
    ), patch.object(harbor.sys, "executable", "/cur/bin/python"):
        harbor.run(config)

    assert captured["cmd"][0] == str(Path("/cur/bin/python").parent / "harbor")


def test_mini_swe_container_uses_docker_host_gateway(tmp_path):
    config = _harbor_config(tmp_path, _VENV_PY)
    config = harbor.HarborRunConfig(
        **{
            **config.__dict__,
            "agent": "mini-swe-agent",
        }
    )

    with patch.object(harbor, "run_with_progress", return_value=17) as run_cmd:
        assert harbor.run(config) == 17

    command = run_cmd.call_args.args[0]
    assert "--config" in command
    harbor_config = json.loads(
        (config.jobs_dir / f"{config.task_name}_harbor_config.json").read_text()
    )
    assert harbor_config["agents"][0]["env"] == {
        "OPENAI_BASE_URL": "http://host.docker.internal:8000/v1",
        "OPENAI_API_BASE": "http://host.docker.internal:8000/v1",
    }
    overlay_path = Path(harbor_config["environment"]["extra_docker_compose"][0])
    assert json.loads(overlay_path.read_text()) == {
        "services": {
            "main": {
                "extra_hosts": ["host.docker.internal:host-gateway"],
            }
        }
    }


def test_host_executed_agent_keeps_loopback_endpoint(tmp_path):
    config = _harbor_config(tmp_path, _VENV_PY)

    with patch.object(harbor, "run_with_progress", return_value=17) as run_cmd:
        assert harbor.run(config) == 17

    command = run_cmd.call_args.args[0]
    endpoint_args = [arg for arg in command if arg.startswith("OPENAI_")]
    assert endpoint_args == [
        "OPENAI_BASE_URL=http://localhost:8000/v1",
        "OPENAI_API_BASE=http://localhost:8000/v1",
    ]
