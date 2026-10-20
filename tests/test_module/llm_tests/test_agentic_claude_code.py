# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Harbor's claude-code agent against a local vLLM server (Anthropic /v1/messages)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from llm_module import DriverContext
from llm_module.drivers.agentic import build_harbor_config
from llm_module.agentic.harbor import run as run_harbor
from reference_config.evals.eval_config import (
    CLAUDE_CODE_AGENT_ENV,
    CLAUDE_CODE_AGENT_KWARGS,
)
from tests.test_module.llm_tests.test_agentic_eval_tests import (
    FakeHarborConfig,
    _harbor_task,
    _server,
)


def _claude_code_task(**config_overrides):
    config = {
        "dataset": "swebenchpro",
        "agent": "claude-code",
        "n_tasks": None,
        "override_cpus": None,
        "override_memory_mb": None,
        "agent_kwargs": dict(CLAUDE_CODE_AGENT_KWARGS),
        "agent_env": dict(CLAUDE_CODE_AGENT_ENV),
        **config_overrides,
    }
    return _harbor_task(
        task_name="swe_bench_pro", agentic_eval_config=FakeHarborConfig(**config)
    )


def _run(task, tmp_path, monkeypatch, **env):
    monkeypatch.setenv("TT_AGENTIC_VERIFIER_SHIM", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = build_harbor_config(
        task, _server(), DriverContext(output_dir=tmp_path, device="P300X2")
    )
    with patch("llm_module.agentic.harbor.run_with_progress") as run_cmd:
        run_cmd.return_value = 0
        with patch("llm_module.agentic.harbor._annotate_result_file"):
            assert run_harbor(cfg) == 0
    harbor_config = json.loads(
        (cfg.jobs_dir / f"{cfg.task_name}_harbor_config.json").read_text()
    )
    return cfg, harbor_config, run_cmd.call_args.kwargs["env"]


def test_model_name_is_the_bare_server_model(tmp_path):
    cfg = build_harbor_config(
        _claude_code_task(), _server(), DriverContext(output_dir=tmp_path, device="P300X2")
    )
    assert cfg.model_name == "Qwen/Qwen3.6-27B"


def test_endpoint_is_gateway_root_with_bearer_token(tmp_path, monkeypatch):
    _, harbor_config, _ = _run(
        _claude_code_task(), tmp_path, monkeypatch, OPENAI_API_KEY="jwt-token"
    )
    agent = harbor_config["agents"][0]
    assert agent["name"] == "claude-code"
    assert agent["model_name"] == "Qwen/Qwen3.6-27B"
    env = agent["env"]
    # Claude Code appends /v1/messages itself: the server root, not /v1.
    assert env["ANTHROPIC_BASE_URL"] == "http://host.docker.internal:8000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "jwt-token"
    assert "OPENAI_BASE_URL" not in env
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert env["API_FORCE_IDLE_TIMEOUT"] == "0"
    assert agent["kwargs"]["reasoning_effort"] == "xhigh"
    assert agent["kwargs"]["version"] == CLAUDE_CODE_AGENT_KWARGS["version"]
    overlay = json.loads(
        Path(harbor_config["environment"]["extra_docker_compose"][0]).read_text()
    )
    assert overlay["services"]["main"]["extra_hosts"] == [
        "host.docker.internal:host-gateway"
    ]


def test_host_anthropic_credentials_are_not_forwarded(tmp_path, monkeypatch):
    _, harbor_config, process_env = _run(
        _claude_code_task(),
        tmp_path,
        monkeypatch,
        OPENAI_API_KEY="jwt-token",
        ANTHROPIC_API_KEY="sk-ant-real-key",
        ANTHROPIC_BASE_URL="https://api.anthropic.com",
        CLAUDE_CODE_OAUTH_TOKEN="oauth",
    )
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"):
        assert key not in process_env
    assert harbor_config["agents"][0]["env"]["ANTHROPIC_AUTH_TOKEN"] == "jwt-token"


def test_explicit_agent_env_endpoint_wins(tmp_path, monkeypatch):
    task = _claude_code_task()
    task.agentic_eval_config.agent_env["ANTHROPIC_BASE_URL"] = "http://proxy:4000"
    _, harbor_config, _ = _run(task, tmp_path, monkeypatch, OPENAI_API_KEY="jwt")
    assert harbor_config["agents"][0]["env"]["ANTHROPIC_BASE_URL"] == "http://proxy:4000"


def test_absolute_dataset_is_a_local_task_directory(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "nl2repobench_tasks"
    tasks_dir.mkdir()
    task = _claude_code_task(dataset=str(tasks_dir))
    task.agentic_eval_config.task_names = ["tinydb"]
    _, harbor_config, _ = _run(task, tmp_path, monkeypatch, OPENAI_API_KEY="jwt")
    dataset = harbor_config["datasets"][0]
    assert dataset["path"] == str(tasks_dir)
    assert "name" not in dataset
