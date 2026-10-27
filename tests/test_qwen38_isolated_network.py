"""Isolated SWE keeps generation policy fixed and binds only the model endpoint on the host."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_module.agentic import harbor
from reference_config.evals import qwen38_goal12 as goal
from reference_config.evals.eval_config import EVAL_CONFIGS, EvalLimitMode
from tests.llm_module.test_harbor_dataset_ref import _run_config
from tests.test_qwen38_audit_mount import spec
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig


def test_host_native_server_has_no_published_bridge_port():
    model = spec()
    model.env_vars["QWEN38_AGENT_ISOLATED_NETWORK"] = "1"
    runtime = RuntimeConfig(model="Qwen3.8-27B", workflow="server", device="p300x2",
                            impl="qwen38_autoport", engine="vLLM", docker_server=True)
    command, _ = generate_docker_run_command(model, runtime)
    assert command[command.index("--network") + 1] == "host"
    assert "--publish" not in command


def test_isolation_only_changes_swe_environment_kwargs(monkeypatch):
    config = EVAL_CONFIGS[goal.MODEL]
    controls = goal.controls()
    controls["swe_agent_isolated_network"] = True
    original = copy.deepcopy(config.tasks[0].agentic_eval_config)
    monkeypatch.setattr(goal, "controls", lambda: controls)
    updated = goal.configure(config, EvalLimitMode).tasks[0].agentic_eval_config
    assert updated.environment_kwargs == {**original.environment_kwargs, "owned_agent_isolation": True}
    assert updated.agent_kwargs == original.agent_kwargs
    assert updated.agent_timeout_sec == original.agent_timeout_sec
    assert config.tasks[0].agentic_eval_config == original


def test_isolation_without_owned_job_is_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("TT_AGENTIC_CONTAINER_OWNER", raising=False)
    config = _run_config(tmp_path, environment_kwargs={"owned_agent_isolation": True})
    with pytest.raises(ValueError, match="require an owned Docker job"):
        harbor._write_harbor_config(config)


def test_isolated_compose_does_not_merge_a_conflicting_host_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("TT_AGENTIC_CONTAINER_OWNER", "owner")
    config = replace(_run_config(tmp_path, environment_kwargs={"owned_agent_isolation": True}),
                     agent="mini-swe-agent")
    path = harbor._write_harbor_config(config)
    data = json.loads(path.read_text())
    overlay = Path(data["environment"]["extra_docker_compose"][0])
    assert "extra_hosts" not in json.loads(overlay.read_text())["services"]["main"]
    assert data["environment"]["kwargs"]["owned_agent_isolation"] is True
    assert data["agents"][0]["env"]["OPENAI_BASE_URL"] == "http://host.docker.internal:8000/v1"
    assert data["environment"]["kwargs"]["owned_endpoint_host"] == "localhost"


@pytest.mark.parametrize("override", [dict(agent="terminus-2"),
    dict(agent_import_path="another.Agent"), dict(environment_type="kubernetes")])
def test_isolation_requires_the_installed_phase_hook(monkeypatch, tmp_path, override):
    monkeypatch.setenv("TT_AGENTIC_CONTAINER_OWNER", "owner")
    config = replace(_run_config(tmp_path, environment_kwargs={"owned_agent_isolation": True}),
                     agent="mini-swe-agent")
    with pytest.raises(ValueError, match="requires standard mini-swe-agent"):
        harbor._write_harbor_config(replace(config, **override))
