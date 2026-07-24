# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the GuideLLM scenario presets and scenario-mode driver."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_module import GuideLLMScenario, LLMRunConfig, build_guidellm_scenarios
from llm_module.config import DriverContext, ServerConnection
from llm_module.drivers.guidellm import GuideLLMDriver


def _spec(modalities=None):
    return SimpleNamespace(
        model_name="m",
        hf_model_repo="org/m",
        supported_modalities=modalities or ["text"],
    )


def test_scenario_is_an_llm_run_config():
    s = GuideLLMScenario(isl=0, osl=0, max_concurrency=1, num_prompts=0, name="x")
    assert isinstance(s, LLMRunConfig)


def test_default_scenarios_text_only_model(tmp_path):
    runs = build_guidellm_scenarios(
        _spec(["text"]), SimpleNamespace(workflow_args=None), output_root=tmp_path
    )
    names = [r.name for r in runs]
    # multi_turn_chat + custom_dataset + omni_modal_text; image/video/audio gated out
    assert names == ["multi_turn_chat", "custom_dataset", "omni_modal_text"]
    # generated multi-turn dataset is materialized on disk
    mt = next(r for r in runs if r.name == "multi_turn_chat")
    assert Path(mt.data).is_file()


def test_omni_modal_modalities_gated_by_model(tmp_path):
    runs = build_guidellm_scenarios(
        _spec(["text", "image", "audio"]),
        SimpleNamespace(workflow_args=None),
        output_root=tmp_path,
    )
    names = [r.name for r in runs]
    assert "omni_modal_image" in names
    assert "omni_modal_audio" in names
    assert "omni_modal_video" not in names  # not advertised


def test_workflow_args_select_subset_and_override(tmp_path):
    runs = build_guidellm_scenarios(
        _spec(["text"]),
        SimpleNamespace(
            workflow_args="scenarios=custom_dataset custom_data=mbpp custom_max_seconds=99"
        ),
        output_root=tmp_path,
    )
    assert [r.name for r in runs] == ["custom_dataset"]
    assert runs[0].max_seconds == 99


def test_auth_token_embedded_in_backend_kwargs(tmp_path):
    runs = build_guidellm_scenarios(
        _spec(["text"]),
        SimpleNamespace(workflow_args="scenarios=custom_dataset"),
        output_root=tmp_path,
        auth_token="tok123",
    )
    assert json.loads(runs[0].backend_kwargs)["api_key"] == "tok123"


def test_scenario_mode_builds_guidellm_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return 0

    monkeypatch.setattr("llm_module.drivers.guidellm.run_command", fake_run)
    monkeypatch.setattr(
        "llm_module.drivers.guidellm.load_json", lambda p: {"benchmarks": []}
    )

    driver = GuideLLMDriver(venv_python=Path("/tmp/venv/bin/python"))
    scenario = GuideLLMScenario(
        isl=0,
        osl=0,
        max_concurrency=1,
        num_prompts=10,
        name="multi_turn_chat",
        data="/data/mt.jsonl",
        profile="synchronous",
        request_type="chat_completions",
        max_requests=10,
        max_seconds=120,
        data_column_mapper='{"text_column":"turn_1"}',
        backend_kwargs='{"max_tokens":64}',
    )
    server = ServerConnection(
        base_url="http://127.0.0.1", service_port=8000, model="org/m"
    )
    ctx = DriverContext(output_dir=tmp_path)

    result = driver.run(scenario, server, ctx)
    cmd = " ".join(captured["cmd"])
    assert "--profile synchronous" in cmd
    assert "--data /data/mt.jsonl" in cmd
    assert "--request-type chat_completions" in cmd
    assert "--max-requests 10" in cmd
    assert "http://127.0.0.1:8000/v1" in cmd  # v1-style /v1 base
    assert result.return_code == 0


def test_sweep_mode_unchanged(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "llm_module.drivers.guidellm.run_command",
        lambda cmd, **k: captured.setdefault("cmd", cmd) or 0,
    )
    monkeypatch.setattr(
        "llm_module.drivers.guidellm.load_json", lambda p: {"benchmarks": []}
    )
    driver = GuideLLMDriver(venv_python=Path("/tmp/venv/bin/python"))
    cfg = LLMRunConfig(isl=128, osl=128, max_concurrency=4, num_prompts=20)
    server = ServerConnection(
        base_url="http://127.0.0.1", service_port=8000, model="org/m"
    )
    driver.run(cfg, server, DriverContext(output_dir=tmp_path))
    cmd = " ".join(captured["cmd"])
    assert "--rate-type concurrent" in cmd
    assert "prompt_tokens=128,output_tokens=128" in cmd


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
