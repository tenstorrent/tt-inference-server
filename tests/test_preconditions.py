# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from workflows.local_preconditions import (
    filter_sensitive_variables,
    get_run_command_reconstruction,
    build_precondition_superset,
)


def test_filter_sensitive_variables_redacts_and_excludes_config():
    env = {
        "JWT_SECRET": "secretvalue",
        "HF_TOKEN": "hf_xxx",
        "MY_PASSWORD": "p@ss",
        "SOME_API_KEY": "key",
        "VLLM_MAX_NUM_BATCHED_TOKENS": "131072",
        "MAX_TOKENS": "2048",
        "UNRELATED": "value",
    }
    filtered = filter_sensitive_variables(env, include_sensitive=False)
    # Secrets should be redacted
    assert filtered["JWT_SECRET"] == "<REDACTED>"
    assert filtered["HF_TOKEN"] == "<REDACTED>"
    assert filtered["MY_PASSWORD"] == "<REDACTED>"
    assert filtered["SOME_API_KEY"] == "<REDACTED>"
    # Config token knobs should not be redacted
    assert filtered["VLLM_MAX_NUM_BATCHED_TOKENS"] == "131072"
    assert filtered["MAX_TOKENS"] == "2048"
    # Unrelated should pass through
    assert filtered["UNRELATED"] == "value"


@pytest.mark.parametrize(
    "env_overrides, expected_contains",
    [
        (
            {
                "HF_MODEL_REPO_ID": "meta-llama/Llama-3.1-8B-Instruct",
                "MESH_DEVICE": "N300",
                "MODEL_IMPL": "tt-transformers",
                "SERVICE_PORT": "7001",
                "VLLM_MAX_MODEL_LEN": "65536",
                "OVERRIDE_TT_CONFIG": '{"foo": "bar"}',
            },
            ["--model Llama-3.1-8B-Instruct", "--device n300", "--service-port 7001", "--override-tt-config '{\"foo\": \"bar\"}'", "--vllm-override-args"],
        ),
    ],
)
def test_get_run_command_reconstruction(monkeypatch, env_overrides, expected_contains):
    # Clear possibly conflicting vars first
    keys_to_clear = [
        "HF_MODEL_REPO_ID",
        "MESH_DEVICE",
        "MODEL_IMPL",
        "SERVICE_PORT",
        "VLLM_MAX_MODEL_LEN",
        "OVERRIDE_TT_CONFIG",
        "CONTAINER_APP_USERNAME",
        "HOSTNAME",
    ]
    for k in keys_to_clear:
        monkeypatch.delenv(k, raising=False)

    for k, v in env_overrides.items():
        monkeypatch.setenv(k, v)

    result = get_run_command_reconstruction()
    assert result["command"] is not None
    for token in expected_contains:
        assert token in result["command"]


def test_build_precondition_superset_inserts_model_spec_first():
    pre = {
        "timestamp": "2025-08-11T22:14:39",
        "environment_vars": {"statistics": {"total_variables": 1}},
    }
    model_spec = {"hf_model_repo": "meta-llama/Llama-3.2-3B-Instruct"}
    superset = build_precondition_superset(pre, model_spec)
    # Ensure keys are ordered: timestamp, tt_model_spec, environment_vars
    assert list(superset.keys())[:2] == ["timestamp", "tt_model_spec"]
    assert superset["tt_model_spec"]["hf_model_repo"] == "meta-llama/Llama-3.2-3B-Instruct"


