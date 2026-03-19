#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm-tt-metal"
    / "src"
    / "run_vllm_api_server.py"
)


def _build_catalog():
    model_spec = {
        "model_id": "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150",
        "model_name": "Mistral-7B-Instruct-v0.3",
        "hf_model_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "inference_engine": "vLLM",
        "impl": {
            "impl_id": "tt-transformers",
            "impl_name": "tt-transformers",
        },
        "device_model_spec": {
            "default_impl": True,
        },
    }
    return {
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "N150": {
                "vLLM": {
                    "tt-transformers": model_spec,
                }
            }
        }
    }


@pytest.fixture
def run_vllm_api_server_module(monkeypatch):
    module_name = "test_run_vllm_api_server_module"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)

    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.snapshot_download = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    vllm = types.ModuleType("vllm")
    vllm.ModelRegistry = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    utils = types.ModuleType("utils")
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, "utils", utils)

    logging_utils = types.ModuleType("utils.logging_utils")
    logging_utils.set_vllm_logging_config = MagicMock()
    monkeypatch.setitem(sys.modules, "utils.logging_utils", logging_utils)

    prompt_client = types.ModuleType("utils.prompt_client")
    prompt_client.run_background_trace_capture = MagicMock()
    monkeypatch.setitem(sys.modules, "utils.prompt_client", prompt_client)

    vllm_run_utils = types.ModuleType("utils.vllm_run_utils")
    vllm_run_utils.create_model_symlink = MagicMock()
    vllm_run_utils.get_encoded_api_key = MagicMock(return_value="encoded-api-key")
    monkeypatch.setitem(sys.modules, "utils.vllm_run_utils", vllm_run_utils)

    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wrapped_catalog", [False, True])
def test_load_model_spec_accepts_legacy_and_wrapped_catalogs(
    monkeypatch, tmp_path, run_vllm_api_server_module, wrapped_catalog
):
    catalog = _build_catalog()
    if wrapped_catalog:
        catalog = {
            "schema_version": "0.1.0",
            "release_version": "0.9.0",
            "model_specs": catalog,
        }

    specs_path = tmp_path / "model_spec.json"
    specs_path.write_text(json.dumps(catalog))

    monkeypatch.setenv("MODEL_SPECS_JSON_PATH", str(specs_path))
    monkeypatch.delenv("RUNTIME_MODEL_SPEC_JSON_PATH", raising=False)

    model_spec = run_vllm_api_server_module.load_model_spec(
        model_arg="mistralai/Mistral-7B-Instruct-v0.3",
        device_arg="n150",
    )

    assert model_spec["model_id"] == "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
    assert model_spec["impl"]["impl_name"] == "tt-transformers"
