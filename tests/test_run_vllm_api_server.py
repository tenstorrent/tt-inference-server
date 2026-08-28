#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from workflows.utils import get_repo_root_path

MODULE_PATH = get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"


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
    utils.__path__ = [str(get_repo_root_path() / "utils")]
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


def test_set_vllm_sys_argv_merges_defaults_and_passthrough_overrides(
    monkeypatch, run_vllm_api_server_module
):
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py", "--stale-wrapper-arg"])

    args = argparse.Namespace(service_port=7001)
    default_vllm_args = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "port": 8000,
        "max_model_len": "8192",
        "seed": "9472",
        "disable-log-requests": False,
    }

    run_vllm_api_server_module.set_vllm_sys_argv(
        args,
        [
            "--max-model-len",
            "4096",
            "--disable-log-requests",
            "--guided-decoding-backend=outlines",
        ],
        default_vllm_args,
    )

    assert default_vllm_args == {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "port": 8000,
        "max_model_len": "8192",
        "seed": "9472",
        "disable-log-requests": False,
    }
    assert sys.argv == [
        "run_vllm_api_server.py",
        "--max-model-len",
        "4096",
        "--disable-log-requests",
        "--guided-decoding-backend=outlines",
        "--port",
        "7001",
        "--model",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "--seed",
        "9472",
    ]


def test_set_vllm_sys_argv_honors_equals_style_overrides(
    monkeypatch, run_vllm_api_server_module
):
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py"])

    run_vllm_api_server_module.set_vllm_sys_argv(
        argparse.Namespace(service_port=None),
        [
            "--max-model-len=4096",
            "--max-log-len",
            "64",
        ],
        {
            "port": 8000,
            "max_model_len": "8192",
            "max-log-len": "32",
        },
    )

    assert sys.argv == [
        "run_vllm_api_server.py",
        "--max-model-len=4096",
        "--max-log-len",
        "64",
        "--port",
        "8000",
    ]


def test_set_vllm_sys_argv_logs_multiline_bash_command(
    monkeypatch, run_vllm_api_server_module
):
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py"])
    mock_logger = MagicMock()
    monkeypatch.setattr(run_vllm_api_server_module, "logger", mock_logger)

    run_vllm_api_server_module.set_vllm_sys_argv(
        argparse.Namespace(service_port=None),
        [
            "--disable-log-requests",
            "--served-model-name",
            "my model",
            "--guided-decoding-backend=outlines backend",
        ],
        {
            "port": 8000,
        },
    )

    mock_logger.info.assert_called_once_with(
        "vLLM command:\n"
        "vllm serve \\\n"
        "  --disable-log-requests \\\n"
        "  --served-model-name 'my model' \\\n"
        "  '--guided-decoding-backend=outlines backend' \\\n"
        "  --port 8000"
    )


@pytest.mark.parametrize(
    ("argv", "expected_port"),
    [
        (["run_vllm_api_server.py", "--port", "9001"], 9001),
        (["run_vllm_api_server.py", "--port=9002"], 9002),
        (["run_vllm_api_server.py"], 8000),
    ],
)
def test_resolve_service_port_reads_port_from_sys_argv(
    monkeypatch, run_vllm_api_server_module, argv, expected_port
):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("SERVICE_PORT", "8000")

    assert run_vllm_api_server_module.resolve_service_port() == expected_port


def test_main_passes_passthrough_port_to_trace_capture(
    monkeypatch, run_vllm_api_server_module
):
    args = argparse.Namespace(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        tt_device="n150",
        device=None,
        engine=None,
        impl=None,
        no_auth=False,
        disable_trace_capture=False,
        service_port=None,
    )
    model_spec = {
        "model_id": "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150",
        "impl": {"impl_id": "tt-transformers"},
        "device_model_spec": {"vllm_args": {"port": 8000}},
    }

    monkeypatch.setattr(
        run_vllm_api_server_module, "parse_args", lambda: (args, ["--port", "9001"])
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "load_model_spec",
        MagicMock(return_value=model_spec),
    )
    monkeypatch.setattr(run_vllm_api_server_module, "set_cache_paths", MagicMock())
    monkeypatch.setattr(
        run_vllm_api_server_module, "ensure_weights_available", MagicMock()
    )
    monkeypatch.setattr(run_vllm_api_server_module, "register_tt_models", MagicMock())
    monkeypatch.setattr(
        run_vllm_api_server_module, "set_metal_timeout_env_vars", MagicMock()
    )
    monkeypatch.setattr(run_vllm_api_server_module, "set_runtime_env_vars", MagicMock())
    monkeypatch.setattr(run_vllm_api_server_module, "runtime_settings", MagicMock())
    monkeypatch.setattr(run_vllm_api_server_module.runpy, "run_module", MagicMock())
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py"])
    start_trace_capture = MagicMock()
    monkeypatch.setattr(
        run_vllm_api_server_module, "start_trace_capture", start_trace_capture
    )

    run_vllm_api_server_module.main()

    start_trace_capture.assert_called_once_with(
        model_spec,
        disable_trace_capture=False,
        service_port=9001,
    )


def _materialized_quetzal_contract(monkeypatch, tmp_path, module):
    package_id = "sha256-test-artifact-test-weights"
    package_root = tmp_path / package_id
    package_root.mkdir()
    monkeypatch.setenv("QUETZAL_VLLM", "1")
    monkeypatch.setenv("QUETZAL_PACKAGE_ID", package_id)
    monkeypatch.setenv("QUETZAL_PACKAGE_ROOT", str(package_root))
    monkeypatch.setenv("QZ_MODELS_ROOT", str(package_root))
    paths = {
        "QZ_QUALIFICATION_MANIFEST": "qualification_manifest.yaml",
        "QUETZAL_PREFILL_GENERATED_PY": "compiled/artifact/full/prefill/generated.py",
        "QUETZAL_DECODE_GENERATED_PY": "compiled/artifact/full/decode/generated.py",
        "QUETZAL_PREFILL_METADATA_JSON": "compiled/artifact/full/prefill/metadata.json",
        "QUETZAL_DECODE_METADATA_JSON": "compiled/artifact/full/decode/metadata.json",
        "QUETZAL_WEIGHTS": "compiled_weights/weights/full/weights.pt",
    }
    for env_name, relative in paths.items():
        artifact = package_root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("test")
        monkeypatch.setenv(env_name, str(artifact))
    required_runtime = "a" * 40
    (package_root / "qualification_manifest.yaml").write_text(
        "models:\n"
        "  - model_id: Qwen/Qwen3.6-27B\n"
        "    charter_pcc:\n"
        f"      required_runtime_tt_metal_commit: {required_runtime}\n"
    )
    monkeypatch.setenv("QUETZAL_MODEL", "Qwen/Qwen3.6-27B")
    monkeypatch.setenv("TT_METAL_COMMIT_SHA_OR_TAG", required_runtime)

    def row(relative):
        path = package_root / relative
        raw = path.read_bytes()
        tree_relative = relative.split("/", 2)[-1]
        return {
            "path": tree_relative,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "object": f"objects/sha256/00/{'0' * 64}",
        }

    compiled_rows = [
        row(relative)
        for relative in paths.values()
        if relative.startswith("compiled/artifact/")
    ]
    weights_rows = [row(paths["QUETZAL_WEIGHTS"])]
    qualification_raw = (package_root / "qualification_manifest.yaml").read_bytes()
    manifest = {
        "schema": "ttq.artifact_bundle/v1",
        "trees": [
            {"role": "compiled", "name": "artifact", "files": compiled_rows},
            {"role": "compiled_weights", "name": "weights", "files": weights_rows},
        ],
        "qualification_manifest": {
            "path": "qualification_manifest.yaml",
            "size": len(qualification_raw),
            "sha256": hashlib.sha256(qualification_raw).hexdigest(),
            "object": f"objects/sha256/00/{'0' * 64}",
        },
    }
    raw_manifest = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    proof = package_root / ".quetzal-bundle-manifests" / f"{manifest_sha256}.json"
    proof.parent.mkdir()
    proof.write_bytes(raw_manifest)
    monkeypatch.setenv("QUETZAL_BUNDLE_MANIFEST_SHA256", manifest_sha256)
    return package_root


def _quetzal_entry_points():
    return {
        "vllm.general_plugins": [
            types.SimpleNamespace(
                name="quetzal_model_registry",
                value="tt_quetzalcoatlus.vllm_plugin:register",
            )
        ]
    }


def test_main_validates_quetzal_before_runtime_and_skips_native_weight_setup(
    monkeypatch, run_vllm_api_server_module
):
    args = argparse.Namespace(
        model="Qwen3.6-27B",
        tt_device="p300x2",
        device=None,
        engine=None,
        impl="quetzal",
        no_auth=False,
        disable_trace_capture=True,
        service_port=None,
    )
    model_spec = {
        "model_id": "id_quetzal_Qwen3.6-27B_p300x2",
        "device_type": "P300X2",
        "impl": {"impl_id": "quetzal"},
        "device_model_spec": {"vllm_args": {"port": 8000}},
    }
    events = []
    monkeypatch.setattr(
        run_vllm_api_server_module, "parse_args", lambda: (args, [])
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "load_model_spec",
        MagicMock(return_value=model_spec),
    )
    monkeypatch.setattr(run_vllm_api_server_module, "set_cache_paths", MagicMock())
    ensure_weights = MagicMock()
    register_native = MagicMock()
    monkeypatch.setattr(
        run_vllm_api_server_module, "ensure_weights_available", ensure_weights
    )
    monkeypatch.setattr(
        run_vllm_api_server_module, "register_tt_models", register_native
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "set_runtime_env_vars",
        MagicMock(side_effect=lambda _spec: events.append("environment")),
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "validate_quetzal_runtime_contract",
        MagicMock(side_effect=lambda _spec: events.append("validation")),
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "runtime_settings",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("runtime")),
    )
    monkeypatch.setattr(
        run_vllm_api_server_module, "set_metal_timeout_env_vars", MagicMock()
    )
    monkeypatch.setattr(
        run_vllm_api_server_module, "set_vllm_sys_argv", MagicMock()
    )
    monkeypatch.setattr(
        run_vllm_api_server_module, "start_trace_capture", MagicMock()
    )
    monkeypatch.setattr(
        run_vllm_api_server_module.runpy, "run_module", MagicMock()
    )

    run_vllm_api_server_module.main()

    assert events == ["environment", "validation", "runtime"]
    ensure_weights.assert_not_called()
    register_native.assert_not_called()


def test_quetzal_runtime_contract_accepts_materialized_content_package(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    run_vllm_api_server_module.validate_quetzal_runtime_contract(
        {"impl": {"impl_id": "quetzal"}},
        entry_points=_quetzal_entry_points(),
    )


def test_quetzal_runtime_contract_rejects_missing_package_file(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    Path(os.environ["QUETZAL_WEIGHTS"]).unlink()
    with pytest.raises(RuntimeError, match="not materialized.*QUETZAL_WEIGHTS"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points=_quetzal_entry_points(),
        )


def test_quetzal_runtime_contract_rejects_missing_trusted_root_proof(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    digest = os.environ["QUETZAL_BUNDLE_MANIFEST_SHA256"]
    (package_root / ".quetzal-bundle-manifests" / f"{digest}.json").unlink()
    with pytest.raises(RuntimeError, match="missing its installed trusted-root proof"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points=_quetzal_entry_points(),
        )


def test_quetzal_runtime_contract_rejects_tampered_executable(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    Path(os.environ["QUETZAL_DECODE_GENERATED_PY"]).write_text("evil")
    with pytest.raises(RuntimeError, match="trusted-root proof|verification"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points=_quetzal_entry_points(),
        )


def test_quetzal_runtime_contract_rejects_path_escape(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    outside = tmp_path / "outside.py"
    outside.write_text("test")
    monkeypatch.setenv("QUETZAL_PREFILL_GENERATED_PY", str(outside))
    with pytest.raises(RuntimeError, match="escapes QUETZAL_PACKAGE_ROOT"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points=_quetzal_entry_points(),
        )


def test_quetzal_runtime_contract_rejects_missing_plugin(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    with pytest.raises(RuntimeError, match="requires the Quetzal.*entry point"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points={"vllm.general_plugins": []},
        )


def test_quetzal_runtime_contract_rejects_tt_metal_revision_mismatch(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    monkeypatch.setenv("TT_METAL_COMMIT_SHA_OR_TAG", "b" * 40)
    with pytest.raises(RuntimeError, match="TT-Metal runtime mismatch"):
        run_vllm_api_server_module.validate_quetzal_runtime_contract(
            {"impl": {"impl_id": "quetzal"}},
            entry_points=_quetzal_entry_points(),
        )


def test_quetzal_runtime_contract_is_noop_for_native(
    run_vllm_api_server_module,
):
    run_vllm_api_server_module.validate_quetzal_runtime_contract(
        {"impl": {"impl_id": "tt_transformers"}}, entry_points={}
    )


def _weights_spec():
    return {
        "model_name": "Mistral-7B-Instruct-v0.3",
        "hf_model_repo": "mistralai/Mistral-7B-Instruct-v0.3",
    }


def test_ensure_weights_available_resumes_partial_download(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    """A non-empty (partially-downloaded) weights dir must still trigger
    snapshot_download so missing files are fetched, rather than being treated
    as complete."""
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    spec = _weights_spec()
    weights_path = tmp_path / "weights" / spec["model_name"]
    weights_path.mkdir(parents=True)
    (weights_path / "config.json").write_text("{}")  # partial download

    result = run_vllm_api_server_module.ensure_weights_available(spec)

    run_vllm_api_server_module.snapshot_download.assert_called_once()
    kwargs = run_vllm_api_server_module.snapshot_download.call_args.kwargs
    assert kwargs["repo_id"] == spec["hf_model_repo"]
    assert Path(kwargs["local_dir"]) == weights_path
    assert result == weights_path
    assert os.environ["MODEL_WEIGHTS_DIR"] == str(weights_path)


def test_ensure_weights_available_falls_back_when_hub_unreachable(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    """If the hub is unreachable but weights already exist locally, startup
    proceeds with the existing weights instead of crashing."""
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    run_vllm_api_server_module.snapshot_download.side_effect = RuntimeError("offline")
    spec = _weights_spec()
    weights_path = tmp_path / "weights" / spec["model_name"]
    weights_path.mkdir(parents=True)
    (weights_path / "config.json").write_text("{}")

    result = run_vllm_api_server_module.ensure_weights_available(spec)

    assert result == weights_path
    assert os.environ["MODEL_WEIGHTS_DIR"] == str(weights_path)


def test_ensure_weights_available_raises_when_unreachable_and_no_weights(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    """If the hub is unreachable and nothing is cached locally, the failure
    must surface rather than starting with no weights."""
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    run_vllm_api_server_module.snapshot_download.side_effect = RuntimeError("offline")

    with pytest.raises(RuntimeError):
        run_vllm_api_server_module.ensure_weights_available(_weights_spec())
