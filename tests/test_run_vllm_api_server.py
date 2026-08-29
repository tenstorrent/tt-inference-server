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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path

MODULE_PATH = get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
DEV_LLM_SPECS_PATH = (
    get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
)
VLLM_DOCKERFILE_PATH = (
    get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
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


def test_diffusiongemma_launch_uses_standalone_plugin_vllm_024_contract(
    monkeypatch, run_vllm_api_server_module
):
    template = next(
        item
        for item in load_templates_from_yaml(DEV_LLM_SPECS_PATH)
        if item.weights == ["google/diffusiongemma-26B-A4B-it"]
    )
    device_spec = template.expand_to_specs()[0].device_model_spec
    monkeypatch.setattr(sys, "argv", ["run_vllm_api_server.py"])

    run_vllm_api_server_module.set_vllm_sys_argv(
        argparse.Namespace(service_port=8000),
        [],
        device_spec.vllm_args,
    )

    # FlexibleArgumentParser accepts underscores, but normalize them here to
    # compare against the canonical vLLM 0.24 serve flag spellings.
    argv = [
        token.replace("_", "-") if token.startswith("--") else token
        for token in sys.argv[1:]
    ]

    def value(flag):
        return argv[argv.index(flag) + 1]

    assert value("--max-model-len") == "262144"
    assert value("--max-num-batched-tokens") == "262144"
    assert value("--max-num-seqs") == "1"
    assert value("--block-size") == "64"
    assert value("--generation-config") == "vllm"
    assert value("--default-chat-template-kwargs") == '{"enable_thinking": true}'
    assert "--no-enable-prefix-caching" in argv
    assert "--no-enable-chunked-prefill" in argv
    assert "--no-async-scheduling" in argv
    assert json.loads(value("--additional-config")) == {
        "tt": {
            "sample_on_device_mode": "all",
            "enable_model_warmup": True,
            "trace_mode": "all",
            "trace_region_size": 3758096384,
        }
    }
    assert "--reasoning-parser" not in argv
    assert "--vllm-dir" not in argv


def test_vllm_dockerfile_checks_out_supplied_standalone_plugin_ref():
    dockerfile = VLLM_DOCKERFILE_PATH.read_text()

    assert "git clone https://github.com/tenstorrent/vllm-tt-plugin.git" in dockerfile
    assert "git checkout ${TT_VLLM_COMMIT_SHA_OR_TAG}" in dockerfile
    # The plugin block is a plain checkout; the ref-resolution fallback is gone.
    # (tt-metal's own block still legitimately uses git fetch.)
    assert "git ls-remote" not in dockerfile
    assert "resolved_sha" not in dockerfile
    assert "git fetch --depth 1 origin ${TT_VLLM_COMMIT_SHA_OR_TAG}" not in dockerfile
    assert "source docs/install-vllm-tt.sh" in dockerfile
    assert "git clone https://github.com/tenstorrent/vllm.git" not in dockerfile


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


def test_model_spec_can_disable_and_clear_inherited_metal_timeout(
    monkeypatch, run_vllm_api_server_module
):
    monkeypatch.setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "5.0")
    monkeypatch.setenv(
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", "stale-triage-command"
    )
    # Registered with monkeypatch so the value set_runtime_env_vars writes
    # into os.environ is removed again at teardown.
    monkeypatch.setenv("DISABLE_METAL_OP_TIMEOUT", "0")
    model_spec = {
        "device_model_spec": {
            "env_vars": {
                "DISABLE_METAL_OP_TIMEOUT": "1",
            }
        }
    }

    run_vllm_api_server_module.set_runtime_env_vars(model_spec)
    run_vllm_api_server_module.set_metal_timeout_env_vars()

    assert os.environ["DISABLE_METAL_OP_TIMEOUT"] == "1"
    assert "TT_METAL_OPERATION_TIMEOUT_SECONDS" not in os.environ
    assert "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" not in os.environ


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
    env_setup_order = []
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "set_metal_timeout_env_vars",
        MagicMock(side_effect=lambda: env_setup_order.append("metal_timeout")),
    )
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "set_runtime_env_vars",
        MagicMock(side_effect=lambda _spec: env_setup_order.append("runtime_env")),
    )
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
    assert env_setup_order == ["runtime_env", "metal_timeout"]


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
    monkeypatch.setenv("QUETZAL_HF_REVISION", "c" * 40)
    monkeypatch.setenv("VLLM_PLUGINS", "quetzal_model_registry,tt")
    monkeypatch.setenv("TT_VLLM_BUILTIN_MODELS", "0")
    monkeypatch.setenv("TT_METAL_COMMIT_SHA_OR_TAG", required_runtime)
    patchset = "d" * 64
    monkeypatch.setenv("QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256", patchset)
    monkeypatch.setenv("TT_METAL_PATCHSET_SHA256", patchset)
    metal_home = tmp_path / "tt-metal"
    metal_home.mkdir()
    (metal_home / ".ttq-runtime-identity.json").write_text(
        json.dumps(
            {"base_revision": required_runtime, "patchset_sha256": patchset}
        )
    )
    monkeypatch.setenv("TT_METAL_HOME", str(metal_home))

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


def _quetzal_model_spec(model="Qwen/Qwen3.6-27B"):
    return {
        "impl": {"impl_id": "quetzal"},
        "hf_model_repo": model,
        "device_model_spec": {
            "max_context": 8192,
            "vllm_args": {"revision": "c" * 40, "tokenizer_revision": "c" * 40},
        },
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
    monkeypatch.setattr(run_vllm_api_server_module, "parse_args", lambda: (args, []))
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
        "validate_quetzal_runtime",
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
    monkeypatch.setattr(run_vllm_api_server_module, "set_vllm_sys_argv", MagicMock())
    monkeypatch.setattr(run_vllm_api_server_module, "start_trace_capture", MagicMock())
    monkeypatch.setattr(run_vllm_api_server_module.runpy, "run_module", MagicMock())

    run_vllm_api_server_module.main()

    assert events == ["environment", "validation", "runtime"]
    ensure_weights.assert_not_called()
    register_native.assert_not_called()


def test_quetzal_runtime_contract_accepts_materialized_content_package(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    run_vllm_api_server_module._validate_quetzal_package_and_runtime(
        package_root, "Qwen/Qwen3.6-27B"
    )


def test_quetzal_runtime_contract_rejects_missing_package_file(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    Path(os.environ["QUETZAL_WEIGHTS"]).unlink()
    with pytest.raises(RuntimeError, match="not a regular file"):
        run_vllm_api_server_module._validate_quetzal_package_and_runtime(
            package_root, "Qwen/Qwen3.6-27B"
        )


def test_quetzal_runtime_contract_rejects_missing_trusted_root_proof(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    digest = os.environ["QUETZAL_BUNDLE_MANIFEST_SHA256"]
    (package_root / ".quetzal-bundle-manifests" / f"{digest}.json").unlink()
    with pytest.raises(RuntimeError, match="missing trusted-root proof"):
        run_vllm_api_server_module._validate_quetzal_package_and_runtime(
            package_root, "Qwen/Qwen3.6-27B"
        )


def test_quetzal_runtime_contract_rejects_tampered_executable(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    Path(os.environ["QUETZAL_DECODE_GENERATED_PY"]).write_text("evil")
    with pytest.raises(RuntimeError, match="trusted-root proof|verification"):
        run_vllm_api_server_module._validate_quetzal_package_and_runtime(
            package_root, "Qwen/Qwen3.6-27B"
        )


def test_quetzal_runtime_contract_rejects_path_escape(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    outside = tmp_path / "outside.py"
    outside.write_text("test")
    monkeypatch.setenv("QUETZAL_PREFILL_GENERATED_PY", str(outside))
    with pytest.raises(RuntimeError, match="escapes QUETZAL_PACKAGE_ROOT"):
        run_vllm_api_server_module._validate_quetzal_package_and_runtime(
            package_root, "Qwen/Qwen3.6-27B"
        )


def test_quetzal_runtime_contract_rejects_missing_plugin(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(monkeypatch, tmp_path, run_vllm_api_server_module)
    monkeypatch.setattr(
        run_vllm_api_server_module, "_general_plugin_entry_points", lambda: []
    )
    with pytest.raises(RuntimeError, match="requires exactly one installed"):
        run_vllm_api_server_module.validate_quetzal_runtime(_quetzal_model_spec())


def test_quetzal_runtime_contract_rejects_tt_metal_revision_mismatch(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_root = _materialized_quetzal_contract(
        monkeypatch, tmp_path, run_vllm_api_server_module
    )
    monkeypatch.setenv("TT_METAL_COMMIT_SHA_OR_TAG", "b" * 40)
    with pytest.raises(RuntimeError, match="TT-Metal runtime mismatch"):
        run_vllm_api_server_module._validate_quetzal_package_and_runtime(
            package_root, "Qwen/Qwen3.6-27B"
        )


def test_quetzal_runtime_contract_rejects_catalog_package_identity_mismatch(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    _materialized_quetzal_contract(monkeypatch, tmp_path, run_vllm_api_server_module)
    with pytest.raises(RuntimeError, match="model identity mismatch"):
        run_vllm_api_server_module.validate_quetzal_runtime(
            _quetzal_model_spec("other/model")
        )


def test_quetzal_runtime_contract_is_noop_for_native(
    run_vllm_api_server_module,
):
    assert (
        run_vllm_api_server_module.validate_quetzal_runtime(
            {"impl": {"impl_id": "tt_transformers"}}
        )
        is None
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
    assert kwargs["revision"] is None
    assert Path(kwargs["local_dir"]) == weights_path
    assert result == weights_path
    assert os.environ["MODEL_WEIGHTS_DIR"] == str(weights_path)


def test_ensure_weights_available_pins_quetzal_revision(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    spec = _weights_spec()
    spec["device_model_spec"] = {"vllm_args": {"revision": revision}}

    run_vllm_api_server_module.ensure_weights_available(spec)

    assert (
        run_vllm_api_server_module.snapshot_download.call_args.kwargs["revision"]
        == revision
    )


def test_validate_quetzal_runtime_admits_only_generated_provider(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    package_id = "sha256-" + "1" * 64 + "-" + "2" * 64
    root = tmp_path / package_id
    root.mkdir(parents=True)
    manifest = root / "qualification_manifest.yaml"
    runtime_commit = "b534549300fe2af11e6ee828675294bc0e359555"
    patchset = "22fb0bd2523b8a5c63fa20c3c8a1586dc9ead5150449d0eb02231fa8173a7edd"
    revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    model = "Qwen/Qwen3.6-27B"
    manifest.write_text(
        "models:\n"
        f"  - model_id: {model}\n"
        "    charter_pcc:\n"
        f"      required_runtime_tt_metal_commit: {runtime_commit}\n"
    )

    artifact_rows = {}
    artifact_env = {
        "QUETZAL_PREFILL_GENERATED_PY": "compiled/Qwen_Qwen3.6-27B/full/prefill/generated.py",
        "QUETZAL_DECODE_GENERATED_PY": "compiled/Qwen_Qwen3.6-27B/full/decode/generated.py",
        "QUETZAL_PREFILL_METADATA_JSON": "compiled/Qwen_Qwen3.6-27B/full/prefill/metadata.json",
        "QUETZAL_DECODE_METADATA_JSON": "compiled/Qwen_Qwen3.6-27B/full/decode/metadata.json",
        "QUETZAL_WEIGHTS": "compiled_weights/Qwen_Qwen3.6-27B/full/weights.pt",
    }
    for env_name, relative in artifact_env.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = env_name.encode()
        path.write_bytes(payload)
        role, name, nested = relative.split("/", 2)
        artifact_rows.setdefault((role, name), []).append(
            {
                "path": nested,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    qualification_payload = manifest.read_bytes()
    bundle = {
        "schema": "ttq.artifact_bundle/v1",
        "trees": [
            {"role": role, "name": name, "files": rows}
            for (role, name), rows in artifact_rows.items()
        ],
        "qualification_manifest": {
            "size": len(qualification_payload),
            "sha256": hashlib.sha256(qualification_payload).hexdigest(),
        },
    }
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode()
    bundle_digest = hashlib.sha256(bundle_bytes).hexdigest()
    trusted = root / ".quetzal-bundle-manifests" / f"{bundle_digest}.json"
    trusted.parent.mkdir()
    trusted.write_bytes(bundle_bytes)

    metal_home = tmp_path / "tt-metal"
    metal_home.mkdir()
    (metal_home / ".ttq-runtime-identity.json").write_text(
        json.dumps({"base_revision": runtime_commit, "patchset_sha256": patchset})
    )
    for key, value in {
        "QUETZAL_VLLM": "1",
        "QUETZAL_MODEL": model,
        "QUETZAL_HF_REVISION": revision,
        "QUETZAL_PACKAGE_ID": package_id,
        "QUETZAL_PACKAGE_ROOT": str(root),
        "QUETZAL_BUNDLE_MANIFEST_SHA256": bundle_digest,
        "QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256": patchset,
        "TT_METAL_COMMIT_SHA_OR_TAG": runtime_commit,
        "TT_METAL_PATCHSET_SHA256": patchset,
        "TT_METAL_HOME": str(metal_home),
        "VLLM_PLUGINS": "quetzal_model_registry,tt",
        "TT_VLLM_BUILTIN_MODELS": "0",
        "QZ_MODELS_ROOT": str(root),
        "QZ_QUALIFICATION_MANIFEST": str(manifest),
        **{key: str(root / value) for key, value in artifact_env.items()},
    }.items():
        monkeypatch.setenv(key, value)
    model_spec = {
        "impl": {"impl_id": "quetzal"},
        "hf_model_repo": model,
        "device_model_spec": {
            "max_context": 8192,
            "vllm_args": {
                "revision": revision,
                "tokenizer_revision": revision,
            },
        },
    }
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "_general_plugin_entry_points",
        lambda: [
            SimpleNamespace(
                name="quetzal_model_registry",
                value="tt_quetzalcoatlus.vllm_plugin:register",
            )
        ],
    )
    entry = {
        "model_id": model,
        "backend": "generated_quetzal",
        "batch_size": 1,
        "target_mesh": "p150x4",
        "emit_hash": "a" * 64,
        "prefill_buckets": [{"seq_len": 8192}],
    }
    monkeypatch.setattr(
        run_vllm_api_server_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(discover_models=lambda path: {"qwen": entry}),
    )

    assert run_vllm_api_server_module.validate_quetzal_runtime(model_spec) == entry

    monkeypatch.setenv("VLLM_PLUGINS", "quetzal_model_registry,tt,tt_model_registry")
    with pytest.raises(RuntimeError, match="requires exactly"):
        run_vllm_api_server_module.validate_quetzal_runtime(model_spec)


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
