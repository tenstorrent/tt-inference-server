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


def _install_artifact_bundle_module(monkeypatch, **attributes):
    serving = types.ModuleType("serving")
    serving.__path__ = []
    artifact_bundle = types.ModuleType("serving.artifact_bundle")
    for name, value in attributes.items():
        setattr(artifact_bundle, name, value)
    monkeypatch.setitem(sys.modules, "serving", serving)
    monkeypatch.setitem(sys.modules, "serving.artifact_bundle", artifact_bundle)
    return artifact_bundle


def _set_quetzal_bundle_env(monkeypatch, root, manifest_sha256="a" * 64):
    monkeypatch.setenv("QUETZAL_PACKAGE_ROOT", str(root))
    monkeypatch.setenv("QUETZAL_BUNDLE_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.delenv("QUETZAL_AUXILIARY_ROOTS_JSON", raising=False)


def _sha256_file(path):
    payload = Path(path).read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_admit_quetzal_bundle_is_noop_for_native_impl(
    monkeypatch, run_vllm_api_server_module
):
    monkeypatch.delenv("QUETZAL_PACKAGE_ROOT", raising=False)

    run_vllm_api_server_module.admit_quetzal_bundle(
        {"impl": {"impl_id": "tt_transformers"}}
    )


def test_admit_quetzal_bundle_rejects_missing_root(run_vllm_api_server_module):
    with pytest.raises(RuntimeError, match="requires QUETZAL_PACKAGE_ROOT"):
        run_vllm_api_server_module.admit_quetzal_bundle(
            {"impl": {"impl_id": "quetzal"}}
        )


def test_admit_quetzal_bundle_rejects_symlinked_root(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    symlink = tmp_path / "bundle-link"
    symlink.symlink_to(bundle, target_is_directory=True)
    _set_quetzal_bundle_env(monkeypatch, symlink)

    with pytest.raises(RuntimeError, match="is not a directory"):
        run_vllm_api_server_module.admit_quetzal_bundle(
            {"impl": {"impl_id": "quetzal"}}
        )


def test_admit_quetzal_bundle_requires_pinned_manifest_digest(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    monkeypatch.setenv("QUETZAL_PACKAGE_ROOT", str(bundle))
    monkeypatch.delenv("QUETZAL_BUNDLE_MANIFEST_SHA256", raising=False)

    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        run_vllm_api_server_module.admit_quetzal_bundle(
            {"impl": {"impl_id": "quetzal"}}
        )


def test_admit_quetzal_portable_v1_bundle(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    manifest_sha256 = "1" * 64
    _set_quetzal_bundle_env(monkeypatch, bundle, manifest_sha256)
    verify_bundle = MagicMock(
        return_value={
            "schema": "ttq.artifact_bundle/v1",
            "total_files": 6,
            "manifest_sha256": manifest_sha256,
        }
    )
    _install_artifact_bundle_module(monkeypatch, verify_bundle=verify_bundle)

    run_vllm_api_server_module.admit_quetzal_bundle({"impl": {"impl_id": "quetzal"}})

    verify_bundle.assert_called_once_with(
        bundle,
        expected_sha256=manifest_sha256,
        auxiliary_roots=None,
    )


def test_admit_quetzal_bundle_propagates_digest_mismatch(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    _set_quetzal_bundle_env(monkeypatch, bundle)
    verify_bundle = MagicMock(side_effect=RuntimeError("root manifest digest mismatch"))
    _install_artifact_bundle_module(monkeypatch, verify_bundle=verify_bundle)

    with pytest.raises(RuntimeError, match="digest mismatch"):
        run_vllm_api_server_module.admit_quetzal_bundle(
            {"impl": {"impl_id": "quetzal"}}
        )


@pytest.mark.parametrize(
    ("auxiliary_json", "expected_error"),
    [
        (None, "v2 bundle requires auxiliary_roots"),
        ('{"wrong": "/immutable/wrong"}', "auxiliary_roots names do not match"),
    ],
)
def test_admit_quetzal_v2_bundle_rejects_missing_or_mismatched_auxiliary_roots(
    monkeypatch,
    tmp_path,
    run_vllm_api_server_module,
    auxiliary_json,
    expected_error,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    _set_quetzal_bundle_env(monkeypatch, bundle)
    if auxiliary_json is not None:
        monkeypatch.setenv("QUETZAL_AUXILIARY_ROOTS_JSON", auxiliary_json)

    def verify_bundle(_root, *, expected_sha256, auxiliary_roots):
        assert expected_sha256 == "a" * 64
        if auxiliary_roots is None:
            raise RuntimeError("v2 bundle requires auxiliary_roots")
        if set(auxiliary_roots) != {"experts"}:
            raise RuntimeError("auxiliary_roots names do not match")

    _install_artifact_bundle_module(monkeypatch, verify_bundle=verify_bundle)

    with pytest.raises(RuntimeError, match=expected_error):
        run_vllm_api_server_module.admit_quetzal_bundle(
            {"impl": {"impl_id": "quetzal"}}
        )


def test_admit_quetzal_v2_bundle_passes_auxiliary_roots(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    _set_quetzal_bundle_env(monkeypatch, bundle)
    auxiliary_root = tmp_path / "sha256-experts"
    monkeypatch.setenv(
        "QUETZAL_AUXILIARY_ROOTS_JSON",
        json.dumps({"experts": str(auxiliary_root)}),
    )
    verify_bundle = MagicMock(
        return_value={
            "schema": "ttq.artifact_bundle/v2",
            "total_files": 6,
            "manifest_sha256": "a" * 64,
            "auxiliary": {"references": 1},
        }
    )
    _install_artifact_bundle_module(monkeypatch, verify_bundle=verify_bundle)

    run_vllm_api_server_module.admit_quetzal_bundle({"impl": {"impl_id": "quetzal"}})

    assert verify_bundle.call_args.kwargs["auxiliary_roots"] == {
        "experts": str(auxiliary_root)
    }


def test_admit_quetzal_installed_shared_bundle(
    monkeypatch, tmp_path, run_vllm_api_server_module
):
    bundle = tmp_path / "bundle"
    artifact_paths = {
        "QZ_QUALIFICATION_MANIFEST": "qualification.json",
        "QUETZAL_PREFILL_GENERATED_PY": "compiled/full/prefill/generated.py",
        "QUETZAL_DECODE_GENERATED_PY": "compiled/full/decode/generated.py",
        "QUETZAL_PREFILL_METADATA_JSON": "compiled/full/prefill/metadata.json",
        "QUETZAL_DECODE_METADATA_JSON": "compiled/full/decode/metadata.json",
        "QUETZAL_WEIGHTS": "compiled_weights/full/weights.pt",
    }
    rows = {}
    for env_name, relative in artifact_paths.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"payload for {env_name}")
        digest, size = _sha256_file(path)
        rows[relative] = {"sha256": digest, "size": size}
        monkeypatch.setenv(env_name, str(path))

    manifest = {
        "schema": "ttq.artifact_bundle/v1",
        "qualification_manifest": rows["qualification.json"],
        "trees": [
            {
                "role": "compiled",
                "name": "full",
                "files": [
                    {"path": relative[len("compiled/full/") :], **rows[relative]}
                    for relative in artifact_paths.values()
                    if relative.startswith("compiled/full/")
                ],
            },
            {
                "role": "compiled_weights",
                "name": "full",
                "files": [
                    {
                        "path": "weights.pt",
                        **rows["compiled_weights/full/weights.pt"],
                    }
                ],
            },
        ],
        "total_bytes": sum(row["size"] for row in rows.values()),
        "total_files": len(rows),
    }
    raw = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    proof = (
        bundle
        / run_vllm_api_server_module.QUETZAL_INSTALLED_MANIFEST_DIR
        / f"{manifest_sha256}.json"
    )
    proof.parent.mkdir()
    proof.write_bytes(raw)
    _set_quetzal_bundle_env(monkeypatch, bundle, manifest_sha256)
    auxiliary_verifier = MagicMock(return_value=None)
    _install_artifact_bundle_module(
        monkeypatch,
        verify_bundle=MagicMock(),
        _canonical_json=_canonical_json,
        _sha256_file=_sha256_file,
        _validate_manifest=lambda value: value,
        _verify_or_admit_auxiliary_references=auxiliary_verifier,
    )

    run_vllm_api_server_module.admit_quetzal_bundle({"impl": {"impl_id": "quetzal"}})

    auxiliary_verifier.assert_called_once_with(manifest, None, hash_payloads=True)


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


def test_main_admits_quetzal_bundle_before_startup_side_effects(
    monkeypatch, run_vllm_api_server_module
):
    args = argparse.Namespace(
        model="meta-llama/Llama-3.2-1B-Instruct",
        tt_device="p300x2",
        device=None,
        engine=None,
        impl="quetzal",
        no_auth=False,
        disable_trace_capture=False,
        service_port=None,
    )
    model_spec = {
        "model_id": "id_quetzal_Llama-3.2-1B-Instruct_p300x2",
        "impl": {"impl_id": "quetzal"},
        "device_model_spec": {"vllm_args": {"port": 8000}},
    }
    monkeypatch.delenv("TT_CACHE_PATH", raising=False)
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("MULTIHOST_ROLE", raising=False)
    monkeypatch.setattr(run_vllm_api_server_module, "parse_args", lambda: (args, []))
    monkeypatch.setattr(
        run_vllm_api_server_module,
        "load_model_spec",
        MagicMock(return_value=model_spec),
    )
    admission_error = RuntimeError("bundle admission failed")
    admit_bundle = MagicMock(side_effect=admission_error)
    monkeypatch.setattr(
        run_vllm_api_server_module, "admit_quetzal_bundle", admit_bundle
    )
    later_calls = [
        "set_cache_paths",
        "ensure_weights_available",
        "register_tt_models",
        "set_runtime_env_vars",
        "set_metal_timeout_env_vars",
        "runtime_settings",
        "set_vllm_sys_argv",
        "start_trace_capture",
    ]
    mocks = {}
    for name in later_calls:
        mocks[name] = MagicMock()
        monkeypatch.setattr(run_vllm_api_server_module, name, mocks[name])
    run_module = MagicMock()
    monkeypatch.setattr(run_vllm_api_server_module.runpy, "run_module", run_module)

    with pytest.raises(RuntimeError, match="bundle admission failed"):
        run_vllm_api_server_module.main()

    admit_bundle.assert_called_once_with(model_spec)
    for mock in mocks.values():
        mock.assert_not_called()
    run_module.assert_not_called()


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
