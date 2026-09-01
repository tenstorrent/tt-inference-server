# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import copy
import hashlib
import json
from pathlib import PurePosixPath

import pytest

import workflows.run_docker_server as run_docker_server_module
from run import apply_quetzal_behavioral_topology
from workflows.model_spec import get_runtime_model_spec, load_templates_from_yaml
from workflows.run_docker_server import (
    _validate_quetzal_models_root,
    _validate_quetzal_runtime_attestation,
    _vllm_override_cli_args,
    generate_docker_run_command,
)
from workflows.runtime_config import RuntimeConfig
from workflows.utils import get_repo_root_path


MODELS = {
    "Qwen3.6-27B": (8192, "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "gemma-4-31B-it": (4096, "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "gpt-oss-120b": (8192, "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"),
}

MODEL_REPOS = {
    "Qwen3.6-27B": "Qwen/Qwen3.6-27B",
    "gemma-4-31B-it": "google/gemma-4-31B-it",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}


def _dev_quetzal_spec(model):
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    template = next(
        item
        for item in templates
        if item.impl.impl_id == "quetzal" and item.weights == [MODEL_REPOS[model]]
    )
    return template.expand_to_specs()[0]


@pytest.mark.parametrize("model", MODELS)
def test_dev_quetzal_specs_are_nondefault_and_revision_pinned(model):
    spec = _dev_quetzal_spec(model)
    max_context, revision = MODELS[model]
    assert spec.impl.impl_name == "quetzal"
    assert spec.inference_engine == "vLLM"
    assert spec.device_model_spec.default_impl is False
    assert spec.device_model_spec.max_concurrency == 1
    assert spec.device_model_spec.max_context == max_context
    assert spec.device_model_spec.vllm_args["revision"] == revision
    assert spec.device_model_spec.vllm_args["tokenizer_revision"] == revision
    assert spec.env_vars["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert spec.env_vars["TT_VLLM_BUILTIN_MODELS"] == "0"
    if model == "Qwen3.6-27B":
        assert "TT_MESH_GRAPH_DESC_PATH" not in spec.env_vars
    assert spec.env_vars["QUETZAL_PACKAGE_ID"].startswith("sha256-")
    assert len(spec.env_vars["QUETZAL_BUNDLE_MANIFEST_SHA256"]) == 64
    assert len(spec.env_vars["QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256"]) == 64
    if model == "gemma-4-31B-it":
        assert (
            spec.env_vars["QUETZAL_REQUIRED_TT_METAL_COMMIT"]
            == "b534549300fe2af11e6ee828675294bc0e359555"
        )

    native, native_impl, _ = get_runtime_model_spec(model=model, device="p300x2")
    assert native_impl != "quetzal"
    assert native.device_model_spec.default_impl is True


def test_gpt_quetzal_forwards_required_agentic_parsers():
    spec = _dev_quetzal_spec("gpt-oss-120b")
    args = spec.device_model_spec.vllm_args
    assert args["enable-auto-tool-choice"] is True
    assert args["tool-call-parser"] == "openai"
    assert args["reasoning-parser"] == "openai_gptoss"
    rendered = _vllm_override_cli_args(json.dumps(args))
    assert "--enable-auto-tool-choice" in rendered
    assert rendered[rendered.index("--tool-call-parser") + 1] == "openai"
    assert rendered[rendered.index("--reasoning-parser") + 1] == "openai_gptoss"


def test_qwen_quetzal_forwards_required_agentic_parsers():
    spec = _dev_quetzal_spec("Qwen3.6-27B")
    args = spec.device_model_spec.vllm_args
    assert args["enable-auto-tool-choice"] is True
    assert args["tool-call-parser"] == "qwen3_coder"
    assert args["reasoning-parser"] == "qwen3"
    rendered = _vllm_override_cli_args(json.dumps(args))
    assert "--enable-auto-tool-choice" in rendered
    assert rendered[rendered.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert rendered[rendered.index("--reasoning-parser") + 1] == "qwen3"


def test_docker_command_mounts_quetzal_root_readonly_and_forwards_impl(tmp_path):
    spec = _dev_quetzal_spec("Qwen3.6-27B")
    package_id = spec.env_vars["QUETZAL_PACKAGE_ID"]
    root = tmp_path / package_id
    root.mkdir()
    (root / "qualification_manifest.yaml").write_text("models: []\n")
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        quetzal_models_root=str(root),
    )

    command, _ = generate_docker_run_command(spec, runtime)

    mount = (
        f"type=bind,src={root.resolve()},"
        f"dst=/home/container_app_user/quetzal/packages/{package_id},readonly"
    )
    assert mount in command
    assert command.count("--impl") == 1
    assert command[command.index("--impl") + 1] == "quetzal"
    assert (
        f"QZ_MODELS_ROOT=/home/container_app_user/quetzal/packages/{package_id}"
        in command
    )
    assert "VLLM_PLUGINS=quetzal_model_registry,tt" in command
    assert "TT_VLLM_BUILTIN_MODELS=0" in command
    assert not any(
        isinstance(arg, str) and "TTIS_QUETZAL_BEHAVIORAL_PACKAGE_ADMISSION=" in arg
        for arg in command
    )


def test_docker_command_behavioral_admission_is_local_explicit_and_readonly(
    monkeypatch, tmp_path
):
    spec = copy.deepcopy(_dev_quetzal_spec("Qwen3.6-27B"))
    package_id = "sha256-" + "1" * 64 + "-" + "2" * 64
    candidate_root = tmp_path / "quetzal"
    package = candidate_root / "nkapre" / "candidates" / package_id
    package.mkdir(parents=True)
    (package / "qualification_manifest.yaml").write_text("models: []\n")
    spec.env_vars["QUETZAL_PACKAGE_ID"] = package_id
    monkeypatch.setattr(
        run_docker_server_module,
        "_QUETZAL_EXABOX_CANDIDATE_ROOT",
        candidate_root,
    )
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        dev_mode=True,
        quetzal_models_root=str(package),
        quetzal_behavioral_package_admission=True,
    )

    command, _ = generate_docker_run_command(spec, runtime)

    assert (
        f"type=bind,src={package.resolve()},"
        f"dst=/home/container_app_user/quetzal/packages/{package_id},readonly"
        in command
    )
    assert "TTIS_QUETZAL_BEHAVIORAL_PACKAGE_ADMISSION=1" in command

    runtime.dev_mode = False
    with pytest.raises(ValueError, match="requires --dev-mode"):
        generate_docker_run_command(spec, runtime)
    runtime.dev_mode = True
    runtime.docker_server = False
    with pytest.raises(ValueError, match="requires --docker-server"):
        generate_docker_run_command(spec, runtime)
    runtime.docker_server = True
    runtime.ci_mode = True
    with pytest.raises(ValueError, match="forbidden in CI mode"):
        generate_docker_run_command(spec, runtime)


def test_docker_command_behavioral_admission_rejects_non_candidate_root(
    monkeypatch, tmp_path
):
    spec = copy.deepcopy(_dev_quetzal_spec("Qwen3.6-27B"))
    package_id = "sha256-" + "1" * 64 + "-" + "2" * 64
    package = tmp_path / package_id
    package.mkdir()
    (package / "qualification_manifest.yaml").write_text("models: []\n")
    spec.env_vars["QUETZAL_PACKAGE_ID"] = package_id
    monkeypatch.setattr(
        run_docker_server_module,
        "_QUETZAL_EXABOX_CANDIDATE_ROOT",
        tmp_path / "other-root",
    )
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        dev_mode=True,
        quetzal_models_root=str(package),
        quetzal_behavioral_package_admission=True,
    )

    with pytest.raises(ValueError, match="behavioral Quetzal package must be under"):
        generate_docker_run_command(spec, runtime)


def test_behavioral_topology_is_local_explicit_and_does_not_mutate_catalogue():
    spec = _dev_quetzal_spec("gemma-4-31B-it")
    original_env = copy.deepcopy(spec.env_vars)
    runtime = RuntimeConfig(
        model="gemma-4-31B-it",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        dev_mode=True,
        quetzal_behavioral_package_admission=True,
        quetzal_behavioral_topology="p150x4-linear-1ch",
    )

    resolved = apply_quetzal_behavioral_topology(runtime, spec)

    assert resolved is not spec
    assert spec.env_vars == original_env
    assert spec.env_vars["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert spec.env_vars["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert resolved.env_vars["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Linear"
    assert resolved.env_vars["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "1"
    assert resolved.env_vars["TTIS_QUETZAL_BEHAVIORAL_MESH_DEVICE"] == "P150x4"
    assert resolved.env_vars["TT_METAL_DISABLE_MULTI_AERISC"] == "1"
    assert resolved.env_vars["HOME"] == "/home/container_app_user/cache_root/home"
    assert resolved.env_vars["USER"] == "container_app_user"
    assert resolved.env_vars["LOGNAME"] == "container_app_user"
    assert resolved.env_vars["XDG_CACHE_HOME"].startswith(
        "/home/container_app_user/cache_root/"
    )
    assert resolved.env_vars["MPLCONFIGDIR"].startswith(
        "/home/container_app_user/cache_root/"
    )
    assert resolved.env_vars["TT_METAL_LOGS_PATH"] == (
        "/home/container_app_user/cache_root/logs"
    )
    assert resolved.env_vars["TT_MESH_GRAPH_DESC_PATH"].endswith(
        "/reference_config/mesh_graph_descriptors/"
        "p150_x4_linear_1ch_mesh_graph_descriptor.textproto"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"quetzal_behavioral_package_admission": False}, "requires .*admission"),
        ({"dev_mode": False}, "requires --dev-mode --docker-server"),
        ({"docker_server": False}, "requires --dev-mode --docker-server"),
        ({"ci_mode": True}, "forbidden in CI mode"),
        ({"runtime_model_spec_json": "/tmp/spec.json"}, "cannot override"),
        ({"impl": "vllm"}, "requires --impl quetzal"),
    ],
)
def test_behavioral_topology_fails_closed(overrides, message):
    spec = _dev_quetzal_spec("gemma-4-31B-it")
    kwargs = {
        "model": "gemma-4-31B-it",
        "workflow": "server",
        "device": "p300x2",
        "impl": "quetzal",
        "engine": "vLLM",
        "docker_server": True,
        "dev_mode": True,
        "quetzal_behavioral_package_admission": True,
        "quetzal_behavioral_topology": "p150x4-linear-1ch",
    }
    kwargs.update(overrides)
    runtime = RuntimeConfig(**kwargs)

    with pytest.raises(ValueError, match=message):
        apply_quetzal_behavioral_topology(runtime, spec)


def test_docker_command_rejects_missing_or_misapplied_quetzal_root(tmp_path):
    qz_spec = _dev_quetzal_spec("Qwen3.6-27B")
    qz_runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
    )
    with pytest.raises(ValueError, match="requires --quetzal-models-root"):
        generate_docker_run_command(qz_spec, qz_runtime)

    native_spec, native_impl, engine = get_runtime_model_spec(
        model="Qwen3.6-27B", device="p300x2"
    )
    native_runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl=native_impl,
        engine=engine,
        quetzal_models_root=str(tmp_path),
    )
    with pytest.raises(ValueError, match="only valid with --impl quetzal"):
        generate_docker_run_command(native_spec, native_runtime)


def test_docker_command_mounts_exact_v2_auxiliary_root_readonly(tmp_path):
    spec = copy.deepcopy(_dev_quetzal_spec("Qwen3.6-27B"))
    package_id = "sha256-v2-" + "1" * 64 + "-" + "2" * 64 + "-" + "3" * 64
    package = tmp_path / package_id
    package.mkdir()
    (package / "qualification_manifest.yaml").write_text("models: []\n")
    name = "openai_gpt-oss-120b-streamed-cache"
    spec.env_vars["QUETZAL_PACKAGE_ID"] = package_id
    spec.env_vars["QUETZAL_REQUIRED_AUXILIARY_NAMES"] = name
    cache = tmp_path / ("sha256-" + "4" * 64)
    cache.mkdir()
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        quetzal_models_root=str(package),
        quetzal_auxiliary_roots=[f"{name}={cache}"],
    )

    command, _ = generate_docker_run_command(spec, runtime)

    destination = f"/home/container_app_user/quetzal/auxiliary/{name}/{cache.name}"
    assert f"type=bind,src={cache.resolve()},dst={destination},readonly" in command
    expected_json = json.dumps({name: destination}, separators=(",", ":"))
    assert f"QUETZAL_AUXILIARY_ROOTS_JSON={expected_json}" in command

    runtime.quetzal_auxiliary_roots = []
    with pytest.raises(ValueError, match="do not match the model spec"):
        generate_docker_run_command(spec, runtime)


def test_gpt_streamed_expert_paths_stay_beneath_exact_auxiliary_mount():
    spec = _dev_quetzal_spec("gpt-oss-120b")
    name = spec.env_vars["QUETZAL_REQUIRED_AUXILIARY_NAMES"]
    auxiliary_roots = json.loads(spec.env_vars["QUETZAL_AUXILIARY_ROOTS_JSON"])

    assert set(auxiliary_roots) == {name}
    mounted_root = PurePosixPath(auxiliary_roots[name])
    assert mounted_root == PurePosixPath(
        "/home/container_app_user/quetzal/auxiliary/"
        "openai_gpt-oss-120b-streamed-cache/"
        "sha256-2b2e528a75cae51a53db4a3e309f075553fe5f5f7fec7d2a29480f6572f2e416"
    )

    expected_paths = {
        "QZ_MOE_STREAMED_EXPERT_CACHE_ROOT": mounted_root / "cache",
        "QZ_MOE_STREAMED_EXPERT_MANIFEST": mounted_root / "manifest/final.json",
    }
    for env_name, expected_path in expected_paths.items():
        configured_path = PurePosixPath(spec.env_vars[env_name])
        assert configured_path.is_relative_to(mounted_root)
        assert configured_path == expected_path


def test_docker_command_mounts_exact_runtime_attestation_readonly(tmp_path):
    spec = copy.deepcopy(_dev_quetzal_spec("Qwen3.6-27B"))
    package_id = spec.env_vars["QUETZAL_PACKAGE_ID"]
    package = tmp_path / package_id
    package.mkdir()
    (package / "qualification_manifest.yaml").write_text("models: []\n")
    payload = b'{"schema":"ttq.runtime_compatibility_attestation/v1"}\n'
    digest = hashlib.sha256(payload).hexdigest()
    attestation = tmp_path / f"{digest}.json"
    attestation.write_bytes(payload)
    attestation.chmod(0o444)
    spec.env_vars["QUETZAL_RUNTIME_ATTESTATION_SHA256"] = digest
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="server",
        device="p300x2",
        impl="quetzal",
        engine="vLLM",
        docker_server=True,
        quetzal_models_root=str(package),
        quetzal_runtime_attestation=str(attestation),
    )

    command, _ = generate_docker_run_command(spec, runtime)

    destination = f"/home/container_app_user/quetzal/runtime-attestations/{digest}.json"
    assert (
        f"type=bind,src={attestation.resolve()},dst={destination},readonly" in command
    )
    assert f"QUETZAL_RUNTIME_ATTESTATION_PATH={destination}" in command

    attestation.chmod(0o644)
    command, _ = generate_docker_run_command(spec, runtime)
    assert destination not in command


def test_release_labels_user_sealed_and_unattested_provenance_without_blocking(
    tmp_path,
    caplog,
):
    spec = copy.deepcopy(_dev_quetzal_spec("Qwen3.6-27B"))
    package_id = spec.env_vars["QUETZAL_PACKAGE_ID"]
    package = tmp_path / package_id
    package.mkdir()
    qualification = package / "qualification_manifest.yaml"
    qualification.write_text("models: []\n")
    runtime = RuntimeConfig(
        model="Qwen3.6-27B",
        workflow="release",
        device="p300x2",
        impl="quetzal",
        quetzal_models_root=str(package),
    )

    assert _validate_quetzal_models_root(runtime, spec) == package.resolve()
    assert "[user_sealed]" in caplog.text

    qualification.chmod(0o444)
    package.chmod(0o555)
    try:
        assert _validate_quetzal_models_root(runtime, spec) == package.resolve()
        assert _validate_quetzal_runtime_attestation(runtime, spec) is None
        assert "[unattested]" in caplog.text

        payload = b'{"schema":"ttq.runtime_compatibility_attestation/v1"}\n'
        digest = hashlib.sha256(payload).hexdigest()
        attestation = tmp_path / f"{digest}.json"
        attestation.write_bytes(payload)
        attestation.chmod(0o444)
        spec.env_vars["QUETZAL_RUNTIME_ATTESTATION_SHA256"] = digest
        runtime.quetzal_runtime_attestation = str(attestation)
        assert _validate_quetzal_runtime_attestation(runtime, spec) == (
            attestation.resolve()
        )
    finally:
        package.chmod(0o755)
        qualification.chmod(0o644)


def test_quetzal_docker_hook_uses_clean_named_context_and_patched_runtime():
    dockerfile = (
        get_repo_root_path() / "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile"
    ).read_text()
    assert 'ARG TT_QUETZAL_COMMIT_SHA=""' in dockerfile
    assert "^[0-9a-f]{40}$" in dockerfile
    assert "COPY --from=quetzal_src / /tmp/quetzal-source/" in dockerfile
    assert "TT_METAL_PATCHSET_SHA256" in dockerfile
    assert "TT_METAL_PATCHSET_MANIFEST_SHA256" in dockerfile
    assert "tt_metal_patchset.py" in dockerfile
    assert "uv build --wheel" in dockerfile
    wheel_install = next(
        line
        for line in dockerfile.splitlines()
        if "uv pip install" in line and "quetzal_wheel" in line
    )
    assert "--no-cache-dir" in wheel_install
    assert "--no-deps" in wheel_install
    assert '"${quetzal_wheel}"' in wheel_install
    assert (
        "uv pip install --no-cache-dir --no-deps --no-build-isolation ."
        not in dockerfile
    )
    assert "TT_QUETZAL_COMMIT_SHA_OR_TAG" not in dockerfile
    assert "git clone --filter=blob:none --no-checkout" not in dockerfile
