# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.quetzal_package import (
    rebase_quetzal_package_env,
    resolve_quetzal_package_mount,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.run_local_server import generate_local_run_command
from workflows.runtime_config import RuntimeConfig


CATALOG_ROOT = Path("/home/container_app_user/cache_root/quetzal/packages/sha256-test")
PACKAGE_PATHS = {
    "QUETZAL_PACKAGE_ROOT": str(CATALOG_ROOT),
    "QZ_MODELS_ROOT": str(CATALOG_ROOT),
    "QZ_QUALIFICATION_MANIFEST": str(CATALOG_ROOT / "qualification_manifest.yaml"),
    "QUETZAL_PREFILL_GENERATED_PY": str(
        CATALOG_ROOT / "compiled/model/full/prefill/generated.py"
    ),
    "QUETZAL_DECODE_GENERATED_PY": str(
        CATALOG_ROOT / "compiled/model/full/decode/generated.py"
    ),
    "QUETZAL_PREFILL_METADATA_JSON": str(
        CATALOG_ROOT / "compiled/model/full/prefill/metadata.json"
    ),
    "QUETZAL_DECODE_METADATA_JSON": str(
        CATALOG_ROOT / "compiled/model/full/decode/metadata.json"
    ),
    "QUETZAL_WEIGHTS": str(CATALOG_ROOT / "compiled_weights/model/full/weights.pt"),
}


def _make_package(tmp_path):
    package = tmp_path / "sha256-test"
    package.mkdir()
    (package / "manifest.json").write_text("{}")
    return package


def _model_spec(impl_id="quetzal", env_vars=None):
    return SimpleNamespace(
        impl=SimpleNamespace(impl_id=impl_id, impl_name=impl_id),
        env_vars={
            **PACKAGE_PATHS,
            "QUETZAL_BUNDLE_MANIFEST_SHA256": "a" * 64,
            "VLLM_PLUGINS": "quetzal_model_registry,tt",
        }
        if env_vars is None
        else env_vars,
        model_name="TestModel",
        hf_model_repo="test/TestModel",
        inference_engine="vLLM",
        model_type=SimpleNamespace(name="LLM"),
        subdevice_type=None,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/test:latest",
    )


def _runtime(package, *, docker=False, local=False, impl="quetzal"):
    return RuntimeConfig(
        model="TestModel",
        workflow="server",
        device="p300x2",
        impl=impl,
        engine="vLLM",
        docker_server=docker,
        local_server=local,
        quetzal_models_root=str(package),
        tt_metal_home="/opt/tt-metal",
    )


def test_resolve_quetzal_package_mount_requires_selected_bundle(tmp_path):
    spec = _model_spec()
    runtime = _runtime(tmp_path, docker=True)
    runtime.quetzal_models_root = None

    with pytest.raises(ValueError, match="requires --quetzal-models-root"):
        resolve_quetzal_package_mount(spec, runtime)


def test_resolve_quetzal_package_mount_rejects_native_impl(tmp_path):
    package = _make_package(tmp_path)

    with pytest.raises(ValueError, match="only valid with --impl quetzal"):
        resolve_quetzal_package_mount(
            _model_spec(impl_id="tt_transformers"),
            _runtime(package, docker=True, impl="tt-transformers"),
        )


def test_resolve_quetzal_package_mount_rejects_symlink(tmp_path):
    package = _make_package(tmp_path)
    package_link = tmp_path / "package-link"
    package_link.symlink_to(package, target_is_directory=True)

    with pytest.raises(ValueError, match="existing real directory"):
        resolve_quetzal_package_mount(
            _model_spec(), _runtime(package_link, docker=True)
        )


def test_resolve_quetzal_package_mount_requires_bundle_proof(tmp_path):
    package = tmp_path / "unproven-package"
    package.mkdir()

    with pytest.raises(ValueError, match="portable manifest or installed proof"):
        resolve_quetzal_package_mount(_model_spec(), _runtime(package, docker=True))


def test_rebase_quetzal_package_env_rejects_path_outside_catalog_root(tmp_path):
    env_vars = {
        **PACKAGE_PATHS,
        "QUETZAL_WEIGHTS": "/different/package/weights.pt",
    }

    with pytest.raises(ValueError, match="QUETZAL_WEIGHTS must be inside"):
        rebase_quetzal_package_env(env_vars, CATALOG_ROOT, tmp_path)


def test_docker_mounts_package_at_catalog_root_readonly(tmp_path):
    package = _make_package(tmp_path)
    spec = _model_spec()
    runtime = _runtime(package, docker=True)

    command, _ = generate_docker_run_command(spec, runtime)

    expected_mount = f"type=bind,src={package.resolve()},dst={CATALOG_ROOT},readonly"
    assert command.count(expected_mount) == 1
    assert command[command.index("--impl") + 1] == "quetzal"
    env_settings = {
        command[index + 1] for index, value in enumerate(command[:-1]) if value == "-e"
    }
    assert f"QUETZAL_PACKAGE_ROOT={CATALOG_ROOT}" in env_settings
    assert "VLLM_PLUGINS=quetzal_model_registry,tt" in env_settings


def test_local_server_rebases_catalog_paths_to_host_package(tmp_path):
    package = _make_package(tmp_path)
    spec = _model_spec()
    tt_metal_home = tmp_path / "tt-metal"
    runtime = _runtime(package, local=True)
    runtime.tt_metal_home = str(tt_metal_home)
    cache_root = tmp_path / "persistent" / "model-cache"
    setup_config = SimpleNamespace(
        host_model_volume_root=cache_root,
        host_tt_metal_cache_dir=cache_root / "tt_metal_cache" / "cache_TestModel",
        persistent_volume_root=cache_root.parent,
        host_weights_dir=None,
        host_hf_cache=None,
        host_model_weights_snapshot_dir=None,
        host_model_weights_mount_dir=None,
    )
    runtime_json = tmp_path / "runtime.json"
    runtime_json.write_text("{}")

    command, env, _ = generate_local_run_command(
        spec,
        runtime,
        runtime_json,
        setup_config,
        repo_root=tmp_path,
    )

    assert env["QUETZAL_PACKAGE_ROOT"] == str(package.resolve())
    assert env["QUETZAL_WEIGHTS"] == str(
        package.resolve() / "compiled_weights/model/full/weights.pt"
    )
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert command[command.index("--impl") + 1] == "quetzal"
    assert command.count("--quetzal-package-root") == 1
    assert command[command.index("--quetzal-package-root") + 1] == str(
        package.resolve()
    )
