# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.quetzal_package import resolve_quetzal_package_mount
from workflows.run_docker_server import generate_docker_run_command
from workflows.run_local_server import generate_local_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.validate_setup import validate_bind_mount_permissions

CATALOG_ROOT = Path("/opt/quetzal/package")
MANIFEST_SHA256 = "a" * 64


def _make_package(tmp_path):
    package = tmp_path / "sha256-test"
    package.mkdir()
    return package


def _model_spec(impl_id="quetzal", env_vars=None):
    return SimpleNamespace(
        impl=SimpleNamespace(impl_id=impl_id, impl_name=impl_id),
        env_vars={
            "QUETZAL_PACKAGE_ROOT": str(CATALOG_ROOT),
            "QUETZAL_BUNDLE_MANIFEST_SHA256": MANIFEST_SHA256,
            "QUETZAL_WEIGHTS": "/legacy/weights.pt",
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


def _make_v2_package(
    tmp_path, *, root_mode=0o555,
    name="openai_gpt-oss-120b-streamed-cache",
):
    digest = "e" * 64
    package_parent = tmp_path / "packages"
    package = package_parent / "sha256-v2-test"
    package.mkdir(parents=True)
    auxiliary = package_parent / "auxiliary" / name / f"sha256-{digest}"
    auxiliary.mkdir(parents=True)
    auxiliary.chmod(root_mode)
    manifest = {
        "schema": "ttq.artifact_bundle/v2",
        "auxiliary_references": [{
            "name": name, "role": "streamed_cache", "sha256": digest,
            "files": [],
        }],
    }
    raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (package / "manifest.json").write_bytes(raw)
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    runtime_root = (
        "/home/container_app_user/quetzal/auxiliary/"
        f"{name}/sha256-{digest}"
    )
    env = {
        "QUETZAL_PACKAGE_ROOT": str(CATALOG_ROOT),
        "QUETZAL_BUNDLE_MANIFEST_SHA256": manifest_sha256,
        "QUETZAL_AUXILIARY_ROOTS_JSON": json.dumps(
            {name: runtime_root}, sort_keys=True, separators=(",", ":")
        ),
        "VLLM_PLUGINS": "quetzal_model_registry,tt",
    }
    return package, auxiliary, env, runtime_root


def _runtime(package, *, docker=False, local=False, impl="quetzal"):
    return RuntimeConfig(
        model="TestModel",
        workflow="server",
        device="p300x2",
        impl=impl,
        engine="vLLM",
        docker_server=docker,
        local_server=local,
        quetzal_package_root=str(package),
        tt_metal_home="/opt/tt-metal",
    )


def test_package_root_is_required_for_quetzal_server(tmp_path):
    runtime = _runtime(tmp_path, docker=True)
    runtime.quetzal_package_root = None

    with pytest.raises(ValueError, match="requires --quetzal-package-root"):
        resolve_quetzal_package_mount(_model_spec(), runtime)


def test_package_root_is_rejected_for_native_impl(tmp_path):
    package = _make_package(tmp_path)

    with pytest.raises(ValueError, match="only valid with --impl quetzal"):
        resolve_quetzal_package_mount(
            _model_spec(impl_id="tt_transformers"),
            _runtime(package, docker=True, impl="tt-transformers"),
        )


def test_package_root_rejects_symlink(tmp_path):
    package = _make_package(tmp_path)
    package_link = tmp_path / "package-link"
    package_link.symlink_to(package, target_is_directory=True)

    with pytest.raises(ValueError, match="existing real directory"):
        resolve_quetzal_package_mount(
            _model_spec(), _runtime(package_link, docker=True)
        )


@pytest.mark.parametrize(
    "env_vars, error",
    [
        ({"QUETZAL_BUNDLE_MANIFEST_SHA256": MANIFEST_SHA256}, "absolute"),
        (
            {
                "QUETZAL_PACKAGE_ROOT": str(CATALOG_ROOT),
                "QUETZAL_BUNDLE_MANIFEST_SHA256": "latest",
            },
            "lowercase SHA-256",
        ),
    ],
)
def test_package_root_requires_catalog_destination_and_digest(
    tmp_path, env_vars, error
):
    package = _make_package(tmp_path)

    with pytest.raises(ValueError, match=error):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env_vars), _runtime(package, docker=True)
        )


def test_docker_mounts_exact_package_readonly_and_exports_only_selection(tmp_path):
    package = _make_package(tmp_path)
    command, _ = generate_docker_run_command(
        _model_spec(), _runtime(package, docker=True)
    )

    assert f"type=bind,src={package.resolve()},dst={CATALOG_ROOT},readonly" in command
    env_settings = {
        command[index + 1] for index, value in enumerate(command[:-1]) if value == "-e"
    }
    assert f"QUETZAL_PACKAGE_ROOT={CATALOG_ROOT}" in env_settings
    assert f"QUETZAL_BUNDLE_MANIFEST_SHA256={MANIFEST_SHA256}" in env_settings
    assert not any(value.startswith("QUETZAL_WEIGHTS=") for value in env_settings)
    assert not any(value.startswith("VLLM_PLUGINS=") for value in env_settings)
    assert command[command.index("--impl") + 1] == "quetzal"


def test_docker_uses_model_spec_impl_when_runtime_json_omits_cli_impl(tmp_path):
    package = _make_package(tmp_path)
    runtime = _runtime(package, docker=True, impl=None)

    command, _ = generate_docker_run_command(_model_spec(), runtime)

    assert command[command.index("--impl") + 1] == "quetzal"


def test_printable_docker_command_contains_readonly_package_mount(tmp_path):
    package = _make_package(tmp_path)
    command, _ = generate_docker_run_command(
        _model_spec(), _runtime(package, docker=True), str_cmd=True
    )

    assert "--mount" in command
    assert f"src={package.resolve()},dst={CATALOG_ROOT},readonly" in command


def test_printable_docker_command_derives_v2_auxiliary_mount(tmp_path):
    package, auxiliary, env, runtime_root = _make_v2_package(tmp_path)
    command, _ = generate_docker_run_command(
        _model_spec(env_vars=env), _runtime(package, docker=True), str_cmd=True
    )
    assert f"src={auxiliary.resolve()},dst={runtime_root},readonly" in command
    assert (
        "QUETZAL_AUXILIARY_ROOTS_JSON=" + env["QUETZAL_AUXILIARY_ROOTS_JSON"]
    ) in command


def test_v2_auxiliary_accepts_canonical_single_segment_edge_name(tmp_path):
    package, auxiliary, env, runtime_root = _make_v2_package(
        tmp_path, name="cache+bf4@rev:1"
    )
    command, _ = generate_docker_run_command(
        _model_spec(env_vars=env), _runtime(package, docker=True), str_cmd=True
    )
    assert f"src={auxiliary.resolve()},dst={runtime_root},readonly" in command


@pytest.mark.parametrize(
    "name",
    [
        "../cache",
        "cache/subdir",
        "cache\\windows",
        "cache,option",
        "cache\ncontrol",
        "cache\u0085control",
    ],
)
def test_v2_auxiliary_rejects_unsafe_or_control_name(tmp_path, name):
    package, _, env, _ = _make_v2_package(tmp_path, name=name)
    with pytest.raises(ValueError, match="unsafe name"):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env), _runtime(package, docker=True)
        )


def test_v2_auxiliary_root_must_exist(tmp_path):
    package, auxiliary, env, _ = _make_v2_package(tmp_path)
    auxiliary.chmod(0o755)
    auxiliary.rename(auxiliary.with_name("unpublished"))
    with pytest.raises(ValueError, match="absent or unpublished"):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env), _runtime(package, docker=True)
        )


def test_v2_auxiliary_root_must_not_be_symlink(tmp_path):
    package, auxiliary, env, _ = _make_v2_package(tmp_path)
    auxiliary.chmod(0o755)
    real = auxiliary.with_name("real")
    auxiliary.rename(real)
    auxiliary.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="absent or unpublished"):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env), _runtime(package, docker=True)
        )


def test_v2_auxiliary_root_must_be_read_only(tmp_path):
    package, _, env, _ = _make_v2_package(tmp_path, root_mode=0o755)
    with pytest.raises(ValueError, match="read-only real directory"):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env), _runtime(package, docker=True)
        )


@pytest.mark.parametrize(
    ("root_mode", "missing_access"),
    [(0o111, "read permission"), (0o444, "execute/traverse permission")],
)
def test_v2_auxiliary_root_must_be_accessible_to_container_uid(
    tmp_path, root_mode, missing_access
):
    package, auxiliary, env, _ = _make_v2_package(tmp_path, root_mode=root_mode)
    runtime = _runtime(package, docker=True)
    runtime.image_user = str(os.getuid())
    mount = resolve_quetzal_package_mount(_model_spec(env_vars=env), runtime)

    try:
        with pytest.raises(ValueError, match="Bind mount permission check failed") as error:
            validate_bind_mount_permissions(runtime, mount)
        assert missing_access in str(error.value)
        # Admission must not mutate an immutable published auxiliary to fix it.
        assert auxiliary.stat().st_mode & 0o777 == root_mode
    finally:
        auxiliary.chmod(0o755)


def test_v2_auxiliary_mapping_is_required(tmp_path):
    package, _, env, _ = _make_v2_package(tmp_path)
    env.pop("QUETZAL_AUXILIARY_ROOTS_JSON")
    with pytest.raises(ValueError, match="requires generated"):
        resolve_quetzal_package_mount(
            _model_spec(env_vars=env), _runtime(package, docker=True)
        )


def test_local_server_selects_host_package_without_asset_fanout(tmp_path, monkeypatch):
    monkeypatch.delenv("QUETZAL_WEIGHTS", raising=False)
    package = _make_package(tmp_path)
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
        _model_spec(), runtime, runtime_json, setup_config, repo_root=tmp_path
    )

    assert env["QUETZAL_PACKAGE_ROOT"] == str(package.resolve())
    assert env["QUETZAL_BUNDLE_MANIFEST_SHA256"] == MANIFEST_SHA256
    assert "QUETZAL_WEIGHTS" not in env
    assert command[command.index("--impl") + 1] == "quetzal"
    assert "--quetzal-package-root" not in command


def test_local_server_uses_model_spec_impl_when_runtime_json_omits_cli_impl(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("QUETZAL_WEIGHTS", raising=False)
    package = _make_package(tmp_path)
    runtime = _runtime(package, local=True, impl=None)
    runtime.tt_metal_home = str(tmp_path / "tt-metal")
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

    command, _, _ = generate_local_run_command(
        _model_spec(), runtime, runtime_json, setup_config, repo_root=tmp_path
    )

    assert command[command.index("--impl") + 1] == "quetzal"
