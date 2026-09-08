# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from pathlib import Path
from types import SimpleNamespace

import pytest

from workflows.quetzal_package import resolve_quetzal_package_mount
from workflows.run_docker_server import generate_docker_run_command
from workflows.run_local_server import generate_local_run_command
from workflows.runtime_config import RuntimeConfig

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
