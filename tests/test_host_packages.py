# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
from types import SimpleNamespace

import pytest

from workflows.host_packages import (
    host_packages_provide_weights,
    resolve_host_package_mounts,
)
from workflows.model_spec import DeviceModelSpec
from workflows.run_docker_server import (
    _docker_entrypoint_args,
    _docker_lifecycle_args,
)
from workflows.validate_setup import validate_bind_mount_permissions
from workflows.workflow_types import DeviceTypes

LAUNCHER = ["python3", "/home/container_app_user/app/src/run_glm_tt.py"]


def _spec(**device_fields):
    return SimpleNamespace(
        device_model_spec=DeviceModelSpec(
            device=DeviceTypes.P150X8,
            max_concurrency=1,
            max_context=4096,
            **device_fields,
        )
    )


def _packages(tmp_path, *names):
    packages = {}
    for name in names:
        (tmp_path / name).mkdir()
        packages[f"/home/container_app_user/{name}"] = str(tmp_path / name)
    return packages


def test_no_packages():
    assert resolve_host_package_mounts(_spec()) == []
    assert not host_packages_provide_weights(_spec())
    assert _docker_entrypoint_args(_spec()) == ([], [])


def test_packages_resolve_in_spec_order(tmp_path):
    packages = _packages(tmp_path, "glm-tt", "weights")
    mounts = resolve_host_package_mounts(_spec(docker_host_packages=packages))
    assert mounts == [
        ((tmp_path / "glm-tt").resolve(), "/home/container_app_user/glm-tt"),
        ((tmp_path / "weights").resolve(), "/home/container_app_user/weights"),
    ]
    assert host_packages_provide_weights(_spec(docker_host_packages=packages))


def test_missing_package_dir_fails_loudly(tmp_path):
    spec = _spec(docker_host_packages={"/opt/pkg": str(tmp_path / "absent")})
    with pytest.raises(ValueError, match="does not exist on this host"):
        resolve_host_package_mounts(spec)
    with pytest.raises(ValueError, match="does not exist on this host"):
        _docker_lifecycle_args(spec)


@pytest.mark.parametrize("container_path", ["relative/pkg", "/"])
def test_container_path_must_be_absolute_and_not_root(tmp_path, container_path):
    spec = _spec(docker_host_packages={container_path: str(tmp_path)})
    with pytest.raises(ValueError, match="non-root absolute path"):
        resolve_host_package_mounts(spec)


def test_packages_mount_readonly(tmp_path):
    packages = _packages(tmp_path, "glm-tt")
    assert _docker_lifecycle_args(_spec(docker_host_packages=packages)) == [
        "--mount",
        f"type=bind,src={(tmp_path / 'glm-tt').resolve()},"
        "dst=/home/container_app_user/glm-tt,readonly",
    ]


def test_entrypoint_splits_around_image():
    assert _docker_entrypoint_args(_spec(docker_entrypoint=LAUNCHER)) == (
        ["--entrypoint", "python3"],
        ["/home/container_app_user/app/src/run_glm_tt.py"],
    )
    assert _docker_entrypoint_args(_spec(docker_entrypoint=["/bin/server"])) == (
        ["--entrypoint", "/bin/server"],
        [],
    )


def test_mock_spec_has_no_packages():
    # run.py's tests pass MagicMock specs: an attribute that is not a dict is no package.
    from unittest.mock import MagicMock

    assert not host_packages_provide_weights(MagicMock())
    assert resolve_host_package_mounts(MagicMock()) == []
    assert _docker_entrypoint_args(MagicMock()) == ([], [])


def test_bind_mount_check_reads_packages_without_fixing(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    package.chmod(0o000)
    try:
        args = SimpleNamespace(
            image_user=os.getuid() + 12345,
            host_volume=None,
            host_hf_cache=None,
            host_weights_dir=None,
        )
        with pytest.raises(ValueError, match=r"docker_host_packages\[/opt/pkg\]"):
            validate_bind_mount_permissions(args, None, [(package, "/opt/pkg")])
        assert (package.stat().st_mode & 0o777) == 0
    finally:
        package.chmod(0o755)


@pytest.mark.parametrize("docker_server", [True, False])
def test_main_skips_hf_weights_for_host_packages(monkeypatch, tmp_path, docker_server):
    """run.py's host setup takes a docker spec's host packages as its weight source."""
    import argparse
    from unittest.mock import MagicMock

    import run as entrypoint

    runtime = MagicMock(
        docker_server=docker_server,
        local_server=not docker_server,
        host_volume=str(tmp_path / "storage"),
        host_hf_cache=None,
        host_weights_dir=None,
        image_user="1000",
        workflow="server",
    )
    model = SimpleNamespace(
        model_id="test-model",
        impl=SimpleNamespace(impl_id="tt_lab_glm"),
        device_model_spec=SimpleNamespace(docker_host_packages={"/opt/pkg": "/x"}),
    )
    monkeypatch.setattr(
        entrypoint, "parse_arguments", lambda: argparse.Namespace(requirements_doc=None)
    )
    monkeypatch.setattr(entrypoint, "resolve_runtime", lambda _: (runtime, model))
    for name in (
        "handle_maintenance_args",
        "export_model_specs_json",
        "bootstrap_uv",
        "handle_secrets",
        "setup_run_logger",
        "ensure_readwriteable_dir",
        "validate_setup",
    ):
        monkeypatch.setattr(entrypoint, name, MagicMock())
    monkeypatch.setattr(entrypoint, "get_current_commit_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        entrypoint, "get_default_workflow_root_log_dir", lambda: tmp_path
    )
    monkeypatch.setattr(entrypoint, "format_cli_args_summary", lambda _: "test")
    monkeypatch.setattr(entrypoint, "resolve_quetzal_package_mount", lambda *_: None)

    class SetupReached(Exception):
        pass

    setup = MagicMock(side_effect=SetupReached)
    monkeypatch.setattr(entrypoint, "setup_host", setup)
    with pytest.raises(SetupReached):
        entrypoint.main()
    # Only a docker server mounts the packages; a local server keeps its own weight source.
    assert setup.call_args.kwargs["package_provides_weights"] is docker_server
