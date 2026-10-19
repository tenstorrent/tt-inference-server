# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

from workflows.model_spec import DeviceModelSpec
from workflows.run_docker_server import _docker_lifecycle_args
from workflows.workflow_types import DeviceTypes


def _spec(**device_fields):
    return SimpleNamespace(
        device_model_spec=DeviceModelSpec(
            device=DeviceTypes.P150X8,
            max_concurrency=1,
            max_context=4096,
            **device_fields,
        )
    )


def test_defaults_add_nothing():
    assert _docker_lifecycle_args(_spec()) == []


def test_stop_timeout():
    assert _docker_lifecycle_args(_spec(docker_stop_timeout=30)) == [
        "--stop-timeout",
        "30",
    ]


def test_host_state_dir_mounted_when_present(tmp_path):
    args = _docker_lifecycle_args(_spec(docker_host_state_dir=str(tmp_path)))
    assert args == ["--mount", f"type=bind,src={tmp_path},dst={tmp_path}"]


def test_host_state_dir_skipped_when_missing(tmp_path):
    missing = tmp_path / "absent"
    assert _docker_lifecycle_args(_spec(docker_host_state_dir=str(missing))) == []
