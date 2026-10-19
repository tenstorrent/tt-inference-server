# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GLM-5.3-Flash (tt_lab_glm) catalog rows and the docker command they produce."""

import dataclasses
import os

import pytest

from workflows.model_spec import (
    _MODEL_SPECS_DIR,
    get_model_spec_map,
    load_templates_from_yaml,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig

GLM_GALAXY_ID = "id_tt-lab-glm_GLM-5.3-Flash_blackhole_galaxy"
# The GLM-5.3-Flash rows are dev-only, as CI's run.py --dev-mode resolves them.
MODEL_SPECS = get_model_spec_map(
    load_templates_from_yaml(_MODEL_SPECS_DIR / "dev" / "llm.yaml", env="dev")
)
LAUNCHER = ["python3", "/home/container_app_user/app/src/run_glm_tt.py"]


def _galaxy_spec(tmp_path):
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    packages = {}
    for container_path in spec.device_model_spec.docker_host_packages:
        host = tmp_path / container_path.strip("/").replace("/", "_")
        host.mkdir()
        packages[container_path] = str(host)
    device = dataclasses.replace(spec.device_model_spec, docker_host_packages=packages)
    return dataclasses.replace(spec, device_model_spec=device), packages


def _galaxy_runtime():
    return RuntimeConfig(
        model="GLM-5.3-Flash",
        workflow="server",
        device="blackhole_galaxy",
        docker_server=True,
    )


def test_glm_galaxy_catalog_row():
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    device = spec.device_model_spec
    assert device.max_concurrency == 4
    assert device.max_context == 262144
    assert device.docker_entrypoint == LAUNCHER
    assert set(device.docker_host_packages) == {
        "/home/container_app_user/glm-tt",
        "/home/container_app_user/glm-weights/ttq",
        "/home/container_app_user/glm-weights/vocab",
    }
    for name in ("GLM_TT_HOME", "GLM_TT_TTQ", "GLM_TT_VOCAB"):
        assert any(
            spec.env_vars[name].startswith(path) for path in device.docker_host_packages
        ), name


def test_glm_galaxy_docker_command(tmp_path):
    spec, packages = _galaxy_spec(tmp_path)
    command, _ = generate_docker_run_command(spec, _galaxy_runtime())

    for container_path, host in packages.items():
        mount = f"type=bind,src={os.path.realpath(host)},dst={container_path},readonly"
        assert command[command.index(mount) - 1] == "--mount"
    image = command.index(spec.docker_image)
    assert command[command.index("--entrypoint") + 1] == "python3"
    assert command.index("--entrypoint") < image
    # MEDIA passes no server arguments: the launcher path is the container's only one.
    assert command[image + 1 :] == LAUNCHER[1:]
    env = {command[i + 1] for i, value in enumerate(command[:-1]) if value == "-e"}
    assert "DEVICE=blackhole_galaxy" in env
    assert f"GLM_TT_TTQ={spec.env_vars['GLM_TT_TTQ']}" in env
    assert "GLM_TT_HOME=/home/container_app_user/glm-tt" in env


def test_glm_galaxy_docker_command_needs_its_packages(tmp_path):
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    device = dataclasses.replace(
        spec.device_model_spec,
        docker_host_packages={"/home/container_app_user/glm-tt": str(tmp_path / "no")},
    )
    spec = dataclasses.replace(spec, device_model_spec=device)
    with pytest.raises(ValueError, match="does not exist on this host"):
        generate_docker_run_command(spec, _galaxy_runtime())
