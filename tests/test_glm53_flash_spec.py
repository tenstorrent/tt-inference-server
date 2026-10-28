# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GLM-5.3-Flash (tt_lab_glm) catalog rows and the docker command they produce."""

import dataclasses
from pathlib import Path

import pytest

from workflows.host_packages import host_packages_provide_weights
from workflows.model_spec import (
    _MODEL_SPECS_DIR,
    get_model_spec_map,
    load_templates_from_yaml,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.setup_host import HostSetupManager

GLM_GALAXY_ID = "id_tt-lab-glm_GLM-5.3-Flash_blackhole_galaxy"
# The GLM-5.3-Flash rows are dev-only, as CI's run.py --dev-mode resolves them.
MODEL_SPECS = get_model_spec_map(
    load_templates_from_yaml(_MODEL_SPECS_DIR / "dev" / "llm.yaml", env="dev")
)
LAUNCHER = ["python3", "/home/container_app_user/app/src/run_glm_tt.py"]


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
    # Two users of the full 262144-position context per group, four groups: none refused.
    assert device.max_concurrency == 8
    assert device.max_context == 262144
    extra = device.env_vars["GLM_TT_EXTRA_ARGS"].split()
    assert extra[extra.index("--users") + 1] == "2"
    assert extra[extra.index("--context") + 1] == "262144"
    assert extra[extra.index("--session-positions") + 1] == "262144"
    # Ethernet groups (the launcher's defaults), not the PCIe overrides of the first CI runs.
    assert "TT_LAB_GLM_ROUND" not in device.env_vars
    assert "TT_LAB_ETH_GATHER" not in device.env_vars
    assert device.env_vars["GLM_TT_MTP"] in spec.metadata["checkpoint_files"]
    assert device.docker_entrypoint == LAUNCHER
    # Weights and the glm-tt package come from the weights repo, not host packages.
    assert spec.hf_weights_repo
    assert not host_packages_provide_weights(spec)
    assert spec.min_disk_gb >= 250
    files = spec.metadata["checkpoint_files"]
    package = spec.env_vars["GLM_TT_PACKAGE"]
    assert package.startswith("glm-tt-")
    assert f"{package}/bin/glm-tt" in files
    assert f"{spec.env_vars['GLM_TT_TTQ']}.manifest" in files
    assert spec.env_vars["GLM_TT_VOCAB"] in files
    assert "GLM_TT_HOME" not in spec.env_vars


def test_glm_galaxy_weights_land_where_the_launcher_looks(tmp_path):
    """TIS downloads the repo to cache_root/weights/<model_name>; the launcher reads
    $CACHE_ROOT/weights/$MODEL, from the env vars the docker command sets."""
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    manager = HostSetupManager(model_spec=spec, host_volume=str(tmp_path))
    config = manager.setup_config
    command, _ = generate_docker_run_command(spec, _galaxy_runtime(), config)
    env = dict(
        command[i + 1].split("=", 1)
        for i, value in enumerate(command[:-1])
        if value == "-e"
    )
    assert config.container_model_weights_path == (
        Path(env["CACHE_ROOT"]) / "weights" / env["MODEL"]
    )
    mount = f"type=bind,src={config.host_model_volume_root},dst={config.cache_root}"
    assert command[command.index(mount) - 1] == "--mount"

    weights = config.host_model_volume_root / "weights" / spec.model_name
    assert not manager.check_model_weights_dir(weights)
    weights.mkdir(parents=True)
    files = spec.metadata["checkpoint_files"]
    for name in files:
        (weights / name).parent.mkdir(parents=True, exist_ok=True)
        (weights / name).write_bytes(b"x")
    assert manager.check_model_weights_dir(weights)
    (weights / files[-1]).unlink()  # one file short: download again
    assert not manager.check_model_weights_dir(weights)


def test_glm_galaxy_docker_command():
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    command, _ = generate_docker_run_command(spec, _galaxy_runtime())
    image = command.index(spec.docker_image)
    assert command[command.index("--entrypoint") + 1] == "python3"
    assert command.index("--entrypoint") < image
    # MEDIA passes no server arguments: the launcher path is the container's only one.
    assert command[image + 1 :] == LAUNCHER[1:]
    env = {command[i + 1] for i, value in enumerate(command[:-1]) if value == "-e"}
    assert "DEVICE=blackhole_galaxy" in env
    assert "GLM_TT_TTQ=glm53-flash-plan0h.ttq" in env
    assert "GLM_TT_MTP=glm53-flash-mtp-plan0h.ttq" in env
    assert not any(
        value.endswith(",readonly") and "glm-tt" in value for value in command
    )


def test_glm_galaxy_docker_command_needs_its_packages(tmp_path):
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    device = dataclasses.replace(
        spec.device_model_spec,
        docker_host_packages={"/home/container_app_user/glm-tt": str(tmp_path / "no")},
    )
    spec = dataclasses.replace(spec, device_model_spec=device)
    with pytest.raises(ValueError, match="does not exist on this host"):
        generate_docker_run_command(spec, _galaxy_runtime())
