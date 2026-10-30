# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The GLM-5.3-Flash (tt_lab_glm) catalog rows and the docker command they produce."""

import dataclasses
from pathlib import Path

import pytest

from workflows.helm_generator.cli import DEFAULT_VALUES_PATH, filter_specs, generate
from workflows.host_packages import host_packages_provide_weights
from workflows.model_spec import (
    _MODEL_SPECS_DIR,
    get_model_spec_map,
    load_templates_from_yaml,
)
from workflows.run_docker_server import generate_docker_run_command
from workflows.runtime_config import RuntimeConfig
from workflows.setup_host import HostSetupManager
from workflows.utils import get_repo_root_path

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
    assert spec.min_disk_gb >= 190  # plan0h 181 GB, MTP 4.3 GB, package
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


def test_glm_galaxy_helm_values_in_sync_with_the_dev_spec(tmp_path):
    """values.yaml's GLM-5.3-Flash entry is generated from this dev row (the chart's
    test_values_in_sync regenerates from the prod catalog only, which keeps it as is):
    MODEL_SPECS_ENV=dev python -m workflows.helm_generator --model GLM-5.3-Flash."""
    work = tmp_path / "values.yaml"
    work.write_text(DEFAULT_VALUES_PATH.read_text())
    specs = filter_specs(MODEL_SPECS.values(), model_names=["GLM-5.3-Flash"])
    assert [spec.model_id for spec in specs] == [GLM_GALAXY_ID]
    generate(values_path=work, specs=specs)
    assert work.read_text() == DEFAULT_VALUES_PATH.read_text()


def test_glm_galaxy_helm_values():
    import yaml

    values = yaml.safe_load(DEFAULT_VALUES_PATH.read_text())
    model = values["models"]["GLM-5.3-Flash"]
    assert model["defaultEngine"] == "media"
    impl = model["media"]["blackhole_galaxy"]["impls"]["tt_lab_glm"]
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    assert f"{impl['image']['repository']}:{impl['image']['tag']}" == spec.docker_image
    # glm-tt serve answers /health and /v1/models only; the chart's default /tt-liveness
    # for a media engine would restart a serving pod.
    assert impl["probes"]["liveness"]["path"] == "/health"
    assert impl["resources"]["requests"]["memory"] == f"{spec.min_ram_gb}Gi"
    env = {item["name"]: item["value"] for item in impl["env"]}
    assert (
        env["GLM_TT_EXTRA_ARGS"] == spec.device_model_spec.env_vars["GLM_TT_EXTRA_ARGS"]
    )
    assert env["GLM_TT_PACKAGE"] == spec.env_vars["GLM_TT_PACKAGE"]
    extra = {item["name"]: item["value"] for item in impl["extraEnv"]}
    assert extra["TT_LAB_DIRTY_MARKER"].startswith(
        "/home/container_app_user/cache_root/"
    )


def test_glm_galaxy_image_downloads_the_spec_weights_repo():
    """The image's entrypoint downloads GLM_TT_WEIGHTS_REPO; it must be the spec's."""
    spec = MODEL_SPECS[GLM_GALAXY_ID]
    dockerfile = (
        get_repo_root_path() / "vllm-tt-metal" / "tt-lab-glm.Dockerfile"
    ).read_text()
    assert f"GLM_TT_WEIGHTS_REPO={spec.hf_weights_repo} " in dockerfile
    assert spec.docker_image.startswith(
        "ghcr.io/tenstorrent/tt-inference-server/tt-lab-glm-ubuntu-22.04-amd64@sha256:"
    )
    # The launcher sits where the spec's docker_entrypoint expects it.
    assert "/home/container_app_user/app/src/" in dockerfile
    assert LAUNCHER[1].startswith("/home/container_app_user/app/src/")
