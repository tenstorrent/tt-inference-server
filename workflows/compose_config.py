# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Docker Compose variable resolution for tt-inference-server.

Resolves variables from ModelSpec/RuntimeConfig/SetupConfig into a flat dict
that populates the static compose templates in deploy/.

The compose templates use ${VAR} and ${VAR:-default} syntax. This module
provides the variable values. Users who don't use run.py can set these
variables manually in .env and run `docker compose up` directly.

Compose templates:
- deploy/docker-compose.vllm.yml         — vLLM models
- deploy/docker-compose.media.yml        — media/forge models
- deploy/docker-compose.multihost-worker.yml     — MPI worker
- deploy/docker-compose.multihost-controller.yml — MPI controller
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

from workflows.utils import get_repo_root_path
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
)

logger = logging.getLogger("run_log")

DEPLOY_DIR = get_repo_root_path() / "deploy"


def _short_uuid():
    return str(uuid.uuid4())[:8]


def get_compose_template_path(model_spec, runtime_config) -> Path:
    """Return the path to the appropriate compose template.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object

    Returns:
        Path to the compose template YAML file
    """
    if model_spec.inference_engine == InferenceEngine.VLLM.value:
        return DEPLOY_DIR / "docker-compose.vllm.yml"
    else:
        return DEPLOY_DIR / "docker-compose.media.yml"


def resolve_compose_vars(
    model_spec,
    runtime_config,
    setup_config=None,
) -> dict:
    """Resolve compose template variables from config objects.

    Returns a flat dict of VAR=value pairs that can be written to .env
    or exported as environment variables for `docker compose up`.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: Optional SetupConfig for volume/weight paths

    Returns:
        Dict of environment variable name → value
    """
    container_name = f"tt-inference-server-{_short_uuid()}"

    env = {
        # Common
        "DOCKER_IMAGE": model_spec.docker_image,
        "CONTAINER_NAME": container_name,
        "SERVICE_PORT": str(runtime_config.service_port),
        "BIND_HOST": runtime_config.bind_host,
    }

    # vLLM-specific
    if model_spec.inference_engine == InferenceEngine.VLLM.value:
        env["HF_MODEL"] = model_spec.hf_model_repo
        env["TT_DEVICE"] = runtime_config.device

    # Media/Forge-specific
    if model_spec.inference_engine in (
        InferenceEngine.MEDIA.value,
        InferenceEngine.FORGE.value,
    ):
        env["MODEL"] = model_spec.model_name
        env["DEVICE"] = model_spec.device_type.name.lower()

    # Volume configuration
    if setup_config:
        if setup_config.host_model_volume_root:
            env["CACHE_VOLUME"] = str(setup_config.host_model_volume_root)
        else:
            from workflows.run_docker_server import generate_docker_volume_name

            env["CACHE_VOLUME"] = generate_docker_volume_name(model_spec)

        env["CACHE_ROOT"] = str(setup_config.cache_root)

        if (
            setup_config.container_model_weights_path
            and setup_config.host_model_weights_mount_dir
        ):
            env["MODEL_WEIGHTS_DIR"] = str(
                setup_config.container_model_weights_path
            )

    return env


def resolve_multihost_vars(
    model_spec,
    runtime_config,
    multihost_config,
    setup=None,
) -> dict:
    """Resolve compose variables for multi-host deployment.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        multihost_config: MultiHostConfig with hosts, interfaces, paths
        setup: Optional MultiHostSetup with generated config paths

    Returns:
        Dict of environment variable name → value
    """
    device = DeviceTypes.from_string(runtime_config.device)

    env = {
        "DOCKER_IMAGE": model_spec.docker_image,
        "HF_MODEL": model_spec.hf_model_repo,
        "TT_DEVICE": runtime_config.device,
        "SHARED_STORAGE_ROOT": multihost_config.shared_storage_root,
        "MULTIHOST_HOSTS": ",".join(multihost_config.hosts),
        "MPI_INTERFACE": multihost_config.mpi_interface,
        "MESH_DEVICE": device.to_mesh_device_str(),
        "SSH_PORT": str(multihost_config.ssh_port),
    }

    if setup:
        env["SSH_CONFIG_DIR"] = str(setup.config_dir)
        env["MPIRUN_DIR"] = str(setup.config_dir / "mpirun")

    return env


def format_env_for_display(env_vars: dict) -> str:
    """Format environment variables for human-readable display.

    Args:
        env_vars: Dict from resolve_compose_vars() or resolve_multihost_vars()

    Returns:
        Formatted string with one VAR=value per line
    """
    lines = []
    for key, value in env_vars.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines)
