# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Docker Compose configuration generation for tt-inference-server.

Generates Docker Compose service definitions as Python dicts for all
inference engines (vLLM, media, forge) and deployment modes (single-node,
multi-host).

The generated configs can be:
- Written to YAML files (docker-compose.yml)
- Used directly by `docker compose up -d`
- Printed for inspection via `--print-compose`
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import yaml

from workflows.utils import default_dotenv_path, get_repo_root_path
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelType,
)

logger = logging.getLogger("run_log")


def _short_uuid():
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Single-node compose generation
# ---------------------------------------------------------------------------


def generate_compose_config(
    model_spec,
    runtime_config,
    setup_config=None,
    json_fpath=None,
) -> dict:
    """Generate Docker Compose config dict for single-node deployment.

    Translates the same logic as generate_docker_run_command() into a
    declarative compose service definition.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: Optional SetupConfig for host volume/weight mounts
        json_fpath: Optional path to runtime model spec JSON file

    Returns:
        Dict suitable for yaml.dump() as a docker-compose.yml
    """
    device = DeviceTypes.from_string(runtime_config.device)
    mesh_device_str = device.to_mesh_device_str()
    container_name = f"tt-inference-server-{_short_uuid()}"

    # Device cache dir (for TT_CACHE_PATH)
    device_cache_dir = (
        DeviceTypes.to_mesh_device_str(model_spec.subdevice_type)
        if model_spec.subdevice_type
        else mesh_device_str
    )

    # Device mapping
    device_path = "/dev/tenstorrent"
    if not runtime_config.device_id:
        devices = [f"{device_path}:{device_path}"]
    else:
        devices = [
            f"{device_path}/{d}:{device_path}/{d}"
            for d in runtime_config.device_id
        ]

    # Base service definition
    service = {
        "image": model_spec.docker_image,
        "container_name": container_name,
        "ipc": "host",
        "devices": devices,
        "volumes": [
            "/dev/hugepages-1G:/dev/hugepages-1G",
        ],
        "env_file": [str(default_dotenv_path)],
        "ports": [
            f"{runtime_config.bind_host}:{runtime_config.service_port}:{runtime_config.service_port}"
        ],
    }

    # User override (only if not default 1000)
    if (
        runtime_config.image_user
        and str(runtime_config.image_user) != "1000"
    ):
        service["user"] = str(runtime_config.image_user)

    # Environment variables
    environment = {}

    # Setup config mounts (cache root volume)
    if setup_config:
        if setup_config.host_model_volume_root:
            service["volumes"].append(
                f"{setup_config.host_model_volume_root}:{setup_config.cache_root}"
            )
        else:
            # Named volume
            from workflows.run_docker_server import generate_docker_volume_name

            volume_name = generate_docker_volume_name(model_spec)
            service["volumes"].append(
                f"{volume_name}:{setup_config.cache_root}"
            )

        # Weights mount
        if setup_config.host_model_weights_mount_dir:
            service["volumes"].append(
                f"{setup_config.host_model_weights_mount_dir}:{setup_config.container_model_weights_mount_dir}:ro"
            )

        # Environment: MODEL_WEIGHTS_DIR
        if (
            setup_config.container_model_weights_path
            and setup_config.host_model_weights_mount_dir
        ):
            environment["MODEL_WEIGHTS_DIR"] = str(
                setup_config.container_model_weights_path
            )

        # Environment: TT_CACHE_PATH
        if (
            setup_config.host_model_volume_root
            and setup_config.container_tt_metal_cache_dir
        ):
            environment["TT_CACHE_PATH"] = str(
                setup_config.container_tt_metal_cache_dir / device_cache_dir
            )

    # Engine-specific config
    if (
        model_spec.inference_engine == InferenceEngine.FORGE.value
        or model_spec.inference_engine == InferenceEngine.MEDIA.value
    ):
        environment["CACHE_ROOT"] = "/home/container_app_user/cache_root"
        environment["MODEL"] = model_spec.model_name
        environment["DEVICE"] = model_spec.device_type.name.lower()

    # Dev mode mounts
    user_home_path = "/home/container_app_user"
    if runtime_config.dev_mode:
        repo_root_path = get_repo_root_path()

        if json_fpath:
            container_model_spec_dir = Path(f"{user_home_path}/model_specs")
            runtime_json_fpath = container_model_spec_dir / json_fpath.name
            service["volumes"].append(
                f"{json_fpath}:{runtime_json_fpath}:ro"
            )
            environment["RUNTIME_MODEL_SPEC_JSON_PATH"] = str(
                runtime_json_fpath
            )

        service["volumes"].extend(
            [
                f"{repo_root_path}/benchmarking:{user_home_path}/app/benchmarking",
                f"{repo_root_path}/evals:{user_home_path}/app/evals",
                f"{repo_root_path}/utils:{user_home_path}/app/utils",
                f"{repo_root_path}/tests:{user_home_path}/app/tests",
            ]
        )

        if model_spec.model_type in (
            ModelType.CNN,
            ModelType.IMAGE,
            ModelType.EMBEDDING,
            ModelType.VIDEO,
            ModelType.TEXT_TO_SPEECH,
            ModelType.AUDIO,
        ):
            service["volumes"].append(
                f"{repo_root_path}/tt-media-server:{user_home_path}/tt-metal/server"
            )
        else:
            service["volumes"].append(
                f"{repo_root_path}/vllm-tt-metal/src:{user_home_path}/app/src"
            )

    # Metal timeout
    if runtime_config.disable_metal_timeout:
        environment["DISABLE_METAL_OP_TIMEOUT"] = "1"

    # Set environment on service
    if environment:
        service["environment"] = environment

    # Command: vLLM gets explicit command with flags, media/forge use Dockerfile CMD
    if runtime_config.interactive:
        service["stdin_open"] = True
        service["tty"] = True
        service["command"] = ["bash", "-c", "sleep infinity"]
    elif model_spec.inference_engine == InferenceEngine.VLLM.value:
        command = [
            "python",
            "run_vllm_api_server.py",
            "--model",
            model_spec.hf_model_repo,
            "--tt-device",
            runtime_config.device,
        ]
        if runtime_config.no_auth:
            command.append("--no-auth")
        if runtime_config.disable_trace_capture:
            command.append("--disable-trace-capture")
        if (
            runtime_config.service_port
            and str(runtime_config.service_port) != "8000"
        ):
            command.extend(
                ["--service-port", str(runtime_config.service_port)]
            )
        service["command"] = command

    # Healthcheck
    service["healthcheck"] = {
        "test": [
            "CMD",
            "curl",
            "-sf",
            f"http://localhost:{runtime_config.service_port}/health",
        ],
        "interval": "30s",
        "timeout": "10s",
        "retries": 3,
        "start_period": "600s",
    }

    compose = {
        "services": {
            "inference-server": service,
        },
    }

    return compose, container_name


# ---------------------------------------------------------------------------
# Multi-host compose generation
# ---------------------------------------------------------------------------


def generate_multihost_worker_compose(
    docker_image: str,
    shared_storage_root: str,
    rank: int,
) -> dict:
    """Generate Docker Compose config for a multi-host Worker.

    Args:
        docker_image: Docker image to use
        shared_storage_root: Shared storage root path
        rank: MPI rank for this worker

    Returns:
        Dict suitable for yaml.dump() as a docker-compose.yml
    """
    from workflows.multihost_config import WORKER_SSH_PORT

    container_name = f"tt-worker-{rank}-{_short_uuid()}"
    remote_public_key_path = "/tmp/authorized_keys.pub"

    service = {
        "image": docker_image,
        "container_name": container_name,
        "network_mode": "host",
        "pid": "host",
        "user": "root",
        "entrypoint": "/usr/local/bin/multihost_entrypoint.sh",
        "devices": ["/dev/tenstorrent:/dev/tenstorrent"],
        "volumes": [
            "/dev/hugepages-1G:/dev/hugepages-1G",
            f"{shared_storage_root}:{shared_storage_root}",
            f"{remote_public_key_path}:/tmp/authorized_keys.pub:ro",
        ],
        "environment": {
            "MULTIHOST_ROLE": "worker",
            "SSH_PORT": str(WORKER_SSH_PORT),
        },
        "restart": "unless-stopped",
        "healthcheck": {
            "test": [
                "CMD",
                "ssh-keyscan",
                "-p",
                str(WORKER_SSH_PORT),
                "localhost",
            ],
            "interval": "5s",
            "retries": 10,
            "start_period": "30s",
        },
    }

    compose = {
        "services": {
            "worker": service,
        },
    }

    return compose, container_name


def generate_multihost_controller_compose(
    model_spec,
    runtime_config,
    setup_config,
    multihost_setup,
    hosts: list[str],
    mpi_interface: str,
    shared_storage_root: str,
    runtime_model_spec_json_path: Optional[Path] = None,
) -> dict:
    """Generate Docker Compose config for a multi-host Controller.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: SetupConfig from setup_host()
        multihost_setup: MultiHostSetup with config paths
        hosts: List of hostnames
        mpi_interface: MPI network interface
        shared_storage_root: Shared storage root path
        runtime_model_spec_json_path: Path to merged model spec JSON

    Returns:
        Dict suitable for yaml.dump() as a docker-compose.yml
    """
    container_name = f"tt-controller-{_short_uuid()}"
    device = DeviceTypes.from_string(runtime_config.device)
    mesh_device = device.to_mesh_device_str()
    container_model_spec_path = "/tmp/runtime_model_spec.json"

    service = {
        "image": model_spec.docker_image,
        "container_name": container_name,
        "network_mode": "host",
        "pid": "host",
        "ipc": "host",
        "user": "root",
        "entrypoint": "/usr/local/bin/multihost_entrypoint.sh",
        "env_file": [str(default_dotenv_path)],
        "devices": ["/dev/tenstorrent:/dev/tenstorrent"],
        "volumes": [
            "/dev/hugepages-1G:/dev/hugepages-1G",
            f"{multihost_setup.config_dir}:/tmp/ssh_config:ro",
            f"{multihost_setup.config_dir / 'mpirun'}:/etc/mpirun:ro",
            f"{shared_storage_root}:{shared_storage_root}",
        ],
        "environment": {
            "MULTIHOST_ROLE": "controller",
            "SSH_CONFIG_SRC": "/tmp/ssh_config",
            "MESH_DEVICE": mesh_device,
            "VLLM_TARGET_DEVICE": "tt",
            "NCCL_SOCKET_IFNAME": mpi_interface,
            "ETH": mpi_interface,
            "HOSTS": ",".join(hosts),
        },
    }

    # Mount runtime model spec JSON
    if runtime_model_spec_json_path:
        service["volumes"].append(
            f"{runtime_model_spec_json_path}:{container_model_spec_path}:ro"
        )
        service["environment"]["RUNTIME_MODEL_SPEC_JSON_PATH"] = (
            container_model_spec_path
        )

    # Cache volume mounts
    if setup_config:
        if setup_config.host_model_volume_root:
            service["volumes"].append(
                f"{setup_config.host_model_volume_root}:{setup_config.cache_root}"
            )
        else:
            from workflows.run_docker_server import generate_docker_volume_name

            volume_name = generate_docker_volume_name(model_spec)
            service["volumes"].append(
                f"{volume_name}:{setup_config.cache_root}"
            )

        if setup_config.host_model_weights_mount_dir:
            service["volumes"].append(
                f"{setup_config.host_model_weights_mount_dir}:{setup_config.container_model_weights_mount_dir}:ro"
            )
            service["environment"]["MODEL_WEIGHTS_DIR"] = str(
                setup_config.container_model_weights_path
            )

    # vLLM command
    command = [
        "python",
        "run_vllm_api_server.py",
        "--model",
        model_spec.hf_model_repo,
        "--tt-device",
        runtime_config.device,
    ]

    if runtime_config.no_auth:
        command.append("--no-auth")
    if runtime_config.disable_trace_capture:
        command.append("--disable-trace-capture")
    if (
        runtime_config.service_port
        and str(runtime_config.service_port) != "8000"
    ):
        command.extend(["--service-port", str(runtime_config.service_port)])

    service["command"] = command

    compose = {
        "services": {
            "controller": service,
        },
    }

    return compose, container_name


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_compose_file(compose_config: dict, output_path: Path) -> Path:
    """Write compose config dict to a YAML file.

    Args:
        compose_config: Dict from generate_compose_config() or similar
        output_path: Path to write the YAML file

    Returns:
        Path to the written file
    """
    with open(output_path, "w") as f:
        yaml.dump(
            compose_config,
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    logger.info(f"Wrote Docker Compose file: {output_path}")
    return output_path


def format_compose_yaml(compose_config: dict) -> str:
    """Format compose config as a YAML string for display.

    Args:
        compose_config: Dict from generate_compose_config() or similar

    Returns:
        YAML-formatted string
    """
    return yaml.dump(
        compose_config,
        default_flow_style=False,
        sort_keys=False,
    )
