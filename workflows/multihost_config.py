# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Multi-host configuration generation for MPI-based distributed inference.

This module generates configuration files needed for multi-host deployment:
- SSH config for Controller to connect to Worker containers
- MPI rankfile for rank-to-host mapping
- override_tt_config JSON for vLLM TT backend
"""

from dataclasses import dataclass, field
from typing import List, Optional

from workflows.workflow_types import DeviceTypes


# Fixed values for multi-host deployment
WORKER_SSH_PORT = 2200
CONTAINER_USER = "container_app_user"
DEFAULT_IDENTITY_FILE = f"/home/{CONTAINER_USER}/.ssh/id_ed25519_multihost"

# Common environment variables passed through to MPI workers
# Model-specific env vars (e.g., DEEPSEEK_V3_*) are in DeviceModelSpec.override_tt_config
ENV_PASSTHROUGH = [
    "VLLM_*",
    "MESH_DEVICE",
    "HF_TOKEN",
    "TT_METAL_HOME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
]


@dataclass
class MultiHostConfig:
    """Configuration for multi-host deployment."""

    hosts: List[str]
    mpi_interface: str
    shared_storage_root: str
    config_pkl_dir: str
    # SSH configuration (used by config generation)
    ssh_key_path: Optional[str] = None
    ssh_user: str = CONTAINER_USER
    ssh_port: int = WORKER_SSH_PORT
    rank_binding_path: Optional[str] = None
    # Orchestrator configuration (used by orchestrator)
    tt_smi_path: str = "tt-smi"
    deepseek_hf_model: Optional[str] = None
    deepseek_cache: Optional[str] = None
    # Internal: tracks if config_pkl_dir was auto-generated (for cleanup)
    _auto_generated_config_pkl_dir: bool = field(default=False, repr=False)


def generate_ssh_config(
    hosts: List[str],
    ssh_port: int = WORKER_SSH_PORT,
    ssh_user: str = CONTAINER_USER,
    identity_file: str = DEFAULT_IDENTITY_FILE,
) -> str:
    """Generate SSH config for Controller to connect to Worker containers.

    Uses real hostnames directly (not aliases) so that MPI rankfile
    resolution works without requiring /etc/hosts entries.

    Args:
        hosts: List of hostnames (e.g., ['host1', 'host2'])
        ssh_port: SSH port on Worker containers (default: 2200)
        ssh_user: SSH username (default: container_app_user)
        identity_file: Path to private key inside container

    Returns:
        SSH config file content as string
    """
    config_lines = []
    for host in hosts:
        config_lines.extend(
            [
                f"Host {host}",
                f"    Port {ssh_port}",
                f"    User {ssh_user}",
                f"    IdentityFile {identity_file}",
                "    StrictHostKeyChecking no",
                "    UserKnownHostsFile /dev/null",
                "    BatchMode yes",
                "",
            ]
        )
    return "\n".join(config_lines)


def generate_rankfile(hosts: List[str]) -> str:
    """Generate MPI rankfile for rank-to-host mapping.

    Each host gets one rank. Uses real hostnames directly so that
    MPI can resolve them via DNS without requiring /etc/hosts entries.

    Args:
        hosts: List of hostnames (e.g., ['host1', 'host2'])

    Returns:
        MPI rankfile content as string
    """
    lines = ["# mpirun rankfile"]
    for i, host in enumerate(hosts):
        lines.append(f"rank {i}={host} slot=0:*")
    return "\n".join(lines) + "\n"


def build_mpi_args(hosts: List[str], rankfile_path: str) -> str:
    """Build mpi_args string for override_tt_config.

    MPI requires --host to know available hosts, and rankfile for
    rank-to-slot mapping. Uses real hostnames directly.

    Args:
        hosts: List of hostnames (e.g., ['host1', 'host2'])
        rankfile_path: Path to the rankfile inside the container

    Returns:
        mpi_args string for vLLM
    """
    host_list = ",".join(hosts)
    return f"--host {host_list} --map-by rankfile:file={rankfile_path} --bind-to none"


def get_rank_binding_path(device_type: DeviceTypes) -> str:
    """Get the appropriate rank binding YAML path based on device type.

    Args:
        device_type: Device type for the deployment

    Returns:
        Path to rank binding YAML file inside container
    """
    tt_metal_home = "/home/container_app_user/tt-metal"
    config_dir = f"{tt_metal_home}/tests/tt_metal/distributed/config"

    binding_files = {
        DeviceTypes.DUAL_GALAXY: "dual_galaxy_rank_bindings.yaml",
        DeviceTypes.QUAD_GALAXY: "quad_galaxy_rank_bindings.yaml",
    }

    if device_type not in binding_files:
        raise ValueError(
            f"Unsupported device type: {device_type.name}. "
            f"Supported: {[d.name for d in binding_files.keys()]}"
        )

    return f"{config_dir}/{binding_files[device_type]}"


def build_override_tt_config(
    hosts: List[str],
    mpi_interface: str,
    config_pkl_dir: str,
    rankfile_path: str,
    device_type: DeviceTypes,
    rank_binding_path: Optional[str] = None,
) -> dict:
    """Build override_tt_config dict for multi-host vLLM deployment.

    Args:
        hosts: List of hostnames
        mpi_interface: Network interface for MPI (e.g., 'cnx1')
        config_pkl_dir: Directory for vLLM config pickle files (under shared storage)
        rankfile_path: Path to rankfile inside container
        device_type: Device type for the deployment (determines rank binding file)
        rank_binding_path: Path to rank binding YAML (auto-detected from device_type if None)

    Returns:
        Dictionary suitable for JSON serialization and passing to vLLM
    """
    if rank_binding_path is None:
        rank_binding_path = get_rank_binding_path(device_type)

    return {
        "rank_binding": rank_binding_path,
        "mpi_args": build_mpi_args(hosts, rankfile_path),
        "extra_ttrun_args": f"--tcp-interface {mpi_interface}",
        "config_pkl_dir": config_pkl_dir,
        "env_passthrough": ENV_PASSTHROUGH,
    }
