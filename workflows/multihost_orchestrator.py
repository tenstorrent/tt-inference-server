# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Multi-host orchestration for distributed vLLM inference.

This module manages the lifecycle of Worker and Controller containers
across multiple hosts for MPI-based distributed inference.

Architecture:
- Controller (rank-0): Runs vLLM API server, initiates MPI processes
- Workers (rank-0+): Run sshd, MPI processes spawned by Controller via SSH

Execution Flow:
1. Orchestrator starts Worker containers on local and remote hosts (sshd mode)
2. Orchestrator starts Controller container (vLLM server mode)
3. Controller's vLLM calls tt-run which spawns MPI processes on Workers via SSH
"""

import atexit
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from workflows.multihost_config import (
    CONTAINER_USER,
    MultiHostConfig,
    WORKER_SSH_PORT,
    build_override_tt_config,
    generate_rankfile,
    generate_ssh_config,
)
from workflows.utils import default_dotenv_path, load_dotenv, write_dotenv
from workflows.workflow_types import DeviceTypes, WorkflowType


def _short_uuid():
    """Generate a short UUID for container naming."""
    return str(uuid.uuid4())[:8]


logger = logging.getLogger("run_log")


# Multi-host environment variable names
ENV_MULTIHOST_HOSTS = "MULTIHOST_HOSTS"
ENV_MPI_INTERFACE = "MPI_INTERFACE"
ENV_SHARED_STORAGE_ROOT = "SHARED_STORAGE_ROOT"
ENV_CONFIG_PKL_DIR = "CONFIG_PKL_DIR"
ENV_TT_SMI_PATH = "TT_SMI_PATH"
ENV_DEEPSEEK_V3_HF_MODEL = "DEEPSEEK_V3_HF_MODEL"
ENV_DEEPSEEK_V3_CACHE = "DEEPSEEK_V3_CACHE"


def _generate_config_pkl_path(shared_storage_root: str) -> str:
    """Generate a unique config pickle directory under shared storage root.

    Creates a session-specific directory for vLLM to write config pickle files.
    Directory structure: {shared_storage_root}/.tt-inference-server/{session_uuid}/config_pkl

    Args:
        shared_storage_root: Path to shared storage accessible by all hosts

    Returns:
        Path to the generated config pickle directory
    """
    session_id = str(uuid.uuid4())[:8]
    return str(
        Path(shared_storage_root)
        / ".tt-inference-server"
        / f"session-{session_id}"
        / "config_pkl"
    )


def _create_config_pkl_dir_with_permissions(config_pkl_dir: str) -> None:
    """Create config pickle directory with permissions for container user (UID 1000).

    Creates the directory hierarchy and sets world-writable permissions with
    sticky bit (0o1777) so that the container user can write pickle files,
    while preventing deletion of files by other users.

    Assumes shared_storage_root is writable by the host user.

    Args:
        config_pkl_dir: Path to config pickle directory to create
    """
    config_pkl_path = Path(config_pkl_dir)
    config_pkl_path.mkdir(parents=True, exist_ok=True)

    # Set permissions with sticky bit (1777 = rwxrwxrwt)
    # - World-writable so UID 1000 (container_app_user) can write
    # - Sticky bit prevents other users from deleting files they don't own
    os.chmod(config_pkl_path, 0o1777)

    # Also set permissions on the session parent directory
    session_dir = config_pkl_path.parent
    if session_dir.exists():
        os.chmod(session_dir, 0o1777)


def _cleanup_config_pkl_dir(config_pkl_dir: str) -> None:
    """Clean up auto-generated config pickle directory.

    Removes the config_pkl directory and its parent session directory if empty.

    Args:
        config_pkl_dir: Path to config pickle directory to clean up
    """
    config_pkl_path = Path(config_pkl_dir)
    if config_pkl_path.exists():
        logger.info(f"Cleaning up config pickle directory: {config_pkl_dir}")
        shutil.rmtree(config_pkl_path, ignore_errors=True)

        # Also try to remove parent session directory if empty
        session_dir = config_pkl_path.parent
        try:
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
                logger.debug(f"Removed empty session directory: {session_dir}")
        except OSError:
            pass  # Directory not empty or permission denied, skip


def setup_multihost_config(
    model_spec, expected_num_hosts: int, dry_run: bool = False
) -> MultiHostConfig:
    """Setup all multi-host configuration.

    Reads from .env or prompts interactively, then writes to .env.
    Returns MultiHostConfig with all values.

    If CONFIG_PKL_DIR is not specified, automatically generates a unique directory
    under SHARED_STORAGE_ROOT that will be cleaned up on exit.

    Args:
        model_spec: ModelSpec object to determine model-specific config requirements
        expected_num_hosts: Expected number of hosts based on device type
        dry_run: If True, skip directory creation and .env writes (for --print-docker-cmd)

    Returns:
        MultiHostConfig object with all configuration values
    """
    # Load existing .env to get any previously set values
    load_dotenv()

    config = {}

    # 1. MULTIHOST_HOSTS (comma-separated host list)
    hosts = os.getenv(ENV_MULTIHOST_HOSTS)
    if not hosts:
        hosts = input(
            f"Enter {ENV_MULTIHOST_HOSTS} (comma-separated, {expected_num_hosts} hosts required): "
        ).strip()

    # Validate host count matches expected
    host_list = [h.strip() for h in hosts.split(",")]
    if len(host_list) != expected_num_hosts:
        raise ValueError(
            f"Expected {expected_num_hosts} hosts for device type, but got {len(host_list)}. "
            f"Hosts provided: {host_list}"
        )

    config[ENV_MULTIHOST_HOSTS] = hosts

    # 2. MPI_INTERFACE
    mpi_interface = os.getenv(ENV_MPI_INTERFACE)
    if not mpi_interface:
        mpi_interface = input(f"Enter {ENV_MPI_INTERFACE} (e.g., 'cnx1'): ").strip()

    config[ENV_MPI_INTERFACE] = mpi_interface

    # 3. SHARED_STORAGE_ROOT
    shared_root = os.getenv(ENV_SHARED_STORAGE_ROOT)
    if not shared_root:
        shared_root = input(f"Enter {ENV_SHARED_STORAGE_ROOT} path: ").strip()

    if not Path(shared_root).exists():
        raise ValueError(f"{ENV_SHARED_STORAGE_ROOT} does not exist: {shared_root}")

    config[ENV_SHARED_STORAGE_ROOT] = shared_root

    # 4. CONFIG_PKL_DIR - auto-generate if not specified
    config_pkl_dir = os.getenv(ENV_CONFIG_PKL_DIR)
    auto_generated = False

    if not config_pkl_dir:
        # Auto-generate a unique path under shared storage
        config_pkl_dir = _generate_config_pkl_path(shared_root)
        auto_generated = True

    if not config_pkl_dir.startswith(shared_root):
        raise ValueError(f"{ENV_CONFIG_PKL_DIR} must be under {shared_root}")

    if not dry_run:
        if auto_generated:
            # Create with world-writable permissions for container user (UID 1000)
            logger.info(f"Auto-generated {ENV_CONFIG_PKL_DIR}: {config_pkl_dir}")
            _create_config_pkl_dir_with_permissions(config_pkl_dir)
            atexit.register(_cleanup_config_pkl_dir, config_pkl_dir)
            logger.debug(f"Registered cleanup for auto-generated {ENV_CONFIG_PKL_DIR}")
        else:
            # User-specified directory - just ensure it exists
            logger.info(f"Creating config pickle directory: {config_pkl_dir}")
            Path(config_pkl_dir).mkdir(parents=True, exist_ok=True)

    # Only save to .env if user explicitly provided the value
    if not auto_generated:
        config[ENV_CONFIG_PKL_DIR] = config_pkl_dir

    # 5. TT_SMI_PATH - path to tt-smi binary on remote hosts
    tt_smi_path = os.getenv(ENV_TT_SMI_PATH)
    if not tt_smi_path:
        tt_smi_path = input(
            f"Enter {ENV_TT_SMI_PATH} (path to tt-smi binary on hosts, or press Enter for 'tt-smi'): "
        ).strip()
        if not tt_smi_path:
            tt_smi_path = "tt-smi"

    config[ENV_TT_SMI_PATH] = tt_smi_path

    # 6. DeepSeek model specific environment variables
    deepseek_hf_model = None
    deepseek_cache = None
    if "deepseek" in model_spec.model_name.lower():
        deepseek_hf_model = os.getenv(ENV_DEEPSEEK_V3_HF_MODEL)
        deepseek_cache = os.getenv(ENV_DEEPSEEK_V3_CACHE)

        if not deepseek_hf_model:
            deepseek_hf_model = input(
                f"Enter {ENV_DEEPSEEK_V3_HF_MODEL} path: "
            ).strip()
        if not deepseek_cache:
            deepseek_cache = input(f"Enter {ENV_DEEPSEEK_V3_CACHE} path: ").strip()

        if not deepseek_hf_model.startswith(shared_root):
            raise ValueError(f"{ENV_DEEPSEEK_V3_HF_MODEL} must be under {shared_root}")
        if not deepseek_cache.startswith(shared_root):
            raise ValueError(f"{ENV_DEEPSEEK_V3_CACHE} must be under {shared_root}")

        if not Path(deepseek_hf_model).exists():
            raise ValueError(
                f"{ENV_DEEPSEEK_V3_HF_MODEL} does not exist: {deepseek_hf_model}"
            )
        if not Path(deepseek_cache).exists():
            raise ValueError(
                f"{ENV_DEEPSEEK_V3_CACHE} does not exist: {deepseek_cache}"
            )

        config[ENV_DEEPSEEK_V3_HF_MODEL] = deepseek_hf_model
        config[ENV_DEEPSEEK_V3_CACHE] = deepseek_cache

    # Write to .env (merges with existing content)
    if not dry_run:
        if config:  # Only write if there's something to save
            write_dotenv(config)
            logger.info("Multi-host configuration saved to .env:")
            for key, value in config.items():
                logger.info(f"  {key}={value}")
    else:
        logger.info("Dry-run mode: skipping .env write and directory creation")

    return MultiHostConfig(
        hosts=host_list,
        mpi_interface=mpi_interface,
        shared_storage_root=shared_root,
        config_pkl_dir=config_pkl_dir,
        tt_smi_path=tt_smi_path,
        deepseek_hf_model=deepseek_hf_model,
        deepseek_cache=deepseek_cache,
        _auto_generated_config_pkl_dir=auto_generated,
    )


@dataclass
class MultiHostSetup:
    """Generated files and paths for multi-host deployment."""

    config_dir: Path
    ssh_config_path: Path
    ssh_key_path: Path
    rankfile_path: Path
    override_tt_config: dict
    worker_container_ids: Dict[str, str]  # host -> container_id


class MultiHostOrchestrator:
    """Orchestrates multi-host vLLM deployment.

    Manages Worker and Controller container lifecycle across hosts.
    """

    def __init__(
        self,
        hosts: List[str],
        mpi_interface: str,
        shared_storage_root: str,
        config_pkl_dir: str,
        docker_image: str,
        runtime_config,
        model_spec,
        setup_config,
        tt_smi_path: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            hosts: List of hostnames for the deployment
            mpi_interface: Network interface for MPI (e.g., 'cnx1')
            shared_storage_root: Shared storage root path (mounted to containers)
            config_pkl_dir: Directory for vLLM config pickle files (under shared_storage_root)
            docker_image: Docker image to use for containers
            runtime_config: RuntimeConfig object
            model_spec: ModelSpec object
            setup_config: SetupConfig from setup_host()
            tt_smi_path: Path to tt-smi binary on hosts (used for validation)
        """
        self.hosts = hosts
        self.mpi_interface = mpi_interface
        self.shared_storage_root = shared_storage_root
        self.config_pkl_dir = config_pkl_dir
        self.docker_image = docker_image
        self.runtime_config = runtime_config
        self.model_spec = model_spec
        self.setup_config = setup_config
        self.tt_smi_path = tt_smi_path

        self.setup: Optional[MultiHostSetup] = None
        self._cleanup_registered = False

    def prepare(self) -> MultiHostSetup:
        """Generate all configuration files for multi-host deployment.

        Creates SSH keypair, SSH config, rankfile, and override_tt_config.
        Returns MultiHostSetup with paths to generated files.
        """
        # Create temp directory for generated configs
        config_dir = Path(tempfile.mkdtemp(prefix="tt_multihost_"))
        logger.info(f"Created multi-host config directory: {config_dir}")

        # Generate ephemeral SSH key pair (Ed25519 for better security and performance)
        ssh_key_path = config_dir / "id_ed25519_multihost"
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(ssh_key_path), "-q"],
            check=True,
        )
        ssh_key_path.chmod(0o600)
        logger.info(f"Generated ephemeral SSH key pair at {ssh_key_path}")

        # Generate SSH config
        ssh_config_content = generate_ssh_config(self.hosts)
        ssh_config_path = config_dir / "config"
        ssh_config_path.write_text(ssh_config_content)
        logger.debug(f"Generated SSH config at {ssh_config_path}")

        # Generate empty known_hosts
        (config_dir / "known_hosts").touch()

        # Generate MPI rankfile
        rankfile_content = generate_rankfile(self.hosts)
        mpirun_dir = config_dir / "mpirun"
        mpirun_dir.mkdir()
        rankfile_path = mpirun_dir / "rankfile"
        rankfile_path.write_text(rankfile_content)
        logger.debug(f"Generated rankfile at {rankfile_path}")

        # Generate override_tt_config
        device_type = DeviceTypes.from_string(self.runtime_config.device)
        override_tt_config = build_override_tt_config(
            hosts=self.hosts,
            mpi_interface=self.mpi_interface,
            config_pkl_dir=self.config_pkl_dir,
            rankfile_path="/etc/mpirun/rankfile",
            device_type=device_type,
        )

        self.setup = MultiHostSetup(
            config_dir=config_dir,
            ssh_config_path=ssh_config_path,
            ssh_key_path=ssh_key_path,
            rankfile_path=rankfile_path,
            override_tt_config=override_tt_config,
            worker_container_ids={},
        )

        return self.setup

    def generate_worker_docker_command(
        self, host: str, rank: int
    ) -> Tuple[List[str], str]:
        """Generate docker run command for a Worker container.

        Worker containers run sshd and wait for MPI processes from Controller.

        Args:
            host: Hostname to run the Worker on
            rank: MPI rank for this Worker

        Returns:
            Tuple of (docker command list, container name)

        Raises:
            RuntimeError: If prepare() has not been called
        """
        if not self.setup:
            raise RuntimeError(
                "Must call prepare() before generate_worker_docker_command()"
            )

        container_name = f"tt-worker-{rank}-{_short_uuid()}"

        # Public key is copied to /tmp/authorized_keys.pub on each host by copy_public_key_to_host()
        remote_public_key_path = "/tmp/authorized_keys.pub"

        # fmt: off
        cmd = [
            "docker", "run",
            "--rm", "-d",
            "--name", container_name,
            "--user", "root",  # entrypoint needs root for chown, sshd needs root for port
            "--net", "host",
            "--pid", "host",
            "--device", "/dev/tenstorrent:/dev/tenstorrent",
            "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
            # Mount shared storage root (contains model weights and config)
            "--mount", f"type=bind,src={self.shared_storage_root},dst={self.shared_storage_root}",
            # Mount public key for multihost_entrypoint.sh to configure authorized_keys
            "--mount", f"type=bind,src={remote_public_key_path},dst=/tmp/authorized_keys.pub,readonly",
            # Environment variables
            "-e", "MULTIHOST_ROLE=worker",
            "-e", f"SSH_PORT={WORKER_SSH_PORT}",
            # Use unified entrypoint script
            "--entrypoint", "/usr/local/bin/multihost_entrypoint.sh",
            self.docker_image,
        ]
        # fmt: on

        return cmd, container_name

    def generate_controller_docker_command(self) -> Tuple[List[str], str]:
        """Generate docker run command for Controller container.

        Controller runs vLLM API server and initiates MPI processes.

        Returns:
            Tuple of (docker command list, container name)
        """
        if not self.setup:
            raise RuntimeError(
                "Must call prepare() before generate_controller_docker_command()"
            )

        container_name = f"tt-controller-{_short_uuid()}"
        mesh_device = DeviceTypes.from_string(
            self.runtime_config.device
        ).to_mesh_device_str()

        # Merge MPI override_tt_config into the model_spec's vllm_args
        # This replaces the CONFIG_JSON mechanism to avoid duplicate --override-tt-config flags
        runtime_model_spec_json_path = self._write_merged_model_spec_json()
        container_model_spec_path = "/tmp/runtime_model_spec.json"

        # fmt: off
        cmd = [
            "docker", "run",
            "--rm",
            "--name", container_name,
            "--user", "root",  # entrypoint needs root for chown
            "--net", "host",
            "--pid", "host",
            "--env-file", str(default_dotenv_path),
            "--ipc", "host",
            "--device", "/dev/tenstorrent:/dev/tenstorrent",
            "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
            # Mount SSH config to temp location (entrypoint copies with correct perms)
            "--mount", f"type=bind,src={self.setup.config_dir},dst=/tmp/ssh_config,readonly",
            # Mount MPI rankfile
            "--mount", f"type=bind,src={self.setup.config_dir / 'mpirun'},dst=/etc/mpirun,readonly",
            # Mount shared storage root (contains model weights and config)
            "--mount", f"type=bind,src={self.shared_storage_root},dst={self.shared_storage_root}",
            # Mount runtime model spec JSON with merged override_tt_config
            "--mount", f"type=bind,src={runtime_model_spec_json_path},dst={container_model_spec_path},readonly",
            # Environment variables
            "-e", "MULTIHOST_ROLE=controller",
            "-e", "SSH_CONFIG_SRC=/tmp/ssh_config",
            "-e", f"MESH_DEVICE={mesh_device}",
            "-e", "VLLM_TARGET_DEVICE=tt",
            "-e", f"NCCL_SOCKET_IFNAME={self.mpi_interface}",
            "-e", f"ETH={self.mpi_interface}",
            "-e", f"HOSTS={','.join(self.hosts)}",
            "-e", f"RUNTIME_MODEL_SPEC_JSON_PATH={container_model_spec_path}",
            # Use unified entrypoint script
            "--entrypoint", "/usr/local/bin/multihost_entrypoint.sh",
        ]
        # fmt: on

        # Add cache volume if setup_config is available
        if self.setup_config:
            if self.setup_config.host_model_volume_root:
                cmd.extend(
                    [
                        "--mount",
                        f"type=bind,src={self.setup_config.host_model_volume_root},dst={self.setup_config.cache_root}",
                    ]
                )
            else:
                cmd.extend(
                    [
                        "--volume",
                        f"{self.setup_config.docker_volume_name}:{self.setup_config.cache_root}",
                    ]
                )

            if self.setup_config.host_model_weights_mount_dir:
                cmd.extend(
                    [
                        "--mount",
                        f"type=bind,src={self.setup_config.host_model_weights_mount_dir},dst={self.setup_config.container_model_weights_mount_dir},readonly",
                    ]
                )
                cmd.extend(
                    [
                        "-e",
                        f"MODEL_WEIGHTS_DIR={self.setup_config.container_model_weights_path}",
                    ]
                )

        # Add image
        cmd.append(self.docker_image)

        # vLLM command via run_vllm_api_server.py (handles --model and --tt-device)
        cmd.extend(
            [
                "python",
                "run_vllm_api_server.py",
                "--model",
                self.model_spec.hf_model_repo,
                "--tt-device",
                self.runtime_config.device,
            ]
        )

        if self.runtime_config.no_auth:
            cmd.append("--no-auth")
        if self.runtime_config.disable_trace_capture:
            cmd.append("--disable-trace-capture")
        if (
            self.runtime_config.service_port
            and str(self.runtime_config.service_port) != "8000"
        ):
            cmd.extend(["--service-port", str(self.runtime_config.service_port)])

        return cmd, container_name

    def _merge_override_tt_config(self, base: dict, override: dict) -> dict:
        """Merge two override_tt_config dicts with special handling for lists.

        For list values (e.g., env_passthrough), concatenates instead of overwriting.
        For other values, override takes precedence.

        Args:
            base: Base config dict
            override: Override config dict (takes precedence for non-list values)

        Returns:
            Merged config dict
        """
        merged = base.copy()
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], list)
                and isinstance(value, list)
            ):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
        return merged

    def _write_merged_model_spec_json(self) -> Path:
        """Write a runtime model spec JSON with merged override_tt_config.

        Merges the MPI override_tt_config from multihost setup into the model_spec's
        device_model_spec.vllm_args.override_tt_config. This allows passing all vLLM
        configuration through a single JSON file instead of using CONFIG_JSON env var.

        Returns:
            Path to the written JSON file
        """
        # Get serialized model spec dict
        spec_dict = self.model_spec.get_serialized_dict()

        # Get the existing override_tt_config from vllm_args (it's a JSON string)
        vllm_args = spec_dict.get("device_model_spec", {}).get("vllm_args", {})
        existing_override_str = vllm_args.get("override_tt_config", "{}")
        existing_override = json.loads(existing_override_str)

        # Merge with MPI override_tt_config from multihost setup
        # Lists (e.g., env_passthrough) are concatenated, others are overwritten
        merged_override = self._merge_override_tt_config(
            existing_override, self.setup.override_tt_config
        )

        # Also merge any override_tt_config from runtime_config
        if self.runtime_config.override_tt_config:
            runtime_override = json.loads(self.runtime_config.override_tt_config)
            merged_override = self._merge_override_tt_config(
                merged_override, runtime_override
            )

        # Update the spec dict with merged override_tt_config
        spec_dict["device_model_spec"]["vllm_args"]["override_tt_config"] = json.dumps(
            merged_override
        )

        # Write to JSON file in config_dir
        json_path = self.setup.config_dir / "runtime_model_spec.json"
        combined = {"runtime_model_spec": spec_dict}
        with open(json_path, "w") as f:
            json.dump(combined, f, indent=2)

        logger.info(f"Wrote merged runtime model spec to: {json_path}")
        logger.debug(
            f"Merged override_tt_config: {json.dumps(merged_override, indent=2)}"
        )

        return json_path

    def start_worker_on_host(self, host: str, rank: int) -> str:
        """Start a Worker container on a remote host.

        Args:
            host: Hostname to start the Worker on
            rank: MPI rank for this Worker

        Returns:
            Container ID of the started Worker
        """
        cmd, container_name = self.generate_worker_docker_command(host, rank)

        # Run docker command on remote host via SSH (port 22 for host, not container port)
        # Use shlex.join to properly quote arguments with special characters like (8,4)
        ssh_cmd = ["ssh", "-p", "22", host, shlex.join(cmd)]
        logger.info(f"Starting Worker container on {host} (rank {rank})")
        logger.debug(f"SSH command: {' '.join(ssh_cmd)}")

        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to start Worker on {host}: {result.stderr}")
            raise RuntimeError(f"Failed to start Worker container on {host}")

        container_id = result.stdout.strip()
        logger.info(f"Started Worker container {container_id[:12]} on {host}")

        if self.setup:
            self.setup.worker_container_ids[host] = container_id

        return container_id

    def ensure_image_on_host(self, host: str) -> bool:
        """Ensure Docker image exists and matches on remote host.

        Compares local and remote image IDs. If they differ or remote
        doesn't have the image, transfers it via docker save/load.

        Args:
            host: Target hostname

        Returns:
            True if image is available (existed or transferred)
        """
        # Get local image ID
        local_result = subprocess.run(
            ["docker", "image", "inspect", self.docker_image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
        )
        if local_result.returncode != 0:
            logger.error(f"Image {self.docker_image} not found locally")
            return False
        local_id = local_result.stdout.strip()

        # Get remote image ID
        remote_result = subprocess.run(
            [
                "ssh",
                "-p",
                "22",
                host,
                "docker",
                "image",
                "inspect",
                self.docker_image,
                "--format",
                "{{.Id}}",
            ],
            capture_output=True,
            text=True,
        )

        if remote_result.returncode == 0:
            remote_id = remote_result.stdout.strip()
            if local_id == remote_id:
                logger.info(f"Image {self.docker_image} up-to-date on {host}")
                return True
            logger.info(f"Image {self.docker_image} outdated on {host}, updating...")
        else:
            logger.info(
                f"Image {self.docker_image} not found on {host}, transferring..."
            )

        # Transfer using subprocess pipeline (no shell=True for security)
        save_proc = subprocess.Popen(
            ["docker", "save", self.docker_image],
            stdout=subprocess.PIPE,
        )
        load_proc = subprocess.Popen(
            ["ssh", "-p", "22", host, "docker", "load"],
            stdin=save_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        save_proc.stdout.close()
        stdout, stderr = load_proc.communicate()

        if load_proc.returncode != 0:
            logger.error(f"Failed to transfer image to {host}: {stderr.decode()}")
            return False

        logger.info(f"Image transferred to {host}")
        return True

    def ensure_images_on_all_hosts(self) -> bool:
        """Ensure Docker image is available on all hosts.

        Transfers images to all hosts in parallel using ThreadPoolExecutor
        for improved performance on multi-host deployments.

        Returns:
            True if all hosts have the image
        """
        with ThreadPoolExecutor(max_workers=len(self.hosts)) as executor:
            results = list(executor.map(self.ensure_image_on_host, self.hosts))
        return all(results)

    def copy_public_key_to_host(self, host: str) -> bool:
        """Copy SSH public key to remote host for Worker authorization.

        Args:
            host: Target hostname

        Returns:
            True if copy succeeded
        """
        if not self.setup:
            raise RuntimeError("Must call prepare() before copy_public_key_to_host()")

        public_key_path = Path(str(self.setup.ssh_key_path) + ".pub")
        remote_path = "/tmp/authorized_keys.pub"

        result = subprocess.run(
            ["scp", "-P", "22", str(public_key_path), f"{host}:{remote_path}"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Failed to copy public key to {host}: {result.stderr}")
            return False

        logger.info(f"Copied public key to {host}:{remote_path}")
        return True

    def copy_public_key_to_all_hosts(self) -> bool:
        """Copy SSH public key to all hosts.

        Returns:
            True if all copies succeeded
        """
        for host in self.hosts:
            if not self.copy_public_key_to_host(host):
                return False
        return True

    def start_all_workers(self) -> Dict[str, str]:
        """Start Worker containers on all hosts.

        Ensures image and SSH public key are available on all hosts before starting.

        Returns:
            Dict mapping hostname to container ID

        Raises:
            RuntimeError: If image distribution or key copy fails
        """
        # Ensure image is available on all hosts
        if not self.ensure_images_on_all_hosts():
            raise RuntimeError("Failed to ensure Docker image on all hosts")

        # Copy SSH public key to all hosts
        if not self.copy_public_key_to_all_hosts():
            raise RuntimeError("Failed to copy SSH public key to all hosts")

        container_ids = {}
        for rank, host in enumerate(self.hosts):
            container_id = self.start_worker_on_host(host, rank)
            container_ids[host] = container_id

        self._register_cleanup()
        return container_ids

    def validate_worker_ssh(self, host: str, rank: int) -> bool:
        """Test SSH connectivity to a Worker container.

        Args:
            host: Hostname where Worker is running
            rank: MPI rank of the Worker

        Returns:
            True if SSH connection succeeds

        Raises:
            RuntimeError: If prepare() has not been called
        """
        if not self.setup:
            raise RuntimeError("Must call prepare() before validate_worker_ssh()")

        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-p",
            str(WORKER_SSH_PORT),
            "-i",
            str(self.setup.ssh_key_path),
            f"{CONTAINER_USER}@{host}",
            "echo",
            "SSH OK",
        ]

        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0 and "SSH OK" in result.stdout
        except subprocess.TimeoutExpired:
            return False

    def wait_for_workers_ready(self, timeout: int = 60) -> bool:
        """Wait for all Workers to be ready for SSH connections.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all Workers are ready
        """
        start_time = time.time()
        ready_hosts = set()

        while time.time() - start_time < timeout:
            for rank, host in enumerate(self.hosts):
                if host in ready_hosts:
                    continue
                if self.validate_worker_ssh(host, rank):
                    ready_hosts.add(host)
                    logger.info(f"Worker on {host} is ready")

            if len(ready_hosts) == len(self.hosts):
                return True

            time.sleep(2)

        missing = set(self.hosts) - ready_hosts
        logger.error(f"Workers not ready after {timeout}s: {missing}")
        return False

    def _register_cleanup(self):
        """Register atexit and signal handlers to clean up containers.

        Registers cleanup for:
        - Normal exit (atexit)
        - SIGTERM (e.g., docker stop, kill)
        - SIGINT (Ctrl+C)

        For server/reports workflows, we skip cleanup registration so that
        Workers remain running after the orchestrator process exits.
        """
        if self._cleanup_registered:
            return

        # For server/reports workflows, containers should keep running after exit
        skip_cleanup_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
        if (
            WorkflowType.from_string(self.runtime_config.workflow)
            in skip_cleanup_workflows
        ):
            return

        def cleanup():
            logger.info("Cleaning up multi-host containers...")
            self.stop_all_workers()
            if self.setup and self.setup.config_dir.exists():
                shutil.rmtree(self.setup.config_dir, ignore_errors=True)

        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.info(f"Received {sig_name}, cleaning up...")
            cleanup()
            sys.exit(128 + signum)

        atexit.register(cleanup)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        self._cleanup_registered = True

    def stop_all_workers(self):
        """Stop all Worker containers."""
        if not self.setup:
            return

        for host, container_id in self.setup.worker_container_ids.items():
            try:
                ssh_cmd = ["ssh", "-p", "22", host, "docker", "stop", container_id]
                subprocess.run(ssh_cmd, capture_output=True, timeout=30)
                logger.info(f"Stopped Worker container on {host}")
            except Exception as e:
                logger.warning(f"Failed to stop Worker on {host}: {e}")

    def check_worker_status(self, host: str) -> Tuple[bool, Optional[str]]:
        """Check if Worker container is still running on a host.

        Args:
            host: Hostname where Worker is running

        Returns:
            Tuple of (is_running, error_message). If running, error_message is None.
        """
        if not self.setup or host not in self.setup.worker_container_ids:
            return False, "Worker not started"

        container_id = self.setup.worker_container_ids[host]
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-p",
                    "22",
                    host,
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Status}}",
                    container_id,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False, f"Container not found: {result.stderr.strip()}"

            status = result.stdout.strip()
            if status != "running":
                # Get logs from exited container
                logs_result = subprocess.run(
                    [
                        "ssh",
                        "-p",
                        "22",
                        host,
                        "docker",
                        "logs",
                        "--tail",
                        "50",
                        container_id,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logs = logs_result.stdout + logs_result.stderr
                return False, f"Container status: '{status}'. Logs:\n{logs}"
            return True, None
        except subprocess.TimeoutExpired:
            return False, "Timeout checking container status"
        except Exception as e:
            return False, str(e)

    def check_all_workers_status(self) -> Tuple[bool, Optional[str]]:
        """Check if all Worker containers are running.

        Returns:
            Tuple of (all_running, error_message). If all running, error_message is None.
        """
        for host in self.hosts:
            running, error = self.check_worker_status(host)
            if not running:
                return False, f"Worker on {host}: {error}"
        return True, None

    def get_worker_logs(self, host: str, tail: int = 100) -> str:
        """Get logs from a Worker container.

        Args:
            host: Hostname where Worker is running
            tail: Number of lines to retrieve

        Returns:
            Log output as string
        """
        if not self.setup or host not in self.setup.worker_container_ids:
            return "Worker not started"

        container_id = self.setup.worker_container_ids[host]
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-p",
                    "22",
                    host,
                    "docker",
                    "logs",
                    "--tail",
                    str(tail),
                    container_id,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Failed to get logs: {e}"


def is_multihost_deployment(runtime_config) -> bool:
    """Check if this is a multi-host deployment based on device type.

    Multi-host deployment is determined by the device type:
    - DUAL_GALAXY: 2-host deployment
    - QUAD_GALAXY: 4-host deployment

    Args:
        runtime_config: RuntimeConfig object

    Returns:
        True if device type requires multi-host deployment
    """
    try:
        device = DeviceTypes.from_string(runtime_config.device)
        return device.is_multihost()
    except (ValueError, AttributeError):
        return False


def get_expected_num_hosts(runtime_config) -> int:
    """Get expected number of hosts based on device type.

    Args:
        runtime_config: RuntimeConfig object

    Returns:
        Expected number of hosts for the device type

    Raises:
        ValueError: If device type is not a multi-host type
    """
    device = DeviceTypes.from_string(runtime_config.device)
    return device.get_multihost_num_hosts()
