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
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import yaml

from packaging.version import InvalidVersion, Version

from workflows.utils import get_repo_root_path, write_dotenv
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
)

logger = logging.getLogger("run_log")

DEPLOY_DIR = get_repo_root_path() / "deploy"
COMPOSE_ENV_PATH = get_repo_root_path() / ".env.compose"


def _short_uuid():
    return str(uuid.uuid4())[:8]


_VERSION_PREFIX_RE = re.compile(r"^(\d+(?:\.\d+){1,2})")


def parse_image_version(image: str) -> Optional[Version]:
    """Parse a leading PEP-440 / semver-style version from a Docker image tag.

    Returns None if the image has no tag, or if the tag's leading characters
    do not form a parseable version. The build suffix after the version
    (e.g. "-fae3df") is ignored.

    Examples:
        >>> parse_image_version("ghcr.io/foo/bar:0.11.0-abc") == Version("0.11.0")
        True
        >>> parse_image_version("ghcr.io/foo/bar:dev") is None
        True
    """
    if ":" not in image:
        return None
    tag = image.rsplit(":", 1)[1]
    match = _VERSION_PREFIX_RE.match(tag)
    if not match:
        return None
    try:
        return Version(match.group(1))
    except InvalidVersion:
        return None


class NoMatchingContractError(Exception):
    """Raised when no contract in contracts.yml matches a given engine/version."""


def lookup_contract(
    engine: str,
    version: Optional[Version],
    contracts_path: Optional[Path] = None,
) -> Path:
    """Return the compose contract file path for `engine` that supports `version`.

    Selection rule: among the contracts listed for `engine`, pick the one with
    the largest `min_version` that is <= `version`. If `version` is None, pick
    the contract with the largest `min_version` (the newest era) and log a
    warning.

    Args:
        engine: Engine name as listed in contracts.yml (e.g. "vllm", "media").
        version: Parsed image version, or None when the tag was unparseable.
        contracts_path: Path to contracts.yml. Defaults to DEPLOY_DIR/contracts.yml.

    Raises:
        NoMatchingContractError: if `engine` is missing from the file, or if no
            contract for `engine` has a `min_version` <= `version`.
    """
    if contracts_path is None:
        contracts_path = DEPLOY_DIR / "contracts.yml"

    with open(contracts_path) as f:
        data = yaml.safe_load(f) or {}

    contracts = data.get("contracts", {}).get(engine)
    if not contracts:
        raise NoMatchingContractError(
            f"No contracts defined for engine '{engine}' in {contracts_path}"
        )

    parsed = sorted(
        ((Version(c["min_version"]), c["file"]) for c in contracts),
        key=lambda x: x[0],
        reverse=True,
    )

    if version is None:
        logger.warning(
            "Image version could not be parsed; falling back to newest "
            "contract '%s' for engine '%s'.",
            parsed[0][1],
            engine,
        )
        return contracts_path.parent / parsed[0][1]

    for min_v, fname in parsed:
        if min_v <= version:
            return contracts_path.parent / fname

    raise NoMatchingContractError(
        f"No contract for engine '{engine}' supports version {version}. "
        f"Lowest min_version available: {parsed[-1][0]}. "
        f"Add a row to {contracts_path} to support this version."
    )


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

    from workflows.utils import default_dotenv_path

    env = {
        # Common
        "DOCKER_IMAGE": model_spec.docker_image,
        "CONTAINER_NAME": container_name,
        "SERVICE_PORT": str(runtime_config.service_port),
        "BIND_HOST": runtime_config.bind_host,
        # Absolute path to secrets .env (compose resolves relative to template dir)
        "ENV_FILE": str(default_dotenv_path),
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
    # NOTE: CACHE_ROOT is intentionally NOT written to .env because it's a
    # container-internal path (/home/container_app_user/cache_root). Writing it
    # to .env would cause run.py's get_default_workflow_root_log_dir() to try
    # creating host directories at the container path. The compose templates
    # use ${CACHE_ROOT:-/home/container_app_user/cache_root} as default.
    if setup_config:
        if setup_config.host_model_volume_root:
            env["CACHE_VOLUME"] = str(setup_config.host_model_volume_root)
        else:
            from workflows.run_docker_server import generate_docker_volume_name

            env["CACHE_VOLUME"] = generate_docker_volume_name(model_spec)

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
    """Resolve compose variables for multi-host deployment (display/dry-run).

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


def resolve_multihost_worker_vars(
    model_spec,
    multihost_config,
    rank: int,
) -> dict:
    """Resolve compose variables for a multi-host Worker.

    Args:
        model_spec: ModelSpec object
        multihost_config: MultiHostConfig
        rank: MPI rank for this worker

    Returns:
        Dict of environment variable name → value for worker compose template
    """
    return {
        "DOCKER_IMAGE": model_spec.docker_image,
        "CONTAINER_NAME": f"tt-worker-{rank}-{_short_uuid()}",
        "SHARED_STORAGE_ROOT": multihost_config.shared_storage_root,
        "SSH_PORT": str(multihost_config.ssh_port),
    }


def resolve_multihost_controller_vars(
    model_spec,
    runtime_config,
    multihost_config,
    setup,
    runtime_model_spec_json_path,
    setup_config=None,
) -> dict:
    """Resolve compose variables for multi-host Controller.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        multihost_config: MultiHostConfig
        setup: MultiHostSetup with generated config paths
        runtime_model_spec_json_path: Path to merged model spec JSON
        setup_config: Optional SetupConfig for cache volume

    Returns:
        Dict of environment variable name → value for controller compose template
    """
    from workflows.utils import default_dotenv_path

    device = DeviceTypes.from_string(runtime_config.device)

    env = {
        "DOCKER_IMAGE": model_spec.docker_image,
        "CONTAINER_NAME": f"tt-controller-{_short_uuid()}",
        "ENV_FILE": str(default_dotenv_path),
        "HF_MODEL": model_spec.hf_model_repo,
        "TT_DEVICE": runtime_config.device,
        "SHARED_STORAGE_ROOT": multihost_config.shared_storage_root,
        "MULTIHOST_HOSTS": ",".join(multihost_config.hosts),
        "MPI_INTERFACE": multihost_config.mpi_interface,
        "MESH_DEVICE": device.to_mesh_device_str(),
        "SSH_CONFIG_DIR": str(setup.config_dir),
        "MPIRUN_DIR": str(setup.config_dir / "mpirun"),
        "RUNTIME_MODEL_SPEC_JSON": str(runtime_model_spec_json_path),
    }

    # Cache volume
    if setup_config:
        if setup_config.host_model_volume_root:
            env["CACHE_VOLUME"] = str(setup_config.host_model_volume_root)
        else:
            from workflows.run_docker_server import generate_docker_volume_name
            env["CACHE_VOLUME"] = generate_docker_volume_name(model_spec)

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


# ---------------------------------------------------------------------------
# Compose availability check
# ---------------------------------------------------------------------------


def is_compose_available() -> bool:
    """Check if Docker Compose v2 is available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Compose execution
# ---------------------------------------------------------------------------


def write_compose_env(
    compose_vars: dict,
    env_file: Path = None,
) -> Path:
    """Write compose variables to a dedicated .env.compose file.

    Uses a separate file from .env (which holds secrets like HF_TOKEN,
    JWT_SECRET) to avoid polluting the secrets file with container-internal
    paths like CACHE_ROOT.

    Docker Compose reads template ${VAR} substitutions from --env-file,
    while the container's runtime secrets come from the env_file: directive
    in the compose template (which points to .env).

    Args:
        compose_vars: Dict from resolve_compose_vars()
        env_file: Path to write (defaults to .env.compose in repo root)

    Returns:
        Path to the written file
    """
    if env_file is None:
        env_file = COMPOSE_ENV_PATH

    # Write fresh (not merge) — compose vars are fully resolved each run
    with open(env_file, "w") as f:
        for key, value in compose_vars.items():
            f.write(f"{key}={value}\n")

    logger.info(f"Wrote compose variables to {env_file}")
    return env_file


def _compose_cmd(compose_file: Path, env_file: Optional[Path] = None) -> list:
    """Build base docker compose command with file and env-file flags."""
    cmd = ["docker", "compose", "-f", str(compose_file)]
    # --env-file controls template ${VAR} substitution (compose variables)
    # This is separate from env_file: in the compose template (secrets/.env)
    env_file = env_file or COMPOSE_ENV_PATH
    cmd.extend(["--env-file", str(env_file)])
    return cmd


def compose_up(
    compose_file: Path,
    env_file: Optional[Path] = None,
    detach: bool = True,
) -> subprocess.CompletedProcess:
    """Run `docker compose up` with the given compose file.

    Args:
        compose_file: Path to docker-compose.yml
        env_file: Path to compose env file (defaults to .env.compose)
        detach: Run in detached mode (default True)

    Returns:
        CompletedProcess result
    """
    cmd = _compose_cmd(compose_file, env_file)
    cmd.append("up")
    if detach:
        cmd.append("-d")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"docker compose up failed: {result.stderr}")
    return result


def compose_down(
    compose_file: Path,
    env_file: Optional[Path] = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run `docker compose down` to stop and remove containers.

    Args:
        compose_file: Path to docker-compose.yml
        env_file: Path to compose env file (defaults to .env.compose)
        timeout: Shutdown timeout in seconds

    Returns:
        CompletedProcess result
    """
    cmd = _compose_cmd(compose_file, env_file)
    cmd.extend(["down", "-t", str(timeout)])

    logger.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)


def compose_ps(
    compose_file: Path,
    env_file: Optional[Path] = None,
) -> str:
    """Run `docker compose ps` to check container status.

    Args:
        compose_file: Path to docker-compose.yml
        env_file: Path to compose env file (defaults to .env.compose)

    Returns:
        Output string from docker compose ps
    """
    cmd = _compose_cmd(compose_file, env_file)
    cmd.append("ps")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def compose_logs(
    compose_file: Path,
    env_file: Optional[Path] = None,
    follow: bool = False,
    log_file_path: Optional[Path] = None,
) -> Optional[subprocess.Popen]:
    """Run `docker compose logs` to stream container logs.

    Args:
        compose_file: Path to docker-compose.yml
        env_file: Path to compose env file (defaults to .env.compose)
        follow: Follow log output (blocking unless log_file_path is set)
        log_file_path: If set, redirect logs to this file and return Popen

    Returns:
        Popen process if log_file_path is set, None otherwise
    """
    cmd = _compose_cmd(compose_file, env_file)
    cmd.append("logs")
    if follow:
        cmd.append("-f")

    if log_file_path:
        log_file = open(log_file_path, "w", buffering=1)
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file, text=True
        )
        return proc
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
