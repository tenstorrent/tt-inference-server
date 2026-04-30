# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Docker Compose variable resolution for tt-inference-server.

Resolves variables from ModelSpec/RuntimeConfig/SetupConfig into a flat dict
that populates the static compose templates in deploy/.

The compose templates use ${VAR} and ${VAR:-default} syntax. This module
provides the variable values. Users who don't use run.py can set these
variables manually in .env and run `docker compose up` directly.

Compose templates (versioned by contract era — see deploy/contracts.yml):
- deploy/docker-compose.vllm-0.11.yml    — vLLM, image versions >= 0.11.0
- deploy/docker-compose.media-0.11.yml   — media/forge, image versions >= 0.11.0
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

SIDECAR_PATH = get_repo_root_path() / ".env.compose.files"


def write_compose_files_sidecar(
    compose_files: list,
    path: Optional[Path] = None,
) -> Path:
    """Write the list of compose files used in this run to a sidecar file.

    The CI workflow (and any external script) reads this to know which
    compose files to pass to `compose logs` and `compose down` for cleanup.

    Output format is the literal string of `-f <path>` arguments separated
    by spaces, ready to be substituted into a `docker compose ...` command.
    Example: `-f /repo/deploy/docker-compose.vllm-0.11.yml -f /repo/deploy/overlays/dev-mode.yml`

    Args:
        compose_files: list of Path or str. Order is preserved (first file
            becomes the base; subsequent files act as overlays).
        path: where to write. Defaults to <repo_root>/.env.compose.files.

    Returns:
        The path written to.
    """
    if path is None:
        path = SIDECAR_PATH
    args = " ".join(f"-f {f}" for f in compose_files)
    path.write_text(args + "\n")
    logger.info(f"Wrote compose file list to {path}")
    return path


def build_compose_command(
    model_spec,
    runtime_config,
    setup_config,
    json_fpath: Optional[Path] = None,
):
    """Pick contract + overlays + compose vars for a single-host run.

    Selects the compose contract file based on image version (via contracts.yml),
    appends overlays based on runtime_config / setup_config / json_fpath, and
    resolves the env-var dict.

    This is shared by `run_compose_server` (which executes compose up) and
    by `run.py`'s `--print-compose` branch (which renders `compose config`),
    so adding a new overlay requires changing only one place.

    Args:
        model_spec: ModelSpec object
        runtime_config: RuntimeConfig object
        setup_config: SetupConfig (or None)
        json_fpath: Optional path to runtime model spec JSON

    Returns:
        Tuple of:
            compose_files: list[Path] — first is the contract, rest are overlays
            compose_vars: dict[str, str] — values for .env.compose
    """
    image_version = parse_image_version(model_spec.docker_image)
    contract_file = lookup_contract(model_spec.inference_engine, image_version)

    overlays = []
    if runtime_config.dev_mode:
        overlays.append(DEPLOY_DIR / "overlays" / "dev-mode.yml")
    if setup_config and getattr(setup_config, "host_model_volume_root", None):
        overlays.append(DEPLOY_DIR / "overlays" / "host-cache.yml")
    if setup_config and getattr(setup_config, "host_model_weights_mount_dir", None):
        overlays.append(DEPLOY_DIR / "overlays" / "host-weights.yml")
    if json_fpath:
        overlays.append(DEPLOY_DIR / "overlays" / "model-spec.yml")

    compose_files = [contract_file, *overlays]

    compose_vars = resolve_compose_vars(model_spec, runtime_config, setup_config)
    if runtime_config.dev_mode:
        compose_vars["REPO_ROOT"] = str(get_repo_root_path())
    if setup_config and getattr(setup_config, "host_model_volume_root", None):
        compose_vars["HOST_CACHE_ROOT"] = str(setup_config.host_model_volume_root)
    if setup_config and getattr(setup_config, "host_model_weights_mount_dir", None):
        compose_vars["HOST_MODEL_WEIGHTS"] = str(setup_config.host_model_weights_mount_dir)
        if getattr(setup_config, "container_model_weights_mount_dir", None):
            compose_vars["CONTAINER_MODEL_WEIGHTS"] = str(setup_config.container_model_weights_mount_dir)
        # Pre-0.11-era extras: ignored by 0.11+ templates, used by pre-0.11 template.
        if getattr(setup_config, "container_model_weights_path", None):
            compose_vars["MODEL_WEIGHTS_PATH"] = str(setup_config.container_model_weights_path)
        if getattr(setup_config, "container_tt_metal_cache_dir", None):
            compose_vars["TT_CACHE_PATH"] = str(setup_config.container_tt_metal_cache_dir)
    if json_fpath:
        compose_vars["RUNTIME_MODEL_SPEC_JSON"] = str(json_fpath)
        compose_vars["TT_MODEL_SPEC_HOST_PATH"] = str(json_fpath)  # pre-0.11 contract uses this

    return compose_files, compose_vars


def run_compose_server(
    model_spec,
    runtime_config,
    setup_config,
    json_fpath: Optional[Path] = None,
):
    """Run a single-host inference server via Docker Compose.

    Picks a contract from contracts.yml based on the image version, stacks
    overlays per runtime_config / setup_config, writes .env.compose and
    .env.compose.files, then runs `docker compose up -d --wait` so the
    healthcheck-based readiness check is the source of truth.

    Returns immediately after the server is healthy. The caller is responsible
    for running the workflow client and calling `run_compose_down` in a
    finally block.

    Returns:
        Dict with keys: container_name, compose_files, env_file.
    """
    from workflows.run_docker_server import ensure_docker_image

    compose_files, compose_vars = build_compose_command(
        model_spec, runtime_config, setup_config, json_fpath
    )

    write_compose_env(compose_vars)
    write_compose_files_sidecar(compose_files)

    if not is_compose_available():
        raise RuntimeError(
            "Docker Compose v2 is required but not available. "
            "Install with `apt-get install docker-compose-plugin` or upgrade Docker."
        )

    if not ensure_docker_image(model_spec.docker_image):
        raise RuntimeError(
            f"Docker image: {model_spec.docker_image} not found on GHCR or locally."
        )

    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd += ["-f", str(f)]
    cmd += ["--env-file", str(COMPOSE_ENV_PATH)]

    logger.info(f"Starting server via: {' '.join(cmd + ['up', '-d', '--wait'])}")
    result = subprocess.run(cmd + ["up", "-d", "--wait"])
    if result.returncode != 0:
        # `compose up --wait` already prints what failed; surface a final dump.
        subprocess.run(cmd + ["logs", "--tail", "200"])
        raise RuntimeError(
            f"`docker compose up --wait` exited {result.returncode}. "
            f"Server failed to become healthy. See logs above."
        )

    return {
        "container_name": compose_vars["CONTAINER_NAME"],
        "compose_files": [str(f) for f in compose_files],
        "env_file": str(COMPOSE_ENV_PATH),
    }


def run_compose_down(compose_files: list, env_file: Optional[Path] = None, timeout: int = 30):
    """Stop and remove containers/networks/volumes for a compose-managed run.

    Args:
        compose_files: List of paths/strings used in the original `compose up`.
        env_file: Path to the env file used. Defaults to COMPOSE_ENV_PATH.
        timeout: Seconds to wait for graceful shutdown before force-stop.
    """
    if env_file is None:
        env_file = COMPOSE_ENV_PATH
    cmd = ["docker", "compose"]
    for f in compose_files:
        cmd += ["-f", str(f)]
    cmd += ["--env-file", str(env_file), "down", "-t", str(timeout)]
    logger.info(f"Tearing down via: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
