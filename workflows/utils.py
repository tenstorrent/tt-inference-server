# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

import base64
import logging
import os
import re
import shlex
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# SDXL num prompts limits
SDXL_DEFAULT_NUM_PROMPTS = 100
SDXL_LOWER_BOUND_NUM_PROMPTS = 2
SDXL_UPPER_BOUND_NUM_PROMPTS = 5000


def get_repo_root_path(marker: str = ".git") -> Path:
    """Return the root directory of the repository by searching for a marker file or directory."""
    current_path = Path(__file__).resolve().parent  # Start from the script's directory
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Repository root not found. No '{marker}' found in parent directories."
    )


def get_version() -> str:
    """Return the version of the repository."""
    version_file = get_repo_root_path(marker="VERSION") / "VERSION"
    assert version_file.exists(), f"Version file not found: {version_file}"
    with version_file.open("r", encoding="utf-8") as file:
        return file.read().strip()


def parse_commits_from_docker_image(
    docker_image: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Extract tt-metal and vllm commits from docker image tag.

    Supports two tag formats:

    1. Format for LLMs: version-tt_metal_commit(40)-vllm_commit(7)-timestamp
       Example: 0.4.0-4733994fc8bea3db5a1ba0aa5b18fd9f658708c0-47f6635-56816832543
    2. Format for media server: version-tt_metal_commit(40)-tt_inference_sha(7)-timestamp
       Example: 0.4.0-d2f891d4af7a12911f9029bbf788462624fcf980-ca7e3d6-57576349393
       (vllm_commit will be None for media server images)

    Note: Media server images are detected by checking if image name contains "tt-media-inference-server".
    For media images, the third component in the tag is NOT a vllm commit, so we ignore it.

    Args:
        docker_image: Full docker image string with tag

    Returns:
        Tuple of (tt_metal_commit, vllm_commit) or (None, None) if parsing fails
    """
    if not docker_image or ":" not in docker_image:
        return None, None

    try:
        image_name, tag = docker_image.rsplit(":", 1)
        is_media_server = "tt-media-inference-server" in image_name

        # Example: 0.4.0-4733994fc8bea3db5a1ba0aa5b18fd9f658708c0-47f6635-56816832543
        expected_tag_pattern = r"^([0-9.]+)-([0-9a-fA-F]{40})-([0-9a-fA-F]{7})-(\d+)$"
        match = re.match(expected_tag_pattern, tag)

        if match:
            version, tt_metal_commit, vllm_commit, timestamp = match.groups()
            if is_media_server:
                # For media server images, ignore the third component (tt_inference_sha) as it's not a vllm commit
                logger.info(
                    f"Parsed commits from media server docker image tag: tt-metal={tt_metal_commit}"
                )
                return tt_metal_commit, None
            else:
                # For vLLM images, return both commits
                logger.info(
                    f"Parsed commits from docker image tag: tt-metal={tt_metal_commit}, vllm={vllm_commit}"
                )
                return tt_metal_commit, vllm_commit

        logger.debug(f"Docker image tag does not match expected format: {tag}")
        return None, None

    except Exception as e:
        logger.debug(f"Failed to parse commits from docker image '{docker_image}': {e}")
        return None, None


def get_run_id(timestamp, model_id, workflow):
    def _short_uuid():
        """Return 8-character random UUID"""
        # Generate UUID4 (random)
        u = uuid.uuid4()
        # Convert to bytes and encode with URL-safe base64
        return base64.urlsafe_b64encode(u.bytes)[:8].decode("utf-8")

    return f"{timestamp}_{model_id}_{workflow}_{_short_uuid()}"


def get_default_workflow_root_log_dir():
    # docker env uses CACHE_ROOT
    default_dir_name = "workflow_logs"
    cache_root = os.getenv("CACHE_ROOT")
    if cache_root:
        default_workflow_root_log_dir = Path(cache_root) / default_dir_name
    else:
        default_workflow_root_log_dir = get_repo_root_path() / default_dir_name
    return default_workflow_root_log_dir


def ensure_readwriteable_dir(path, raise_on_fail=True, logger=logger):
    """
    Ensures that the given path is a directory.
    If it doesn't exist, create it.
    If it exists, check if it is readable and writable.

    Args:
        path (str or Path): The directory path to check.
        raise_on_fail (bool): Whether to raise an exception on failure. Defaults to True.

    Returns:
        bool: True if the directory is readable and writable, False otherwise.

    Raises:
        ValueError: If the path exists but is not a directory (when raise_on_fail=True).
        PermissionError: If the directory is not readable or writable (when raise_on_fail=True).
    """
    path = Path(path)  # Convert to Path object

    try:
        if not path.exists():
            path.mkdir(
                parents=True, exist_ok=True
            )  # Create directory and necessary parents
            logger.info(f"Directory created: {path}")
        elif not path.is_dir():
            logger.error(f"'{path}' exists but is not a directory.")
            if raise_on_fail:
                raise ValueError(f"'{path}' exists but is not a directory.")
            return False

        # Check read/write permissions using a test file
        try:
            # Create a temporary file in the target directory
            with tempfile.NamedTemporaryFile(dir=path, delete=True) as tmpfile:
                # Try writing to the file
                file_data = b"test"
                tmpfile.write(file_data)
                tmpfile.flush()

                # Try reading from the file
                tmpfile.seek(0)
                data = tmpfile.read()
                logger.info(f"Directory '{path}' is readable and writable.")
                return data == file_data
        except IOError:
            logger.error(f"Directory '{path}' is not writable.")
            if raise_on_fail:
                raise PermissionError(f"Directory '{path}' is not writable.")
            return False

    except Exception as e:
        logger.exception(f"An error occurred while checking directory '{path}': {e}")
        if raise_on_fail:
            raise
        return False


def stream_subprocess_output(pipe, logger, level):
    with pipe:
        for line in iter(pipe.readline, ""):
            logger.log(level, line.strip(), extra={"raw": True})


def run_command(
    command, logger, log_file_path=None, shell=False, copy_env=True, env=None, check=True
):
    """
    Note: logger must be passed because the common use case is to capture the command's
    stdout and stderr in the caller's logger.
    """
    if not copy_env:
        raise NotImplementedError("TODO")

    if not env:
        env = os.environ.copy()
    # TODO: force usage to always use argument list
    # use shlex to log full command before running

    if command is None:
        logger.error("No command provided to run_command.")
    elif isinstance(command, str):
        command = shlex.split(command)

    assert isinstance(command, list), "Command must be a list of cmd arguments."

    logger.info(f"Running command: {shlex.join(command)}")

    if not log_file_path:
        # capture all output to stdout and stderr in current process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        stdout_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(process.stdout, logger, logging.DEBUG),
        )
        stderr_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(process.stderr, logger, logging.ERROR),
        )

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
        return_code = process.returncode
    else:
        logger.info(f"Logging output to: {log_file_path} ...")
        with open(log_file_path, "a", buffering=1) as log_file:
            result = subprocess.run(
                command,
                shell=shell,
                stdout=log_file,
                stderr=log_file,
                check=check,
                text=True,
                env=env,
            )
            return_code = result.returncode

    if return_code != 0:
        if check:
            raise RuntimeError(f"⛔ Command failed with return code: {return_code}")
        else:
            logger.error(f"⛔ Command failed with return code: {return_code}, check=False, continuing...")
    return return_code


default_dotenv_path = get_repo_root_path() / ".env"


def load_dotenv(dotenv_path=default_dotenv_path, logger=logger):
    """Manually loads environment variables from a .env file"""
    dotenv_file = Path(dotenv_path)

    if not dotenv_file.exists():
        return False

    with dotenv_file.open("r") as file:
        for line in file:
            # Ignore empty lines and comments
            if line.strip() == "" or line.startswith("#"):
                continue
            # Parse key=value pairs
            key, value = map(str.strip, line.split("=", 1))
            os.environ[key] = value
            logger.info(f"loaded env var from .env file: {key}")
    return True


def write_dotenv(env_vars, dotenv_path=default_dotenv_path, logger=logger):
    """Writes environment variables to a .env file"""
    dotenv_path = Path(dotenv_path)

    with open(dotenv_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")
            logger.info(f"writting env var to .env file: {key}")
    logger.info(f"Environment variables written to {dotenv_path}")
    return True


def map_configs_by_attr(config_list: List[Config], attr: str) -> Dict[str, Config]:  # noqa: F821
    """Returns a dictionary mapping the specified attribute to the Config instances.

    Raises:
        ValueError: If duplicate keys are found.
    """
    attr_map = {}
    for config in config_list:
        key = getattr(config, attr)
        if key in attr_map:
            raise ValueError(f"Duplicate key found: {key}")
        attr_map[key] = config
    return attr_map


def get_default_hf_home_path() -> Path:
    # first: check if HOST_HF_HOME is set in env
    # second: check if HF_HOME is set in env
    # third: default to ~/.cache/huggingface
    default_hf_home = os.getenv(
        "HOST_HF_HOME",
        str(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")),
    )
    return Path(default_hf_home)


def get_weights_hf_cache_dir(hf_repo: str) -> Path:
    local_repo_name = hf_repo.replace("/", "--")
    hf_home = get_default_hf_home_path()

    # Check both potential snapshot directory locations
    possible_snapshot_dirs = [
        hf_home / f"models--{local_repo_name}" / "snapshots",
        hf_home / "hub" / f"models--{local_repo_name}" / "snapshots",
    ]

    valid_snapshot_dir = None
    for snapshot_dir in possible_snapshot_dirs:
        if snapshot_dir.is_dir():
            snapshots = list(snapshot_dir.glob("*"))
            if snapshots:
                valid_snapshot_dir = snapshot_dir
                break

    if not valid_snapshot_dir:
        return None

    # Get the most recent snapshot
    snapshots = list(valid_snapshot_dir.glob("*"))
    most_recent_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)

    return most_recent_snapshot


def is_streaming_enabled_for_whisper(self) -> bool:
    """Determine if streaming is enabled for the Whisper model based on CLI args. Default to True if not set."""
    logger.info("Checking if streaming is enabled for Whisper model")
    cli_args = getattr(self.model_spec, "cli_args", {})

    # Check if streaming arg exists and has a valid value
    streaming_value = cli_args.get("streaming")
    if streaming_value is None:
        return True

    # Convert to string and check if it's 'true'
    streaming_enabled = str(streaming_value).lower() == "true"

    return streaming_enabled


def is_preprocessing_enabled_for_whisper(self) -> bool:
    """Determine if preprocessing is enabled for the Whisper model based on CLI args. Default to True if not set."""
    logger.info("Checking if preprocessing is enabled for Whisper model")

    cli_args = getattr(self.model_spec, "cli_args", {})
    preprocessing_value = cli_args.get("preprocessing")
    if preprocessing_value is None:
        return True

    # Convert to string and check if it's 'true'
    preprocessing_enabled = str(preprocessing_value).lower() == "true"

    return preprocessing_enabled


def is_sdxl_num_prompts_enabled(self) -> int:
    """Determine the number of prompts to use for SDXL based on CLI args. Default to 100 if not set."""
    logger.info("Checking if sdxl_num_prompts is set")

    cli_args = getattr(self.model_spec, "cli_args", {})
    sdxl_num_prompts = cli_args.get("sdxl_num_prompts")
    if sdxl_num_prompts is None:
        return SDXL_DEFAULT_NUM_PROMPTS

    # Convert to int and return
    num_prompts = int(sdxl_num_prompts)
    if (
        num_prompts < SDXL_LOWER_BOUND_NUM_PROMPTS
        or num_prompts > SDXL_UPPER_BOUND_NUM_PROMPTS
    ):
        return SDXL_DEFAULT_NUM_PROMPTS

    return num_prompts


def get_num_calls(self) -> int:
    """Get number of calls from benchmark parameters."""
    logger.info("Extracting number of calls from benchmark parameters")

    # Guard clause: Handle single config object case (evals)
    if hasattr(self.all_params, "tasks") and not isinstance(
        self.all_params, (list, tuple)
    ):
        return 2  # hard coding for evals

    # Handle list/iterable case (benchmarks)
    if isinstance(self.all_params, (list, tuple)):
        return next(
            (
                getattr(param, "num_eval_runs", 2)
                for param in self.all_params
                if hasattr(param, "num_eval_runs")
            ),
            2,
        )

    return 2
