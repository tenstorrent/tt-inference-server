# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import os
import subprocess
import shlex
import threading
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


def get_run_id(timestamp, model, device, workflow):
    return f"{timestamp}_{model}_{device}_{workflow}"


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
            test_file = path / ".test_write_access"
            with test_file.open("w") as f:
                f.write("test")
            test_file.unlink()  # Remove test file after write test
            logger.info(f"Directory '{path}' is readable and writable.")
            return True
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
    command, logger, log_file_path=None, shell=False, copy_env=True, env=None
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
    else:
        logger.info(f"Logging output to: {log_file_path} ...")
        with open(log_file_path, "a", buffering=1) as log_file:
            _ = subprocess.run(
                command,
                shell=shell,
                stdout=log_file,
                stderr=log_file,
                check=True,
                text=True,
                env=env,
            )


def load_dotenv(dotenv_path=get_repo_root_path() / ".env", logger=logger):
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


def write_dotenv(env_vars, dotenv_path=get_repo_root_path() / ".env", logger=logger):
    """Writes environment variables to a .env file"""
    dotenv_path = Path(dotenv_path)

    with open(dotenv_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")
            logger.info(f"writting env var to .env file: {key}")
    logger.info(f"Environment variables written to {dotenv_path}")
    return True


def map_configs_by_attr(config_list: List["Config"], attr: str) -> Dict[str, "Config"]:  # noqa: F821
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


@dataclass
class BenchmarkTaskParams:
    isl: int
    osl: int
    max_concurrency: int
    num_prompts: int
    ref_ttft_ms: float = None
    ref_tput_user: float = None
    ref_tput: float = None
    tolerance: float = 0.10
