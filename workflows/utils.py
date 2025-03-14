# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys
import os
import subprocess
import shlex
from datetime import datetime
from pathlib import Path
# from workflows.logger import get_logger


def get_repo_root_path(marker: str = ".git") -> Path:
    """Return the root directory of the repository by searching for a marker file or directory."""
    current_path = Path(__file__).resolve().parent  # Start from the script's directory
    for parent in current_path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(
        f"Repository root not found. No '{marker}' found in parent directories."
    )


def get_default_workflow_root_log_dir():
    # docker env uses CACHE_ROOT
    default_dir_name = "workflow_logs"
    cache_root = os.getenv("CACHE_ROOT")
    if cache_root:
        default_workflow_root_log_dir = Path(cache_root) / default_dir_name
    else:
        default_workflow_root_log_dir = get_repo_root_path() / default_dir_name
    return default_workflow_root_log_dir


def get_logger(log_level=logging.DEBUG):
    # Create a custom logger
    logger = logging.getLogger("run_log")
    logger.setLevel(log_level)  # Set the minimum logging level

    # Disable propagation to prevent duplicate logs
    logger.propagate = False
    # prevent duplicate handlers
    if logger.handlers:
        return logger

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = get_default_workflow_root_log_dir()
    log_path = Path(log_dir) / f"run_log_{timestamp}.log"
    # Create handlers
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger


logger = get_logger()


def ensure_readwriteable_dir(path, raise_on_fail=True):
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


def run_command(command, shell=False, copy_env=True):
    if not copy_env:
        raise NotImplementedError("TODO")
    env = os.environ.copy()
    # TODO: force usage to always use argument list
    # use shlex to log full command before running
    logger.info("Running command: %s", command)
    result = subprocess.run(
        shlex.split(command),
        shell=shell,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    if result.stdout:
        logger.info("Stdout: %s", result.stdout)
    if result.stderr:
        logger.error("Stderr: %s", result.stderr)
    if result.returncode != 0:
        logger.error("Command failed with exit code %s", result.returncode)
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result


def load_dotenv(dotenv_path=get_repo_root_path() / ".env"):
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


def write_dotenv(env_vars, dotenv_path=get_repo_root_path() / ".env"):
    """Writes environment variables to a .env file"""
    dotenv_path = Path(dotenv_path)

    with open(dotenv_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")
            logger.info(f"writting env var to .env file: {key}")
    logger.info(f"Environment variables written to {dotenv_path}")
    return True
