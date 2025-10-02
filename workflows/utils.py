# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import logging
import os
import subprocess
import shlex
import tempfile
import threading
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import uuid

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


def get_run_id(timestamp, model_id, workflow):
    def _short_uuid():
        """Return 8-character random UUID"""
        # Generate UUID4 (random)
        u = uuid.uuid4()
        # Convert to bytes and encode with URL-safe base64
        return base64.urlsafe_b64encode(u.bytes)[:8].decode('utf-8')
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
                test_path = tmpfile.name
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
        return_code = process.returncode
    else:
        logger.info(f"Logging output to: {log_file_path} ...")
        with open(log_file_path, "a", buffering=1) as log_file:
            result = subprocess.run(
                command,
                shell=shell,
                stdout=log_file,
                stderr=log_file,
                check=False,
                text=True,
                env=env,
            )
            return_code = result.returncode

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


@dataclass
class PerformanceTarget:
    ttft_ms: float = None
    tput_user: float = None
    tput: float = None
    tolerance: float = 0.0


@dataclass
class BenchmarkTaskParams:
    isl: int = None
    osl: int = None
    max_concurrency: int = None
    num_prompts: int = None
    image_height: int = None
    image_width: int = None
    images_per_prompt: int = 0
    task_type: str = "text"
    theoretical_ttft_ms: float = None
    theoretical_tput_user: float = None
    targets: Dict[str, PerformanceTarget] = field(default_factory=dict)
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.10,
            "customer_complete": 0.50,
            "customer_sellable": 0.80,
        }
    )

    # has to go in here so init can read it
    num_inference_steps: int = None  # Used for CNN models


    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        for target_name, peak_perf in self.target_peak_perf.items():
            if target_name not in self.targets.keys():
                if self.theoretical_ttft_ms or self.theoretical_tput_user:
                    self.targets[target_name] = PerformanceTarget(
                        ttft_ms=self.theoretical_ttft_ms / peak_perf
                        if self.theoretical_ttft_ms
                        else None,
                        tput_user=self.theoretical_tput_user * peak_perf
                        if self.theoretical_tput_user
                        else None,
                    )

@dataclass
class BenchmarkTaskParamsCNN(BenchmarkTaskParams):
    num_eval_runs: int = 15
    target_peak_perf: Dict[str, float] = field(
        default_factory=lambda: {
            "customer_functional": 0.30,
            "customer_complete": 0.70,
            "customer_sellable": 0.80,
        }
    )
    
    def __post_init__(self):
        self._infer_data()
    
    def _infer_data(self):
        super()._infer_data()
