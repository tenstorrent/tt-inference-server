# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import subprocess
import shlex
import atexit
import time
import logging
from datetime import datetime
from pathlib import Path

from workflows.utils import (
    get_repo_root_path,
    get_default_workflow_root_log_dir,
    ensure_readwriteable_dir,
    run_command,
)
from workflows.workflow_types import WorkflowType
from workflows.model_spec import ModelType
from workflows.log_setup import clean_log_file

logger = logging.getLogger("run_log")


def find_tt_metal_venv(
    tt_metal_home: Path, tt_metal_python_venv_dir: str = None
) -> Path:
    """
    Find the TT-Metal virtual environment directory.

    Args:
        tt_metal_home: Path to TT_METAL_HOME
        tt_metal_python_venv_dir: Override path from CLI args

    Returns:
        Path to the virtual environment directory

    Raises:
        FileNotFoundError: If no valid venv is found
    """
    if tt_metal_python_venv_dir:
        venv_path = Path(tt_metal_python_venv_dir)
        if venv_path.exists() and (venv_path / "bin" / "python").exists():
            logger.info(f"Using provided TT-Metal venv: {venv_path}")
            return venv_path
        else:
            raise FileNotFoundError(
                f"Provided venv path does not exist or is invalid: {venv_path}"
            )

    # Check PYTHON_ENV_DIR first as direct path
    python_env_dir = os.getenv("PYTHON_ENV_DIR")
    if python_env_dir:
        venv_path = Path(python_env_dir)
        python_path = venv_path / "bin" / "python"
        if venv_path.exists() and python_path.exists():
            logger.info(f"Using PYTHON_ENV_DIR venv: {venv_path}")
            return venv_path
        else:
            logger.warning(
                f"PYTHON_ENV_DIR is set but invalid: {venv_path}. "
                f"Falling back to searching in TT_METAL_HOME."
            )

    # List of possible venv names to search for in tt_metal_home
    possible_venv_names = [
        "python_env",
        "python_env_vllm",
    ]

    logger.info(
        f"Searching for TT-Metal venv in {tt_metal_home} with names: {possible_venv_names}"
    )

    for venv_name in possible_venv_names:
        venv_path = tt_metal_home / venv_name
        python_path = venv_path / "bin" / "python"

        if venv_path.exists() and python_path.exists():
            logger.info(f"Found TT-Metal venv: {venv_path}")
            return venv_path

    raise FileNotFoundError(
        f"No valid TT-Metal virtual environment found. "
        f"Checked PYTHON_ENV_DIR: {python_env_dir}. "
        f"Searched in {tt_metal_home} for: {possible_venv_names}. "
        f"Please ensure TT-Metal is properly set up with a virtual environment or set PYTHON_ENV_DIR to a valid venv path."
    )


def check_vllm_installation(venv_path: Path) -> bool:
    """
    Check if vLLM is installed in the virtual environment.

    Args:
        venv_path: Path to the virtual environment

    Returns:
        True if vLLM is installed, False otherwise
    """
    python_path = venv_path / "bin" / "python"

    try:
        # Check if vLLM can be imported
        result = subprocess.run(
            [str(python_path), "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            vllm_version = result.stdout.strip()
            logger.info(f"vLLM is installed, version: {vllm_version}")
            return True
        else:
            logger.error(f"vLLM import failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout while checking vLLM installation")
        return False
    except Exception as e:
        logger.error(f"Error checking vLLM installation: {e}")
        return False


def install_vllm(venv_path: Path, vllm_commit: str):
    """
    Install vLLM from the specified commit in the virtual environment.

    Args:
        venv_path: Path to the virtual environment
        vllm_commit: Git commit hash to checkout

    Raises:
        RuntimeError: If installation fails
    """
    logger.info(f"Installing vLLM commit {vllm_commit} in venv: {venv_path}")
    
    try:
        # Install vLLM directly from GitHub using commit SHA
        pip_path = venv_path / "bin" / "pip"
        vllm_git_url = f"git+https://github.com/tenstorrent/vllm.git@{vllm_commit}"
        
        logger.info(f"Installing vLLM directly from GitHub: {vllm_git_url}")
        
        install_command = [
            str(pip_path), 
            "install", 
            vllm_git_url,
            "--extra-index-url", 
            "https://download.pytorch.org/whl/cpu"
        ]
        
        install_return_code = run_command(install_command, logger=logger)
        if install_return_code != 0:
            raise RuntimeError(f"vLLM installation failed with return code: {install_return_code}")
        
        logger.info("vLLM installation completed successfully")
        
        # Verify installation
        if check_vllm_installation(venv_path):
            logger.info("vLLM installation verified successfully")
        else:
            raise RuntimeError("vLLM installation verification failed")
            
    except Exception as e:
        logger.error(f"Failed to install vLLM: {e}")
        raise RuntimeError(f"vLLM installation failed: {e}")


def install_additional_requirements(venv_path: Path):
    """
    Install additional requirements from vllm-tt-metal-llama3/requirements.txt.

    Args:
        venv_path: Path to the virtual environment

    Raises:
        RuntimeError: If requirements installation fails
    """
    pip_path = venv_path / "bin" / "pip"
    repo_root = get_repo_root_path()
    requirements_file = repo_root / "vllm-tt-metal-llama3" / "requirements.txt"
    
    if requirements_file.exists():
        logger.info(f"Installing additional requirements from: {requirements_file}")
        
        requirements_command = [
            str(pip_path),
            "install", 
            "-r",
            str(requirements_file)
        ]
        
        requirements_return_code = run_command(requirements_command, logger=logger)
        if requirements_return_code != 0:
            raise RuntimeError(f"Requirements installation failed with return code: {requirements_return_code}")
            
        logger.info("Additional requirements installation completed successfully")
    else:
        logger.warning(f"Requirements file not found: {requirements_file}")


def ensure_vllm_installation(venv_path: Path, vllm_commit: str):
    """
    Ensure vLLM is installed in the virtual environment, installing if necessary.

    Args:
        venv_path: Path to the virtual environment
        vllm_commit: Git commit hash to install if vLLM is not present

    Raises:
        RuntimeError: If installation fails
    """
    if not check_vllm_installation(venv_path):
        logger.info("vLLM not found in virtual environment, installing...")
        install_vllm(venv_path, vllm_commit)
        install_additional_requirements(venv_path)
    else:
        logger.info("vLLM is already installed in virtual environment")
        # Still install additional requirements in case they're missing
        install_additional_requirements(venv_path)


def setup_local_server_environment(model_spec, setup_config, json_fpath: Path) -> dict:
    """
    Set up environment variables for local server execution.

    Args:
        model_spec: Model specification object
        setup_config: Setup configuration object with host paths
        json_fpath: Path to model spec JSON file

    Returns:
        Dictionary of environment variables
    """
    # Start with current environment
    env = os.environ.copy()

    # Set up paths similar to docker server
    repo_root_path = get_repo_root_path()

    # TT-Metal specific environment variables
    tt_metal_home = Path(os.getenv("TT_METAL_HOME"))
    if not tt_metal_home.exists():
        raise ValueError(f"TT_METAL_HOME not set or does not exist: {tt_metal_home}")

    # Set environment variables needed by run_vllm_api_server.py
    # Use host paths from setup_config similar to docker server
    local_env_vars = {
        "CACHE_ROOT": str(setup_config.host_model_volume_root),
        "TT_CACHE_PATH": str(setup_config.host_tt_metal_cache_dir),
        "MODEL_WEIGHTS_PATH": str(setup_config.host_model_weights_snapshot_dir),
        "TT_LLAMA_TEXT_VER": model_spec.impl.impl_id,
        "TT_MODEL_SPEC_JSON_PATH": str(json_fpath),
        "TT_METAL_HOME": str(tt_metal_home),
        "PYTHONPATH": f"{tt_metal_home}:{repo_root_path}",
    }

    # Add vLLM specific environment variables
    if hasattr(model_spec, "device_id") and model_spec.cli_args.device_id:
        # Set device-specific environment if needed
        pass

    # Update environment with local variables
    env.update(local_env_vars)

    # Log environment setup
    logger.info("Local server environment variables:")
    for key, value in local_env_vars.items():
        logger.info(f"  {key}={value}")

    return env


def run_local_server(model_spec, setup_config, json_fpath: Path):
    """
    Run the vLLM inference server locally using TT-Metal virtual environment.

    Args:
        model_spec: Model specification object
        setup_config: Setup configuration object with host paths
        json_fpath: Path to model spec JSON file
    """
    args = model_spec.cli_args

    # Step 1: Validate TT_METAL_HOME
    tt_metal_home_str = os.getenv("TT_METAL_HOME")
    if not tt_metal_home_str:
        raise ValueError(
            "TT_METAL_HOME environment variable must be set for --local-server"
        )

    tt_metal_home = Path(tt_metal_home_str)
    if not tt_metal_home.exists():
        raise ValueError(f"TT_METAL_HOME does not exist: {tt_metal_home}")

    logger.info(f"Using TT_METAL_HOME: {tt_metal_home}")

    # Step 2: Find and validate virtual environment
    venv_path = find_tt_metal_venv(tt_metal_home, args.tt_metal_python_venv_dir)

    # Step 3: Set up logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_server_log_dir = get_default_workflow_root_log_dir() / "local_server"
    ensure_readwriteable_dir(local_server_log_dir)
    local_server_log_path = (
        local_server_log_dir
        / f"vllm_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )

    # Step 4: Ensure vLLM installation
    ensure_vllm_installation(venv_path, model_spec.vllm_commit)
    
    # Step 5: Set up environment
    env = setup_local_server_environment(model_spec, setup_config, json_fpath)

    # Step 6: Prepare command
    python_path = venv_path / "bin" / "python"
    repo_root_path = get_repo_root_path()
    server_script_path = (
        repo_root_path / "vllm-tt-metal-llama3" / "src" / "run_vllm_api_server.py"
    )

    if not server_script_path.exists():
        raise FileNotFoundError(f"vLLM server script not found: {server_script_path}")

    # Command to run the server
    command = [str(python_path), str(server_script_path)]

    logger.info(f"Starting local vLLM server with command: {shlex.join(command)}")
    logger.info(f"Server logs will be written to: {local_server_log_path}")
    logger.info(f"Virtual environment: {venv_path}")

    # Step 7: Open log file and start process in background
    local_server_log_file = open(local_server_log_path, "w", buffering=1)
    logger.info(f"Starting local vLLM server with log file: {local_server_log_path}")

    # Start process in background
    process = subprocess.Popen(
        command,
        stdout=local_server_log_file,
        stderr=local_server_log_file,
        text=True,
        env=env,
    )

    logger.info(f"Started local vLLM server process with PID: {process.pid}")

    # Step 8: Set up atexit cleanup based on workflow type
    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(args.workflow) not in skip_workflows:

        def teardown_local_server():
            logger.info("atexit: Stopping local vLLM server process...")
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate gracefully, killing it...")
                    process.kill()
                    process.wait()
            local_server_log_file.close()
            clean_log_file(local_server_log_path)
            logger.info("run_local_server cleanup finished.")

        atexit.register(teardown_local_server)
    else:

        def exit_log_messages():
            local_server_log_file.close()
            logger.info(f"Created local server process PID: {process.pid}")
            logger.info(f"Local server logs: {local_server_log_path}")
            logger.info(f"You can view logs via: tail -f {local_server_log_path}")
            logger.info(f"To stop the process run: kill {process.pid}")

        atexit.register(exit_log_messages)

    return process
