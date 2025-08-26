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

logger = logging.getLogger("run_log")


def setup_local_server_logger(log_file_path: Path):
    """
    Set up a file handler for the local server logger to write to the log file.
    
    Args:
        log_file_path: Path to the log file
    """
    # Create a file handler for the local server logs
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    logger.info(f"Local server logger configured to write to: {log_file_path}")
    return file_handler


def find_tt_metal_venv(tt_metal_home: Path, tt_metal_python_venv_dir: str = None) -> Path:
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
            raise FileNotFoundError(f"Provided venv path does not exist or is invalid: {venv_path}")
    
    # List of possible venv names to search for
    possible_venv_names = [
        os.getenv("PYTHON_ENV_DIR", "").split("/")[-1] if os.getenv("PYTHON_ENV_DIR") else None,
        "python_env",
        "python_env_vllm",
    ]
    
    # Filter out None values
    possible_venv_names = [name for name in possible_venv_names if name]
    
    logger.info(f"Searching for TT-Metal venv in {tt_metal_home} with names: {possible_venv_names}")
    
    for venv_name in possible_venv_names:
        venv_path = tt_metal_home / venv_name
        python_path = venv_path / "bin" / "python"
        
        if venv_path.exists() and python_path.exists():
            logger.info(f"Found TT-Metal venv: {venv_path}")
            return venv_path
    
    raise FileNotFoundError(
        f"No valid TT-Metal virtual environment found in {tt_metal_home}. "
        f"Searched for: {possible_venv_names}. "
        f"Please ensure TT-Metal is properly set up with a virtual environment."
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
            timeout=30
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
        "MODEL_WEIGHTS_PATH": str(setup_config.host_model_weights_mount_dir),
        "TT_LLAMA_TEXT_VER": model_spec.impl.impl_id,
        "TT_MODEL_SPEC_JSON_PATH": str(json_fpath),
        "TT_METAL_HOME": str(tt_metal_home),
        "PYTHONPATH": f"{tt_metal_home}:{repo_root_path / 'vllm-tt-metal-llama3' / 'src'}",
    }
    
    # Add vLLM specific environment variables
    if hasattr(model_spec, 'device_id') and model_spec.cli_args.device_id:
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
        json_fpath: Path to model spec JSON file
    """
    args = model_spec.cli_args
    
    # Step 1: Validate TT_METAL_HOME
    tt_metal_home_str = os.getenv("TT_METAL_HOME")
    if not tt_metal_home_str:
        raise ValueError("TT_METAL_HOME environment variable must be set for --local-server")
    
    tt_metal_home = Path(tt_metal_home_str)
    if not tt_metal_home.exists():
        raise ValueError(f"TT_METAL_HOME does not exist: {tt_metal_home}")
    
    logger.info(f"Using TT_METAL_HOME: {tt_metal_home}")
    
    # Step 2: Find and validate virtual environment
    try:
        venv_path = find_tt_metal_venv(tt_metal_home, args.tt_metal_python_venv_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    
    
    # Step 4: Set up logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    local_server_log_dir = get_default_workflow_root_log_dir() / "local_server"
    ensure_readwriteable_dir(local_server_log_dir)
    local_server_log_path = (
        local_server_log_dir
        / f"vllm_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )
    
    # Step 3: Check vLLM installation
    if not check_vllm_installation(venv_path):
        raise RuntimeError(
            f"vLLM is not installed in the virtual environment: {venv_path}. "
            f"Please install vLLM in the TT-Metal virtual environment."
        )
    # Step 5: Set up environment
    env = setup_local_server_environment(model_spec, setup_config, json_fpath)
    
    # Step 6: Prepare command
    python_path = venv_path / "bin" / "python"
    repo_root_path = get_repo_root_path()
    server_script_path = repo_root_path / "vllm-tt-metal-llama3" / "src" / "run_vllm_api_server.py"
    
    if not server_script_path.exists():
        raise FileNotFoundError(f"vLLM server script not found: {server_script_path}")
    
    # Command to run the server
    command = [str(python_path), str(server_script_path)]
    
    logger.info(f"Starting local vLLM server with command: {shlex.join(command)}")
    logger.info(f"Server logs will be written to: {local_server_log_path}")
    logger.info(f"Virtual environment: {venv_path}")
    
    # Step 7: Set up logger to write to file
    file_handler = setup_local_server_logger(local_server_log_path)
    
    try:
        # Step 8: Run the server using run_command for stdout/stderr streaming
        logger.info("Starting vLLM server with streaming output to console and file...")
        
        return_code = run_command(
            command=command,
            logger=logger,
            log_file_path=None,  # No separate log file, use logger's file handler
            env=env
        )
        
        if return_code != 0:
            raise RuntimeError(f"vLLM server failed with return code: {return_code}")
            
        logger.info("✅ vLLM server completed successfully")
        
    finally:
        # Remove the file handler to avoid duplicate logging
        logger.removeHandler(file_handler)
        file_handler.close()
    
    # Note: run_command is synchronous, so the server has already completed
    # No need for process management or cleanup since run_command handles it
    return None
