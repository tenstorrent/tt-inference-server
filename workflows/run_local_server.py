# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
import atexit
import time
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from workflows.utils import (
    get_repo_root_path,
    get_default_workflow_root_log_dir,
    ensure_readwriteable_dir,
)
from workflows.model_config import MODEL_CONFIGS
from workflows.log_setup import clean_log_file
from workflows.workflow_types import WorkflowType, DeviceTypes

logger = logging.getLogger("run_log")


def run_local_server(args, setup_config):
    """
    Run the vLLM inference server locally (not in Docker).
    This requires tt-metal and vLLM to be installed locally.
    """
    model_name = args.model
    repo_root_path = get_repo_root_path()
    model_config = MODEL_CONFIGS[model_name]
    service_port = args.service_port
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Setup logging
    local_log_file_dir = get_default_workflow_root_log_dir() / "local_server"
    ensure_readwriteable_dir(local_log_file_dir)
    local_log_file_path = (
        local_log_file_dir
        / f"vllm_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )
    
    # Get device configuration
    device = DeviceTypes.from_string(args.device)
    mesh_device_str = DeviceTypes.to_mesh_device_str(device)
    
    # Set up environment variables
    env = os.environ.copy()
    env.update({
        "SERVICE_PORT": str(service_port),
        "MESH_DEVICE": mesh_device_str,
        "HF_MODEL_REPO_ID": model_config.hf_model_repo,
    })
    
    # Load additional environment variables from setup config
    if setup_config.env_file and Path(setup_config.env_file).exists():
        with open(setup_config.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env[key] = value
    
    # Set up the vLLM server command
    vllm_script_path = repo_root_path / "vllm-tt-metal-llama3" / "src" / "run_vllm_api_server.py"
    
    if not vllm_script_path.exists():
        raise FileNotFoundError(f"vLLM server script not found at {vllm_script_path}")
    
    # Add the vllm-tt-metal-llama3/src directory to Python path
    vllm_src_path = str(vllm_script_path.parent)
    if vllm_src_path not in sys.path:
        sys.path.insert(0, vllm_src_path)
    
    # Also add the utils directory
    utils_path = str(repo_root_path / "utils")
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    logger.info(f"Running local vLLM server with log file: {local_log_file_path}")
    logger.info(f"Model: {model_config.hf_model_repo}")
    logger.info(f"Device: {args.device} (MESH_DEVICE: {mesh_device_str})")
    logger.info(f"Port: {service_port}")
    
    # Open log file
    local_log_file = open(local_log_file_path, "w", buffering=1)
    
    try:
        # Start the vLLM server process
        process = subprocess.Popen(
            [sys.executable, str(vllm_script_path)],
            stdout=local_log_file,
            stderr=local_log_file,
            text=True,
            env=env,
            cwd=str(repo_root_path)
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Check if the process is still running
        if process.poll() is not None:
            logger.error("vLLM server process exited unexpectedly")
            local_log_file.close()
            return
        
        logger.info(f"Started local vLLM server with PID: {process.pid}")
        
        # Set up cleanup function
        def teardown_local_server():
            logger.info("atexit: Stopping local inference server ...")
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate gracefully, forcing kill")
                    process.kill()
            local_log_file.close()
            # Clean up log file
            clean_log_file(local_log_file_path)
            logger.info("run_local_server cleanup finished.")
        
        # Register cleanup function
        atexit.register(teardown_local_server)
        
        # For server-only workflow, just keep running
        skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
        if WorkflowType.from_string(args.workflow) not in skip_workflows:
            # For other workflows, the process will be managed by the workflow runner
            pass
        else:
            # For server-only workflow, keep the process running
            logger.info(f"Local server running on port {service_port}")
            logger.info(f"Server logs: {local_log_file_path}")
            logger.info(f"Stop server with: kill {process.pid}")
            
            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                teardown_local_server()
    
    except Exception as e:
        logger.error(f"Failed to start local server: {e}")
        local_log_file.close()
        raise
    
    return process


