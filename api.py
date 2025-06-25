from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os
import logging
from pathlib import Path
from run import main as run_main, parse_arguments, WorkflowType, DeviceTypes
from workflows.model_config import MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TT Inference Server API",
    description="API wrapper for the TT Inference Server run script",
    version="1.0.0"
)

class RunRequest(BaseModel):
    model: str
    workflow: str
    device: str
    impl: Optional[str] = None
    local_server: Optional[bool] = False
    docker_server: Optional[bool] = False
    interactive: Optional[bool] = False
    workflow_args: Optional[str] = None
    service_port: Optional[str] = "7000"
    disable_trace_capture: Optional[bool] = False
    dev_mode: Optional[bool] = False
    override_docker_image: Optional[str] = None
    device_id: Optional[str] = None
    override_tt_config: Optional[str] = None
    vllm_override_args: Optional[str] = None
    # Optional secrets - can be passed through API if not set in environment
    jwt_secret: Optional[str] = None
    hf_token: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "TT Inference Server API is running"}

@app.post("/run")
async def run_inference(request: RunRequest):
    try:
        # Ensure we're in the correct working directory
        script_dir = Path(__file__).parent.absolute()
        original_cwd = Path.cwd()
        
        logger.info(f"Current working directory: {original_cwd}")
        logger.info(f"Script directory: {script_dir}")
        
        if original_cwd != script_dir:
            logger.info(f"Changing working directory from {original_cwd} to {script_dir}")
            os.chdir(script_dir)
        else:
            logger.info("Already in correct working directory")
        
        # Set required environment variables for automatic setup
        env_vars_to_set = {
            "AUTOMATIC_HOST_SETUP": "True",
            "HOST_HF_HOME": "/root/.cache/huggingface"
        }
        
        # Handle secrets - use from request if provided and not already in environment
        if request.jwt_secret and not os.getenv("JWT_SECRET"):
            logger.info("Setting JWT_SECRET from request")
            env_vars_to_set["JWT_SECRET"] = request.jwt_secret
        elif not os.getenv("JWT_SECRET"):
            logger.warning("JWT_SECRET not set - this may cause issues")
            
        if request.hf_token and not os.getenv("HF_TOKEN"):
            logger.info("Setting HF_TOKEN from request")
            env_vars_to_set["HF_TOKEN"] = request.hf_token
        elif not os.getenv("HF_TOKEN"):
            logger.warning("HF_TOKEN not set - this may cause issues with model downloads")
            
        # Set environment variables
        for key, value in env_vars_to_set.items():
            if key in ["JWT_SECRET", "HF_TOKEN"]:
                logger.info(f"Setting environment variable: {key}=[REDACTED]")
            else:
                logger.info(f"Setting environment variable: {key}={value}")
            os.environ[key] = value

        
        # Convert the request to command line arguments
        sys.argv = ["run.py"]  # Reset sys.argv
        
        # Add required arguments
        sys.argv.extend(["--model", request.model])
        sys.argv.extend(["--workflow", request.workflow])
        sys.argv.extend(["--device", request.device])
        sys.argv.extend(["--docker-server"])
        # sys.argv.extend(["--dev-mode"])   # TODO: Uncomment this for dev branch
        sys.argv.extend(["--service-port", "7000"])
        
        # Add optional arguments if they are set
        if request.impl:
            sys.argv.extend(["--impl", request.impl])
        if request.local_server:
            sys.argv.append("--local-server")
        if request.interactive:
            sys.argv.append("--interactive")
        if request.workflow_args:
            sys.argv.extend(["--workflow-args", request.workflow_args])
        if request.disable_trace_capture:
            sys.argv.append("--disable-trace-capture")
        if request.override_docker_image:
            sys.argv.extend(["--override-docker-image", request.override_docker_image])
        # TODO: Uncomment this for dev branch
        # if request.device_id:
        #     sys.argv.extend(["--device-id", request.device_id])
        if request.override_tt_config:
            sys.argv.extend(["--override-tt-config", request.override_tt_config])
        if request.vllm_override_args:
            sys.argv.extend(["--vllm-override-args", request.vllm_override_args])

        # Log the command being executed
        logger.info(f"Executing command: {' '.join(sys.argv)}")
        
        # Log current environment variables that might be relevant
        relevant_env_vars = ["JWT_SECRET", "HF_TOKEN", "AUTOMATIC_HOST_SETUP", "SERVICE_PORT", "HOST_HF_HOME"]
        for var in relevant_env_vars:
            value = os.getenv(var)
            if value:
                # Don't log the actual secrets, just indicate they're set
                if var in ["JWT_SECRET", "HF_TOKEN"]:
                    logger.info(f"Environment variable {var}: [SET]")
                else:
                    logger.info(f"Environment variable {var}: {value}")
            else:
                logger.info(f"Environment variable {var}: [NOT SET]")
        
        try:
            # Run the main function
            logger.info("Starting run_main()...")
            return_code, container_info = run_main()
            logger.info(f"run_main() completed with return code: {return_code}")
            logger.info(f"container_info:= {container_info}")

            if return_code == 0:
                # Store container info in the registry
                container_name = container_info["container_name"]
                logger.info(f"container_name:= {container_name}")
                
                # For docker server workflow, try to get container information from logs
                response_data = {"status": "success", "message": "Inference completed successfully. Container info: " + container_name, "container_name": container_name}

                # Change container network to tt_studio_network
                try:
                    import docker
                    client = docker.from_env()
                    
                    # List all running containers
                    all_containers = client.containers.list()
                    logger.info(f"all_containers:= {all_containers}")
                    
                    # Set of known containers to exclude
                    known_containers = {"tt_studio_agent", "tt_studio_frontend", "tt_studio_backend_api", "tt_studio_chroma"}
                    
                    # Find the new container (not in the known set)
                    new_container = None
                    for container in all_containers:
                        if container.name not in known_containers:
                            new_container = container
                            break
                    
                    if new_container:
                        original_name = new_container.name
                        logger.info(f"Found new container: {original_name}")
                        
                        # Connect to network
                        network = client.networks.get("tt_studio_network")
                        network.connect(new_container)
                        logger.info(f"Connected container {original_name} to tt_studio_network")
                        
                        # Rename the container to container_info["container_name"]
                        target_name = container_info["container_name"]
                        new_container.rename(target_name)
                        logger.info(f"Renamed container from {original_name} to {target_name}")
                    else:
                        logger.error("No new container found to connect to network")
                        
                except Exception as e:
                    logger.error(f"Failed to connect container to network: {str(e)}")
                    # Continue execution even if network connection fails
                
                return response_data
            else:
                raise HTTPException(status_code=500, detail=f"Inference failed with return code: {return_code}")
        finally:
            # Always restore the original working directory
            if original_cwd != script_dir:
                logger.info(f"Restoring working directory to {original_cwd}")
                os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}", exc_info=True)
        # Restore working directory in case of exception
        if 'original_cwd' in locals() and 'script_dir' in locals() and original_cwd != script_dir:
            os.chdir(original_cwd)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {"models": list(set(config.model_name for _, config in MODEL_CONFIGS.items()))}

@app.get("/workflows")
async def get_available_workflows():
    """Get list of available workflows"""
    return {"workflows": [w.name.lower() for w in WorkflowType]}

@app.get("/devices")
async def get_available_devices():
    """Get list of available devices"""
    return {"devices": [d.name.lower() for d in DeviceTypes]} 