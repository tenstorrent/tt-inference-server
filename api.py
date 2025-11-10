from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
import os
import logging
import time
import docker
import threading
import uuid
import re
import json
from collections import deque
from pathlib import Path
from run import main as run_main, parse_arguments, WorkflowType, DeviceTypes
from workflows.model_config import MODEL_CONFIGS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure FastAPI logger to also write to file
def setup_fastapi_file_logging():
    """Set up file logging for FastAPI - writes to persistent volume like backend"""
    try:
        # Get persistent storage volume path (FastAPI runs on host, so use HOST_PERSISTENT_STORAGE_VOLUME)
        host_persistent_volume = os.getenv("HOST_PERSISTENT_STORAGE_VOLUME")
        
        if not host_persistent_volume:
            # Fallback: try to infer from TT_STUDIO_ROOT
            tt_studio_root = os.getenv("TT_STUDIO_ROOT")
            if tt_studio_root:
                host_persistent_volume = os.path.join(tt_studio_root, "tt_studio_persistent_volume")
            else:
                # Last resort: infer from script location
                script_dir = Path(__file__).parent.absolute()
                if script_dir.name == "tt-inference-server":
                    host_persistent_volume = str(script_dir.parent / "tt_studio_persistent_volume")
                else:
                    host_persistent_volume = str(script_dir / "tt_studio_persistent_volume")
        
        host_persistent_volume = Path(host_persistent_volume)
        
        # Follow backend pattern: backend_volume/fastapi_logs/
        fastapi_logs_dir = host_persistent_volume / "backend_volume" / "fastapi_logs"
        fastapi_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Also create a simple fastapi.log in the root for backward compatibility
        # This matches the expectation from run.py and startup.sh
        tt_studio_root = os.getenv("TT_STUDIO_ROOT")
        if not tt_studio_root:
            # Infer from persistent volume path
            if host_persistent_volume.name == "tt_studio_persistent_volume":
                root_log_dir = host_persistent_volume.parent
            else:
                root_log_dir = host_persistent_volume
        else:
            root_log_dir = Path(tt_studio_root)
        
        root_log_file = root_log_dir / "fastapi.log"
        
        # Create file handlers
        # 1. Detailed log in backend_volume/fastapi_logs/ (following backend pattern)
        detailed_log_file = fastapi_logs_dir / "fastapi.log"
        detailed_handler = logging.FileHandler(detailed_log_file, mode='a')
        detailed_handler.setLevel(logging.DEBUG)
        
        # 2. Simple log in root (for backward compatibility)
        root_handler = logging.FileHandler(root_log_file, mode='a')
        root_handler.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        root_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        
        detailed_handler.setFormatter(detailed_formatter)
        root_handler.setFormatter(root_formatter)
        
        # Configure the module logger
        logger.addHandler(detailed_handler)
        logger.addHandler(root_handler)
        logger.setLevel(logging.DEBUG)
        
        # Also configure FastAPI's logger
        fastapi_logger = logging.getLogger("fastapi")
        fastapi_logger.addHandler(detailed_handler)
        fastapi_logger.addHandler(root_handler)
        fastapi_logger.setLevel(logging.INFO)
        fastapi_logger.propagate = False
        
        # Configure uvicorn loggers
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_logger.addHandler(detailed_handler)
        uvicorn_logger.addHandler(root_handler)
        uvicorn_logger.setLevel(logging.INFO)
        uvicorn_logger.propagate = False
        
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.addHandler(detailed_handler)
        uvicorn_access_logger.addHandler(root_handler)
        uvicorn_access_logger.setLevel(logging.INFO)
        uvicorn_access_logger.propagate = False
        
        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        uvicorn_error_logger.addHandler(detailed_handler)
        uvicorn_error_logger.addHandler(root_handler)
        uvicorn_error_logger.setLevel(logging.INFO)
        uvicorn_error_logger.propagate = False
        
        logger.info(f"FastAPI file logging configured - writing to {detailed_log_file} and {root_log_file}")
        logger.debug(f"Detailed log absolute path: {detailed_log_file.absolute()}")
        logger.debug(f"Root log absolute path: {root_log_file.absolute()}")
        
    except Exception as e:
        # Log to both stdout and try to log to a fallback location
        error_msg = f"Failed to setup FastAPI file logging: {e}"
        print(error_msg)
        logger.error(error_msg, exc_info=True)
        
        # Try to write error to a fallback log location
        try:
            fallback_log = Path(__file__).parent / "fastapi_setup_error.log"
            with open(fallback_log, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                import traceback
                f.write(traceback.format_exc())
        except:
            pass  # If even fallback fails, just continue

# Initialize file logging
setup_fastapi_file_logging()

# Global progress store with thread-safe access
progress_store: Dict[str, Dict[str, Any]] = {}
log_store: Dict[str, deque] = {}
progress_lock = threading.Lock()

# Maximum number of log messages to keep per job
MAX_LOG_MESSAGES = 100

# Regex pattern for structured progress signals
PROG_RE = re.compile(r"TT_PROGRESS stage=(\w+) pct=(\d{1,3}) msg=(.*)$")

class ProgressHandler(logging.Handler):
    """Custom logging handler to capture progress from run.py execution"""
    
    def __init__(self, job_id: str):
        super().__init__()
        self.job_id = job_id
        
        # Initialize log store for this job
        with progress_lock:
            if job_id not in log_store:
                log_store[job_id] = deque(maxlen=MAX_LOG_MESSAGES)
        
    def emit(self, record):
        message = record.getMessage()
        
        # Store raw log message
        with progress_lock:
            if self.job_id in log_store:
                log_store[self.job_id].append({
                    "timestamp": record.created,
                    "level": record.levelname,
                    "message": message
                })
        
        # 1) Structured DEBUG path - prefer this when available
        structured_parsed = False
        if record.levelno <= logging.DEBUG:
            m = PROG_RE.search(message)
            if m:
                stage, pct, text = m.group(1), int(m.group(2)), m.group(3)
                status = "running"
                if stage == "complete":
                    status = "completed"
                elif stage == "error":
                    status = "error"

                with progress_lock:
                    if self.job_id in progress_store:
                        cur = progress_store[self.job_id]
                        prev = cur.get("progress", 0)
                        pct = max(prev, pct)  # monotonic clamp
                        progress_store[self.job_id].update({
                            "status": status,
                            "stage": stage,
                            "progress": pct,
                            "message": text[:200],
                            "last_updated": time.time(),
                        })
                    else:
                        # Initialize if not exists
                        progress_store[self.job_id] = {
                            "status": status,
                            "stage": stage,
                            "progress": pct,
                            "message": text[:200],
                            "last_updated": time.time(),
                        }
                structured_parsed = True

        # 2) Fallback: existing INFO-based heuristics (only if structured parsing didn't work)
        if not structured_parsed:
            stage = "unknown"
            progress = 0
            status = "running"
        
            # Based on the fastapi.log patterns, parse deployment stages
            if any(keyword in message.lower() for keyword in ["validate_runtime_args", "handle_secrets", "validate_local_setup"]):
                stage = "initialization"
                progress = 5
            elif any(keyword in message.lower() for keyword in ["setup_host", "setting up python venv", "loaded environment"]):
                stage = "setup"
                progress = 15
            elif any(keyword in message.lower() for keyword in ["downloading model", "huggingface-cli download", "setup already completed"]):
                stage = "model_preparation"
                progress = 40
            elif any(keyword in message.lower() for keyword in ["docker run command", "running docker container"]):
                stage = "container_setup"
                progress = 70
            elif any(keyword in message.lower() for keyword in ["searching for container", "looking for container"]):
                stage = "finalizing"
                progress = 85
            elif any(keyword in message.lower() for keyword in ["connected container", "tt_studio_network"]):
                stage = "finalizing"
                progress = 90
            elif "renamed container" in message.lower():
                # This is the KEY indicator that deployment is complete!
                stage = "complete"
                progress = 100
                status = "completed"
            elif "✅" in message or "completed successfully" in message.lower():
                stage = "complete"
                progress = 100
                status = "completed"
            elif any(keyword in message for keyword in ["⛔", "Error", "Failed", "error"]):
                status = "error"
                stage = "error"
                
            # Update progress store (only if we have meaningful progress)
            if progress > 0 or status in ["error", "completed"]:
                with progress_lock:
                    if self.job_id in progress_store:
                        current_progress = progress_store[self.job_id].get("progress", 0)
                        # Only update if progress is moving forward, we hit an error, or deployment is completed
                        if progress > current_progress or status == "error" or status == "completed":
                            progress_store[self.job_id].update({
                                "status": status,
                                "stage": stage,
                                "progress": progress,
                                "message": message[:200],  # Truncate long messages
                                "last_updated": time.time()
                            })
                    else:
                        # Initialize if not exists
                        progress_store[self.job_id] = {
                            "status": status,
                            "stage": stage,
                            "progress": progress,
                            "message": message[:200],
                            "last_updated": time.time()
                        }

app = FastAPI(
    title="TT Inference Server API",
    description="Fast API wrapper for the TT Inference Server run script",
    version="1.1.0"
)

# Test logging on startup
logger.info("FastAPI application initialized")
logger.info("Progress tracking system enabled")
logger.debug("Debug logging test message")

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

def setup_run_logging_to_fastapi():
    """Configure run.py logging to also write to FastAPI logger"""
    # Get the run_log logger that run.py uses
    run_logger = logging.getLogger("run_log")
    
    # Create a custom handler that forwards to FastAPI logger
    class FastAPIHandler(logging.Handler):
        def emit(self, record):
            # Forward the log record to FastAPI logger
            logger.info(f"[RUN.PY] {record.getMessage()}")
    
    # Add the FastAPI handler to run_logger
    fastapi_handler = FastAPIHandler()
    fastapi_handler.setLevel(logging.DEBUG)  # Capture DEBUG messages too
    
    # Check if this handler is already added to avoid duplicates
    handler_exists = any(isinstance(h, type(fastapi_handler)) and 
                        hasattr(h, 'emit') and 
                        h.emit.__func__ == fastapi_handler.emit.__func__ 
                        for h in run_logger.handlers)
    
    if not handler_exists:
        run_logger.addHandler(fastapi_handler)
        logger.info("Added FastAPI logging handler to run_log logger")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "TT Inference Server API is running"}

@app.get("/test-logging")
async def test_logging():
    """Test endpoint to verify logging is working"""
    logger.info("Test logging endpoint called")
    logger.debug("Debug level test message")
    logger.warning("Warning level test message")
    return {
        "message": "Logging test completed", 
        "check": "fastapi.log file for log messages",
        "timestamp": time.time()
    }

@app.get("/run/progress/{job_id}")
async def get_run_progress(job_id: str):
    """Get progress for a running deployment job"""
    with progress_lock:
        progress = progress_store.get(job_id, {
            "status": "not_found",
            "stage": "unknown",
            "progress": 0,
            "message": "Job not found",
            "last_updated": time.time()
        })
        
        # Add stalled detection (>120s no updates)
        if progress["status"] == "running" and "last_updated" in progress:
            time_since_update = time.time() - progress["last_updated"]
            if time_since_update > 120:  # 2 minutes
                progress = progress.copy()  # Don't modify the stored version
                progress["status"] = "stalled"
                progress["message"] = f"No progress updates for {int(time_since_update)}s - deployment may be stalled"
                
    return progress

@app.get("/run/logs/{job_id}")
async def get_run_logs(job_id: str, limit: int = 50):
    """Get recent log messages for a deployment job"""
    with progress_lock:
        logs = log_store.get(job_id, deque())
        # Convert deque to list and get last 'limit' messages
        log_list = list(logs)[-limit:] if logs else []
    
    return {
        "job_id": job_id,
        "logs": log_list,
        "total_messages": len(log_list)
    }

@app.get("/run/stream/{job_id}")
async def stream_run_progress(job_id: str):
    """Stream real-time progress updates via Server-Sent Events"""
    
    def event_generator():
        last_progress = None
        
        # Send initial progress if available
        with progress_lock:
            if job_id in progress_store:
                last_progress = progress_store[job_id].copy()
                yield f"data: {json.dumps(last_progress)}\n\n"
        
        # Poll for updates and stream changes
        while True:
            try:
                with progress_lock:
                    current_progress = progress_store.get(job_id)
                    
                    if current_progress:
                        # Check if progress has changed
                        if not last_progress or current_progress != last_progress:
                            last_progress = current_progress.copy()
                            
                            # Add stalled detection
                            if current_progress["status"] == "running" and "last_updated" in current_progress:
                                time_since_update = time.time() - current_progress["last_updated"]
                                if time_since_update > 120:  # 2 minutes
                                    last_progress["status"] = "stalled"
                                    last_progress["message"] = f"No progress updates for {int(time_since_update)}s - deployment may be stalled"
                            
                            yield f"data: {json.dumps(last_progress)}\n\n"
                            
                            # Stop streaming if deployment is complete or failed
                            if last_progress["status"] in ["completed", "error", "failed", "cancelled"]:
                                break
                    else:
                        # Job not found
                        yield f"data: {json.dumps({'status': 'not_found', 'message': 'Job not found'})}\n\n"
                        break
                
                # Wait before next poll
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in SSE stream: {str(e)}")
                yield f"data: {json.dumps({'status': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
                break
    
    # Only enable SSE if TT_PROGRESS_SSE is set
    if os.getenv("TT_PROGRESS_SSE") != "1":
        raise HTTPException(status_code=404, detail="SSE endpoint not enabled")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/run")
async def run_inference(request: RunRequest):
    try:
        # Generate a unique job ID for this deployment
        job_id = str(uuid.uuid4())[:8]
        
        # Initialize progress tracking
        with progress_lock:
            progress_store[job_id] = {
                "status": "starting",
                "stage": "initialization",
                "progress": 0,
                "message": "Starting deployment...",
                "last_updated": time.time()
            }
            log_store[job_id] = deque(maxlen=MAX_LOG_MESSAGES)
        
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
            # Setup run.py logging to also write to FastAPI logger
            setup_run_logging_to_fastapi()
            
            # Create and attach progress handler to capture run.py logs
            progress_handler = ProgressHandler(job_id)
            run_logger = logging.getLogger("run_log")
            run_logger.addHandler(progress_handler)
            
            # Run the main function
            logger.info("Starting run_main()...")
            return_code, container_info = run_main()
            logger.info(f"run_main() completed with return code: {return_code}")
            logger.info(f"container_info:= {container_info}")
            
            # Remove the progress handler
            run_logger.removeHandler(progress_handler)

            if return_code == 0:
                # Update final progress status
                with progress_lock:
                    if job_id in progress_store:
                        progress_store[job_id].update({
                            "status": "completed",
                            "stage": "complete",
                            "progress": 100,
                            "message": "Deployment completed successfully",
                            "last_updated": time.time()
                        })
                
                # Store container info in the registry
                container_name = container_info["container_name"]
                logger.info(f"container_name:= {container_name}")
                
                # For docker server workflow, try to get container information from logs
                response_data = {
                    "job_id": job_id,
                    "status": "completed",
                    "progress_url": f"/run/progress/{job_id}",
                    "logs_url": f"/run/logs/{job_id}",
                    "container_name": container_name,
                    "message": "Deployment completed successfully"
                }

                # Change container network to tt_studio_network
                try:
                    client = docker.from_env()
                    
                    # Set retry parameters
                    max_retries = 10
                    retry_interval = 3  # seconds
                    attempt = 0
                    
                    # Extract relevant container information from run.py result
                    target_container_name = container_info.get("container_name")
                    target_container_id = container_info.get("container_id")
                    service_port = container_info.get("service_port")
                    logger.info(f"Searching for container with name: {target_container_name}, ID: {target_container_id}, port: {service_port}")
                    
                    # Find the specific container created by run.py
                    new_container = None
                    while attempt < max_retries and not new_container:
                        # List all running containers
                        all_containers = client.containers.list()
                        logger.info(f"all_containers (attempt {attempt+1}/{max_retries}):= {all_containers}")
                        
                        # Search priority:
                        # 1. By exact container ID (most reliable)
                        # 2. By exact container name
                        # 3. By port mapping (containers exposing the configured service port)
                        
                        # 1. Look by container ID (most reliable)
                        if target_container_id:
                            logger.info(f"Looking for container with ID: {target_container_id}")
                            for container in all_containers:
                                if container.id.startswith(target_container_id):
                                    new_container = container
                                    logger.info(f"Found container by ID: {container.id}")
                                    break
                        
                        # 2. Look by exact container name
                        if not new_container and target_container_name:
                            logger.info(f"Looking for container with name: {target_container_name}")
                            for container in all_containers:
                                if container.name == target_container_name:
                                    new_container = container
                                    logger.info(f"Found container by name: {container.name}")
                                    break
                        
                        # 3. Look by port mapping (if service_port is provided)
                        if not new_container and service_port:
                            logger.info(f"Looking for containers exposing port: {service_port}")
                            for container in all_containers:
                                container_ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
                                for port_config in container_ports.values():
                                    if port_config and port_config[0].get('HostPort') == service_port:
                                        new_container = container
                                        logger.info(f"Found container by port mapping: {container.name} (exposing port {service_port})")
                                        break
                                if new_container:
                                    break
                        
                        # If still not found, wait and retry
                        if not new_container:
                            attempt += 1
                            if attempt < max_retries:
                                logger.info(f"Container not found, retrying in {retry_interval} seconds (attempt {attempt}/{max_retries})...")
                                time.sleep(retry_interval)
                            else:
                                logger.error(f"Container not found after {max_retries} attempts")
                    
                    if new_container:
                        original_name = new_container.name
                        logger.info(f"Found container: {original_name}")
                        
                        # Connect to network
                        network = client.networks.get("tt_studio_network")
                        network.connect(new_container)
                        logger.info(f"Connected container {original_name} to tt_studio_network")
                        
                        # Rename the container to container_info["container_name"] if needed
                        if original_name != target_container_name and target_container_name:
                            new_container.rename(target_container_name)
                            logger.info(f"Renamed container from {original_name} to {target_container_name}")
                    else:
                        logger.error("Failed to find the container created by run.py after multiple attempts")
                        
                except Exception as e:
                    logger.error(f"Failed to connect container to network: {str(e)}")
                    # Continue execution even if network connection fails
                
                return Response(
                    content=json.dumps(response_data),
                    media_type="application/json",
                    status_code=status.HTTP_202_ACCEPTED,
                    headers={"Location": f"/run/progress/{job_id}"}
                )
            else:
                # Update progress for failure
                with progress_lock:
                    if job_id in progress_store:
                        progress_store[job_id].update({
                            "status": "failed",
                            "stage": "error",
                            "progress": 0,
                            "message": f"Deployment failed with return code: {return_code}",
                            "last_updated": time.time()
                        })
                raise HTTPException(status_code=500, detail=f"Inference failed with return code: {return_code}")
        finally:
            # Always restore the original working directory
            if original_cwd != script_dir:
                logger.info(f"Restoring working directory to {original_cwd}")
                os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}", exc_info=True)
        
        # Update progress for exception
        if 'job_id' in locals():
            with progress_lock:
                if job_id in progress_store:
                    progress_store[job_id].update({
                        "status": "error",
                        "stage": "error",
                        "progress": 0,
                        "message": f"Deployment error: {str(e)[:200]}",
                        "last_updated": time.time()
                    })
        
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