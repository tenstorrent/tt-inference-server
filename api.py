from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse, JSONResponse
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
from datetime import datetime
from run import main as run_main, parse_arguments, WorkflowType, DeviceTypes
from workflows.model_spec import MODEL_SPECS

# Set up logging
# DO NOT use basicConfig() - it interferes with file handlers
# Instead, configure logging manually
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set level on the logger itself

# Configure FastAPI logger to also write to file
def setup_fastapi_file_logging():
    """Set up file logging for FastAPI - writes to fastapi.log at repo root"""
    try:
        # Put the log file at the repo root:
        # <repo_root>/fastapi.log, assuming this file lives in <repo_root>/tt-inference-server/
        root_log_dir = Path(__file__).parent.parent.resolve()
        root_log_dir.mkdir(parents=True, exist_ok=True)
        root_log_file = root_log_dir / "fastapi.log"

        root_handler = logging.FileHandler(root_log_file, mode="a", encoding="utf-8")
        root_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
        )
        root_handler.setFormatter(formatter)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove any existing handlers to avoid duplicate logs (especially uvicorn's default ones)
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        root_logger.addHandler(root_handler)

        # Let sub-loggers propagate into the root
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

        for name, level in [
            ("fastapi", logging.INFO),
            ("uvicorn", logging.INFO),
            ("uvicorn.access", logging.INFO),
            ("uvicorn.error", logging.INFO),
        ]:
            l = logging.getLogger(name)
            l.setLevel(level)
            l.propagate = True

        logger.info(f"FastAPI file logging configured - writing to {root_log_file}")
        logger.debug(f"Root log absolute path: {root_log_file.absolute()}")

    except Exception as e:
        error_msg = f"Failed to setup FastAPI file logging: {e}"
        print(error_msg, file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)

        # Try to write error to a local fallback log
        try:
            fallback_log = Path(__file__).parent / "fastapi_setup_error.log"
            with open(fallback_log, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
                f.write(traceback.format_exc())
        except Exception:
            pass  # If even fallback fails, just continue

# Initialize file logging
setup_fastapi_file_logging()

# Global progress store with thread-safe access
progress_store: Dict[str, Dict[str, Any]] = {}
log_store: Dict[str, deque] = {}
progress_lock = threading.Lock()

# Store per-deployment log handlers for cleanup
deployment_log_handlers: Dict[str, logging.FileHandler] = {}

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
    version="1.3.0"
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
    # Internal flag to track if this is already a retry (to prevent infinite loops)
    is_retry: Optional[bool] = False
    skip_system_sw_validation: Optional[bool] = False


def normalize_device_alias(device: str) -> str:
    if not device:
        return device
    alias_map = {
        "p300cx2": "p150x4",
    }
    return alias_map.get(device.strip().lower(), device)

def get_fastapi_logs_dir():
    """Get the FastAPI logs directory at repo root"""
    root_log_dir = Path(__file__).parent.parent.resolve()
    fastapi_logs_dir = root_log_dir / "fastapi_logs"
    fastapi_logs_dir.mkdir(parents=True, exist_ok=True)
    return fastapi_logs_dir

def create_deployment_log_handler(job_id: str, model: str, device: str):
    """Create a per-deployment log file handler with model and device in filename"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fastapi_logs_dir = get_fastapi_logs_dir()
    
    # Create log file with pattern: fastapi_YYYY-MM-DD_HH-MM-SS_ModelName_device_server.log
    log_filename = f"fastapi_{timestamp}_{model}_{device}_server.log"
    log_file_path = fastapi_logs_dir / log_filename
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # Use workflow log format
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    # Store handler reference for cleanup
    with progress_lock:
        deployment_log_handlers[job_id] = file_handler
    
    logger.info(f"Created per-deployment log file: {log_file_path}")
    return file_handler, log_file_path

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

def sync_tokens_from_tt_studio():
    """
    Cross-check and sync JWT_SECRET and HF_TOKEN from TT Studio's .env 
    to inference server's .env file if they differ.
    """
    from workflows.utils import load_dotenv
    
    # Paths to .env files
    tt_studio_root = os.getenv("TT_STUDIO_ROOT")
    if not tt_studio_root:
        logger.warning("TT_STUDIO_ROOT environment variable not set, cannot sync tokens")
        return
    
    tt_studio_env = Path(tt_studio_root) / "app" / ".env"
    inference_server_env = Path(__file__).parent / ".env"
    
    # Read TT Studio .env values
    tt_studio_jwt = None
    tt_studio_hf = None
    
    if tt_studio_env.exists():
        with open(tt_studio_env, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'JWT_SECRET':
                            tt_studio_jwt = value
                        elif key == 'HF_TOKEN':
                            tt_studio_hf = value
    else:
        logger.warning(f"TT Studio .env file not found at {tt_studio_env}")
        return
    
    # Read inference server .env values
    inference_jwt = None
    inference_hf = None
    env_lines = []
    
    if inference_server_env.exists():
        with open(inference_server_env, 'r') as f:
            env_lines = f.readlines()
            for line in env_lines:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#'):
                    if '=' in line_stripped:
                        key, value = line_stripped.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'JWT_SECRET':
                            inference_jwt = value
                        elif key == 'HF_TOKEN':
                            inference_hf = value
    
    # Check for differences and update if needed
    updated = False
    
    # Update or add JWT_SECRET
    if tt_studio_jwt and tt_studio_jwt != inference_jwt:
        logger.info("JWT_SECRET differs between TT Studio and inference server - updating inference server .env")
        # Remove old JWT_SECRET line if exists
        env_lines = [line for line in env_lines 
                    if not line.strip().startswith('JWT_SECRET=')]
        # Add new JWT_SECRET
        env_lines.append(f"JWT_SECRET={tt_studio_jwt}\n")
        updated = True
    
    # Update or add HF_TOKEN
    if tt_studio_hf and tt_studio_hf != inference_hf:
        logger.info("HF_TOKEN differs between TT Studio and inference server - updating inference server .env")
        # Remove old HF_TOKEN line if exists
        env_lines = [line for line in env_lines 
                    if not line.strip().startswith('HF_TOKEN=')]
        # Add new HF_TOKEN
        env_lines.append(f"HF_TOKEN={tt_studio_hf}\n")
        updated = True
    
    # Write back if updated
    if updated:
        with open(inference_server_env, 'w') as f:
            f.writelines(env_lines)
        logger.info(f"Updated inference server .env file at {inference_server_env}")
        # Reload environment variables
        load_dotenv()
    else:
        logger.info("JWT_SECRET and HF_TOKEN are already synchronized")

@app.post("/run")
async def run_inference(request: RunRequest):
    deployment_log_handler = None
    deployment_log_path = None
    try:
        original_device = request.device
        normalized_device = normalize_device_alias(request.device)
        if normalized_device != original_device:
            logger.info(
                "Normalizing device alias from %s to %s",
                original_device,
                normalized_device,
            )
        # Generate a unique job ID for this deployment
        job_id = str(uuid.uuid4())[:8]
        
        # Create per-deployment log file
        deployment_log_handler, deployment_log_path = create_deployment_log_handler(
            job_id, request.model, normalized_device
        )
        
        # Attach deployment log handler to relevant loggers
        logger.addHandler(deployment_log_handler)
        run_logger = logging.getLogger("run_log")
        run_logger.addHandler(deployment_log_handler)
        
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
        
        # Sync tokens from TT Studio before setting environment variables
        try:
            sync_tokens_from_tt_studio()
        except Exception as e:
            logger.warning(f"Failed to sync tokens from TT Studio: {e}")
            # Continue anyway - tokens might be set via request or environment
        
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
        # Note: Since the FastAPI server now runs as the actual user (not root),
        # Path.home() in get_default_hf_home_path() will correctly return the user's home directory
        # (e.g., /home/username/.cache/huggingface instead of /root/.cache/huggingface)
        env_vars_to_set = {
            "AUTOMATIC_HOST_SETUP": "True",
            "TT_PROGRESS_DEBUG": "1",  # Enable structured progress emission
            "TT_PROGRESS_SSE": "1",     # Enable SSE endpoint for real-time progress
            "SERVICE_PORT": "7000"      # Set SERVICE_PORT to match --service-port argument
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
        sys.argv.extend(["--device", normalized_device])
        sys.argv.extend(["--docker-server"])
        # Add dev-mode if requested (used for auto-retry on failure)
        if request.dev_mode:
            sys.argv.extend(["--dev-mode"])
        # Skip system software validation if requested (handles prerelease versions like '2.6.0-rc1')
        if request.skip_system_sw_validation:
            sys.argv.extend(["--skip-system-sw-validation"])
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
            
            # Extract and log docker workflow log file path if available
            if container_info and isinstance(container_info, dict):
                docker_log_file_path = container_info.get("docker_log_file_path")
                if docker_log_file_path:
                    logger.info(f"Docker workflow log file: {docker_log_file_path}")
            
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
                container_id = container_info.get("container_id")
                docker_log_file_path = container_info.get("docker_log_file_path")
                logger.info(f"container_name:= {container_name}")
                logger.info(f"container_id:= {container_id}")
                if docker_log_file_path:
                    logger.info(f"docker_log_file_path:= {docker_log_file_path}")
                
                # For docker server workflow, try to get container information from logs
                response_data = {
                    "job_id": job_id,
                    "status": "completed",
                    "progress_url": f"/run/progress/{job_id}",
                    "logs_url": f"/run/logs/{job_id}",
                    "container_name": container_name,
                    "container_id": container_id,  # Add container_id to response
                    "docker_log_file_path": docker_log_file_path,  # Add workflow log file path
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
                        
                        # Update response_data with actual container ID if we found it
                        if new_container.id:
                            response_data["container_id"] = new_container.id
                            logger.info(f"Updated response_data with container_id: {new_container.id}")
                        
                        # Connect to network
                        network = client.networks.get("tt_studio_network")
                        network.connect(new_container)
                        logger.info(f"Connected container {original_name} to tt_studio_network")
                        
                        # Rename the container to the model name for easier identification
                        model_name = request.model.replace('/', '-')  # Sanitize model name for container naming
                        if original_name != model_name:
                            new_container.rename(model_name)
                            logger.info(f"Renamed container from {original_name} to {model_name}")
                            # Update response_data with new name
                            response_data["container_name"] = model_name
                    else:
                        logger.error("Failed to find the container created by run.py after multiple attempts")
                        
                except Exception as e:
                    logger.error(f"Failed to connect container to network: {str(e)}")
                    # Continue execution even if network connection fails
                
                # Log the final response_data before sending
                logger.info(f"Final response_data before sending: {response_data}")
                logger.info(f"response_data contains docker_log_file_path: {'docker_log_file_path' in response_data}")
                if 'docker_log_file_path' in response_data:
                    logger.info(f"response_data['docker_log_file_path'] = {response_data.get('docker_log_file_path')}")
                
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
                
                # Auto-retry with dev_mode if this is the first attempt
                if not request.is_retry and not request.dev_mode:
                    logger.info(f"Deployment failed with return code {return_code}, auto-retrying with dev_mode=True")
                    
                    # Update progress to show retry
                    with progress_lock:
                        if job_id in progress_store:
                            progress_store[job_id].update({
                                "status": "retrying",
                                "stage": "retry",
                                "progress": 0,
                                "message": "Retrying deployment with dev_mode enabled...",
                                "last_updated": time.time()
                            })
                    
                    # Create a new request with dev_mode=True, skip_system_sw_validation=True, and is_retry=True
                    retry_request = request.copy(update={"dev_mode": True, "skip_system_sw_validation": True, "is_retry": True})

                    # Recursively call run_inference with the retry request
                    return await run_inference(retry_request)
                
                # Return JSONResponse instead of raising HTTPException to include job_id
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "job_id": job_id,
                        "message": f"Deployment failed with return code: {return_code}",
                        "progress_url": f"/run/progress/{job_id}",
                        "logs_url": f"/run/logs/{job_id}"
                    }
                )
        finally:
            # Clean up per-deployment log handler
            if deployment_log_handler:
                try:
                    logger.removeHandler(deployment_log_handler)
                    run_logger = logging.getLogger("run_log")
                    run_logger.removeHandler(deployment_log_handler)
                    deployment_log_handler.close()
                    if 'job_id' in locals():
                        with progress_lock:
                            if job_id in deployment_log_handlers:
                                del deployment_log_handlers[job_id]
                        logger.info(f"Cleaned up per-deployment log handler for job {job_id}")
                    else:
                        logger.info("Cleaned up per-deployment log handler (job_id not available)")
                except Exception as e:
                    logger.error(f"Error cleaning up deployment log handler: {e}")
            
            # Always restore the original working directory
            if original_cwd != script_dir:
                logger.info(f"Restoring working directory to {original_cwd}")
                os.chdir(original_cwd)
            
    except Exception as e:
        logger.error(f"Error in run_inference: {str(e)}", exc_info=True)
        
        # Clean up per-deployment log handler if it was created
        if 'deployment_log_handler' in locals() and deployment_log_handler:
            try:
                logger.removeHandler(deployment_log_handler)
                run_logger = logging.getLogger("run_log")
                run_logger.removeHandler(deployment_log_handler)
                deployment_log_handler.close()
                if 'job_id' in locals():
                    with progress_lock:
                        if job_id in deployment_log_handlers:
                            del deployment_log_handlers[job_id]
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up deployment log handler in exception handler: {cleanup_error}")
        
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
        
        # Auto-retry with dev_mode if this is the first attempt
        if not request.is_retry and not request.dev_mode:
            logger.info(f"Deployment failed with exception, auto-retrying with dev_mode=True")
            
            # Update progress to show retry
            if 'job_id' in locals():
                with progress_lock:
                    if job_id in progress_store:
                        progress_store[job_id].update({
                            "status": "retrying",
                            "stage": "retry",
                            "progress": 0,
                            "message": "Retrying deployment with dev_mode enabled...",
                            "last_updated": time.time()
                        })
            
            # Create a new request with dev_mode=True, skip_system_sw_validation=True, and is_retry=True
            retry_request = request.copy(update={"dev_mode": True, "skip_system_sw_validation": True, "is_retry": True})

            # Recursively call run_inference with the retry request
            return await run_inference(retry_request)
        
        # Return JSONResponse instead of raising HTTPException to include job_id
        if 'job_id' in locals():
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "job_id": job_id,
                    "message": f"Deployment error: {str(e)}",
                    "progress_url": f"/run/progress/{job_id}",
                    "logs_url": f"/run/logs/{job_id}"
                }
            )
        else:
            # If job_id wasn't created yet, raise HTTPException
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get list of available models"""
    return {"models": list(set(spec.model_name for _, spec in MODEL_SPECS.items()))}

@app.get("/workflows")
async def get_available_workflows():
    """Get list of available workflows"""
    return {"workflows": [w.name.lower() for w in WorkflowType]}

@app.get("/devices")
async def get_available_devices():
    """Get list of available devices"""
    return {"devices": [d.name.lower() for d in DeviceTypes]} 