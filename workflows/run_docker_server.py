# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
import shlex
import atexit
import time
import logging
import uuid
from datetime import datetime
import json
import requests

from workflows.utils import (
    get_repo_root_path,
)
from workflows.model_spec import MODEL_SPECS
from workflows.utils import (
    get_default_workflow_root_log_dir,
    ensure_readwriteable_dir,
    run_command,
    default_dotenv_path,
)
from workflows.log_setup import clean_log_file
from workflows.workflow_types import WorkflowType, DeviceTypes, ServerTypes

logger = logging.getLogger("run_log")


def short_uuid():
    return str(uuid.uuid4())[:8]


def wait_for_tt_server_ready(port: str, max_attempts: int = 120, sleep_time: int = 10):
    """Wait for TT_SERVER to be ready by checking the liveness endpoint."""
    logger.info("Waiting for inference server to be ready...")
    logger.info(f"Max attempts: {max_attempts}, sleep time: {sleep_time}")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/tt-liveness", timeout=5)
            http_status = response.status_code
        except requests.exceptions.RequestException:
            http_status = 0
        
        logger.info(f"HTTP status: {http_status}")
        logger.info(f"Attempt: {attempt + 1}/{max_attempts}")
        
        if http_status == 200:
            logger.info(f"Inference server is ready! (HTTP {http_status})")
            return True
            
        time.sleep(sleep_time)
    
    logger.error(f"Server failed to become ready after {max_attempts} attempts")
    return False


def ensure_docker_image(image_name):
    # First attempt to pull the image to get latest version
    logger.info(f"Pulling docker image: {image_name}")
    logger.info("This may take several minutes ...")
    cmd = ["docker", "pull", image_name]
    pull_return_code = run_command(cmd, logger=logger)
    
    if pull_return_code == 0:
        logger.info("✅ Docker image pulled successfully.")
        # Verify the pulled image is available
        verify_return_code = run_command(
            [
                "docker",
                "inspect",
                "--format='ID: {{.Id}}, Created: {{.Created}}'",
                image_name,
            ],
            logger=logger,
        )
        if verify_return_code == 0:
            logger.info("✅ Docker image available. See SHA and built timestamp above.")
            return True
        else:
            logger.error("⛔ Docker image pull succeeded but verification failed.")
            return False
    
    # Pull failed, check if image exists locally as fallback
    logger.error(f"⛔ Docker image pull failed with return code: {pull_return_code}")
    logger.info("Checking for local image as fallback...")
    
    local_check_return_code = run_command(
        [
            "docker",
            "inspect",
            "--format='ID: {{.Id}}, Created: {{.Created}}'",
            image_name,
        ],
        logger=logger,
    )
    
    if local_check_return_code == 0:
        logger.info("✅ Using local docker image. See SHA and built timestamp above.")
        return True
    
    # Both pull and local check failed
    err_str = f"⛔ Docker image pull failed and local image not found. image_name={image_name}"
    if "-release-" in image_name:
        err_str += " NOTE: You are running in release mode, use '--dev-mode' CLI arg to run the dev image."
    logger.error(err_str)
    return False


def run_docker_server(model_spec, setup_config, json_fpath):
    args = model_spec.cli_args
    repo_root_path = get_repo_root_path()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Choose log file name based on server type
    if model_spec.server_type == ServerTypes.VLLM:
        service_name = "vllm"
    elif model_spec.server_type == ServerTypes.TT_SERVER:
        service_name = "tt_server"
    else:
        raise ValueError(f"Invalid server type: {model_spec.server_type}")
    
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    docker_log_file_path = (
        docker_log_file_dir
        / f"{service_name}_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )
    device = DeviceTypes.from_string(args.device)
    mesh_device_str = device.to_mesh_device_str()
    container_name = f"tt-inference-server-{short_uuid()}"

    # TODO: remove this once https://github.com/tenstorrent/tt-metal/issues/23785 has been closed
    device_cache_dir = (
        DeviceTypes.to_mesh_device_str(model_spec.subdevice_type)
        if model_spec.subdevice_type
        else mesh_device_str
    )

    # create device mapping string to pass to docker run
    device_path = "/dev/tenstorrent"
    if not getattr(args, "device_id", None):
        device_map_strs = ["--device", f"{device_path}:{device_path}"]
    else:
        device_map_strs = []
        for d in args.device_id:
            device_map_strs.extend(["--device", f"{device_path}/{d}:{device_path}/{d}"])

    # ensure docker image is available
    assert ensure_docker_image(
        model_spec.docker_image
    ), f"Docker image: {model_spec.docker_image} not found on GHCR or locally."

    # Update environment variables based on server type
    if model_spec.server_type == ServerTypes.TT_SERVER:
        # Environment variables for tt-server (tt-metal-sdxl container)
        docker_env_vars = {
            "CACHE_ROOT": setup_config.cache_root,
            "TT_CACHE_PATH": setup_config.container_tt_metal_cache_dir / device_cache_dir,
            "MODEL_WEIGHTS_PATH": setup_config.container_model_weights_path,
            "MODEL": args.model,
            "DEVICE": args.device,
            "MODEL_RUNNER": model_spec.device_model_spec.env_vars.get("MODEL_RUNNER", "tt-sdxl"),
            "MODEL_SERVICE": model_spec.device_model_spec.env_vars.get("MODEL_SERVICE", "image"),
            "LOG_LEVEL": "INFO",
            "DEVICE_IDS": ",".join(str(i) for i in args.device_id) if getattr(args, "device_id", None) else "0",
            "MAX_QUEUE_SIZE": "64",
            "MAX_BATCH_SIZE": "32",
        }
    elif model_spec.server_type == ServerTypes.VLLM:
        # Environment variables for vLLM container
        # CACHE_ROOT needed for the docker container entrypoint
        # TT_CACHE_PATH has host path
        # MODEL_WEIGHTS_PATH has dynamic path
        # TT_LLAMA_TEXT_VER must be set BEFORE import time of run_vllm_api_server.py for vLLM registry
        # TT_MODEL_SPEC_JSON_PATH has dynamic path
        docker_json_fpath = setup_config.container_model_spec_dir / json_fpath.name
        docker_env_vars = {
            "CACHE_ROOT": setup_config.cache_root,
            "TT_CACHE_PATH": setup_config.container_tt_metal_cache_dir / device_cache_dir,
            "MODEL_WEIGHTS_PATH": setup_config.container_model_weights_path,
            "TT_LLAMA_TEXT_VER": model_spec.impl.impl_id,
            "TT_MODEL_SPEC_JSON_PATH": docker_json_fpath,
        }

    # fmt: off
    # note: --env-file is just used for secrets, avoids persistent state on host
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        "--env-file", str(default_dotenv_path),
    ]
    
    # Add server-type specific Docker parameters
    if model_spec.server_type == ServerTypes.TT_SERVER:
        # TT_SERVER specific parameters
        docker_command.extend([
            # "-d",  # Run in detached mode for TT_SERVER
            "--cap-add", "sys_nice",
            "--security-opt", "seccomp=unconfined", 
            "--user", "root",
        ])
    else:
        # VLLM and other server types use existing logic
        docker_command.extend([
            "--cap-add", "ALL",
        ])
    
    # Add common parameters
    docker_command.extend([
        *device_map_strs,
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        # note: order of mounts matters, model_volume_root must be mounted before nested mounts
        "--mount", f"type=bind,src={setup_config.host_model_volume_root},dst={setup_config.cache_root}",
        "--mount", f"type=bind,src={setup_config.host_model_weights_mount_dir},dst={setup_config.container_model_weights_mount_dir},readonly",
        "--shm-size", "32G",
        "--publish", f"{model_spec.cli_args.service_port}:{model_spec.cli_args.service_port}",
    ])
    # fmt: on

    # Add model spec JSON mount only for vLLM server
    # TODO: add to tt-server
    if model_spec.server_type == ServerTypes.VLLM:
        docker_json_fpath = setup_config.container_model_spec_dir / json_fpath.name
        docker_command.extend([
            "--mount", f"type=bind,src={json_fpath},dst={docker_json_fpath},readonly",
        ])
    
    if args.interactive:
        docker_command.append("-it")

    for key, value in docker_env_vars.items():
        if value:
            docker_command.extend(["-e", f"{key}={str(value)}"])
        else:
            logger.info(f"Skipping {key} in docker run command, value={value}")

    if args.dev_mode:
        # development mounts
        user_home_path = "/home/container_app_user"
        # fmt: off
        if model_spec.server_type == ServerTypes.VLLM:
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/vllm-tt-metal-llama3/src,dst={user_home_path}/app/src",
                "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
                "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
                "--mount", f"type=bind,src={repo_root_path}/locust,dst={user_home_path}/app/locust",
                "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
                "--mount", f"type=bind,src={repo_root_path}/tests,dst={user_home_path}/app/tests",
            ]
        elif model_spec.server_type == ServerTypes.TT_SERVER:
            docker_command += [
                "--mount", f"type=bind,src={repo_root_path}/tt-metal-sdxl,dst={user_home_path}/tt-metal/server",
                "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
                "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
                "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            ]
        # fmt: on

    # add docker image at end
    docker_command.append(model_spec.docker_image)
    if args.interactive:
        docker_command.extend(["bash", "-c", "sleep infinity"])
    logger.info(f"Docker run command:\n{shlex.join(docker_command)}\n")

    # Open docker log file for both server types
    docker_log_file = open(docker_log_file_path, "w", buffering=1)
    logger.info(f"Running docker container with log file: {docker_log_file_path}")
    
    # # Initialize variables used in cleanup
    # container_id = ""
    # docker_logs_process = None

    # # Handle different execution modes based on server type
    # if model_spec.server_type == ServerTypes.TT_SERVER:
    #     # Run TT_SERVER in detached mode
    #     logger.info(f"Running TT_SERVER container in detached mode")
    #     result = subprocess.run(docker_command, capture_output=True, text=True)
    #     if result.returncode != 0:
    #         logger.error(f"Failed to start TT_SERVER container: {result.stderr}")
    #         docker_log_file.close()
    #         raise RuntimeError("TT_SERVER container failed to start")
        
    #     # Get container ID
    #     container_id = subprocess.check_output(
    #         ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.ID}}"],
    #         text=True,
    #     ).strip()
        
    #     if not container_id:
    #         logger.error(f"TT_SERVER container {container_name} not found after startup")
    #         docker_log_file.close()
    #         raise RuntimeError("TT_SERVER container not found after startup")
        
    #     logger.info(f"TT_SERVER container started with ID: {container_id}")
        
    #     # Start docker logs process to stream to file
    #     docker_logs_process = subprocess.Popen(
    #         ["docker", "logs", "-f", container_name],
    #         stdout=docker_log_file,
    #         stderr=docker_log_file,
    #         text=True
    #     )
        
    #     # Wait for server to be ready
    #     if not wait_for_tt_server_ready(model_spec.cli_args.service_port):
    #         logger.error("TT_SERVER failed to become ready")
    #         if docker_logs_process:
    #             docker_logs_process.terminate()
    #         docker_log_file.close()
    #         subprocess.run(["docker", "stop", container_name])
    #         raise RuntimeError("TT_SERVER failed to become ready")
        
    # else:
        # Original logic for VLLM and other server types (non-detached mode)
    # note: running without -d (detached mode) because logs from tt-metal cannot
    # be accessed otherwise, e.g. via docker logs <container_id>
    # this has added benefit of providing a docker run command users can run
    # for debugging more easily
    _ = subprocess.Popen(
        docker_command, stdout=docker_log_file, stderr=docker_log_file, text=True
    )

    # poll for container to start
    TIMEOUT = 30  # seconds
    POLL_INTERVAL = 0.5  # seconds
    start_time = time.time()
    container_id = ""

    while (time.time() - start_time) < TIMEOUT:
        container_id = subprocess.check_output(
            ["docker", "ps", "-f", f"name={container_name}", "--format", "{{.ID}}"],
            text=True,
        ).strip()
        if container_id:
            break
        time.sleep(POLL_INTERVAL)

    if not container_id:
        logger.error(
            f"TIMEOUT={TIMEOUT} seconds has passed. (docker pull has already run)"
        )
        logger.error(f"Docker container {container_name} failed to start.")
        logger.error(f"Docker image: {model_spec.docker_image}")
        logger.error("Check logs for more information.")
        logger.error(f"Docker logs are streamed to: {docker_log_file_path}")
        docker_log_file.close()
        raise RuntimeError("Docker container failed to start.")

    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(args.workflow) not in skip_workflows:

        def teardown_docker():
            logger.info("atexit: Stopping inference server Docker container ...")
            if model_spec.server_type == ServerTypes.TT_SERVER and docker_logs_process:
                docker_logs_process.terminate()
            subprocess.run(["docker", "stop", container_name])
            docker_log_file.close()
            # remove asci escape formating from log file
            clean_log_file(docker_log_file_path)
            logger.info("run_docker cleanup finished.")

        atexit.register(teardown_docker)
    else:

        def exit_log_messages():
            # note: closing the file in this process does not stop the container from
            # streaming output to the log file
            docker_log_file.close()
            logger.info(f"Created Docker container ID: {container_id}")
            logger.info(f"Access container logs via: docker logs -f {container_id}")
            logger.info(
                f"Docker logs are also streamed to log file: {docker_log_file_path}"
            )
            logger.info(
                f"To stop the running container run: docker stop {container_id}"
            )

        atexit.register(exit_log_messages)

    return
