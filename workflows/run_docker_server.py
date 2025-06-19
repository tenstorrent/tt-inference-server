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

from workflows.utils import (
    get_repo_root_path,
)
from workflows.model_config import MODEL_CONFIGS
from workflows.utils import (
    get_default_workflow_root_log_dir,
    ensure_readwriteable_dir,
    run_command,
    get_model_id,
    default_dotenv_path,
)
from workflows.log_setup import clean_log_file
from workflows.workflow_types import WorkflowType, DeviceTypes

logger = logging.getLogger("run_log")


def short_uuid():
    return str(uuid.uuid4())[:8]


def ensure_docker_image(image_name):
    logger.info(f"running: docker pull {image_name}")
    logger.info("this may take several minutes ...")
    cmd = ["docker", "pull", image_name]
    pull_return_code = run_command(cmd, logger=logger)
    if pull_return_code != 0:
        logger.error(
            f"⛔ Docker image pull from ghcr.io failed with return code: {pull_return_code}"
        )
        logger.info("Attempting to run image from local images ...")
    else:
        logger.info("✅ Docker Image pulled successfully.")
    return_code = run_command(
        [
            "docker",
            "inspect",
            "--format='ID: {{.Id}}, Created: {{.Created}}'",
            image_name,
        ],
        logger=logger,
    )
    if return_code != 0:
        err_str = "⛔ Docker image does not exist locally."
        if "-release-" in image_name:
            err_str += " You are running in release mode, use '--dev-mode' CLI argto run the dev image."
        logger.error(err_str)
        return False
    logger.info("✅ Docker Image available locally. See SHA and built timestamp above.")
    return True


def run_docker_server(args, setup_config):
    model_id = get_model_id(args.impl, args.model, args.device)
    repo_root_path = get_repo_root_path()
    model_config = MODEL_CONFIGS[model_id]
    service_port = args.service_port
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    docker_log_file_path = (
        docker_log_file_dir
        / f"vllm_{timestamp}_{args.model}_{args.device}_{args.workflow}.log"
    )
    docker_image = model_config.docker_image
    device = DeviceTypes.from_string(args.device)
    mesh_device_str = DeviceTypes.to_mesh_device_str(device)
    container_name = f"tt-inference-server-{short_uuid()}"

    device_cache_dir = (
        DeviceTypes.to_mesh_device_str(model_config.subdevice_type)
        if model_config.subdevice_type
        else mesh_device_str
    )

    device_path = "/dev/tenstorrent"
    if hasattr(args, "device_id") and args.device_id is not None:
        device_path = f"{device_path}/{args.device_id}"

    if args.dev_mode:
        # use dev image
        docker_image = docker_image.replace("-release-", "-dev-")

    if args.override_docker_image:
        docker_image = args.override_docker_image

    # ensure docker image is available
    assert ensure_docker_image(
        docker_image
    ), f"Docker image: {docker_image} not found on GHCR or locally."

    docker_env_vars = {
        "SERVICE_PORT": service_port,
        "MESH_DEVICE": mesh_device_str,
        "MODEL_IMPL": model_config.impl.impl_name,
        "CACHE_ROOT": setup_config.cache_root,
        "TT_CACHE_PATH": setup_config.container_tt_metal_cache_dir / device_cache_dir,
        "MODEL_WEIGHTS_PATH": setup_config.container_model_weights_path,
        "HF_MODEL_REPO_ID": model_config.hf_model_repo,
        "MODEL_SOURCE": setup_config.model_source,
        "VLLM_MAX_NUM_SEQS": model_config.device_model_spec.max_concurrency,
        "VLLM_MAX_MODEL_LEN": model_config.device_model_spec.max_context,
        "VLLM_MAX_NUM_BATCHED_TOKENS": model_config.device_model_spec.max_context,
    }

    # Pass model config override_tt_config if it exists
    if model_config.device_model_spec.override_tt_config:
        json_str = json.dumps(model_config.device_model_spec.override_tt_config)
        docker_env_vars["OVERRIDE_TT_CONFIG"] = json_str
        logger.info(
            f"setting from model config: OVERRIDE_TT_CONFIG={model_config.device_model_spec.override_tt_config}"
        )

    # Pass CLI override_tt_config if provided
    if hasattr(args, "override_tt_config") and args.override_tt_config:
        docker_env_vars["OVERRIDE_TT_CONFIG"] = args.override_tt_config
        logger.info(f"setting from CLI: OVERRIDE_TT_CONFIG={args.override_tt_config}")

    # Pass CLI vllm_override_args if provided
    if hasattr(args, "vllm_override_args") and args.vllm_override_args:
        docker_env_vars["VLLM_OVERRIDE_ARGS"] = args.vllm_override_args
        logger.info(f"setting from CLI: VLLM_OVERRIDE_ARGS={args.vllm_override_args}")

    # fmt: off
    # note: --env-file is just used for secrets, avoids persistent state on host
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        "--env-file", str(default_dotenv_path),
        "--cap-add", "ALL",
        "--device", f"{device_path}:{device_path}",
        "--mount", "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G",
        # note: order of mounts matters, model_volume_root must be mounted before nested mounts
        "--mount", f"type=bind,src={setup_config.host_model_volume_root},dst={setup_config.cache_root}",
        "--mount", f"type=bind,src={setup_config.host_model_weights_mount_dir},dst={setup_config.container_model_weights_mount_dir},readonly",
        "--shm-size", "32G",
        "--publish", f"{service_port}:{service_port}",  # map host port 8000 to container port 8000
    ]
    if args.interactive:
        docker_command.append("--interactive")
    # fmt: on

    # override existing env vars when running on Blackhole
    if DeviceTypes.is_blackhole(device):
        docker_command += [
            "-e",
            "ARCH_NAME=blackhole",
            "-e",
            "WH_ARCH_YAML=",
        ]

    for key, value in docker_env_vars.items():
        if value:
            docker_command.extend(["-e", f"{key}={str(value)}"])
    if args.dev_mode:
        # development mounts
        # Define the environment file path for the container.
        user_home_path = "/home/container_app_user"
        # fmt: off
        docker_command += [
            "--mount", f"type=bind,src={repo_root_path}/vllm-tt-metal-llama3/src,dst={user_home_path}/app/src",
            "--mount", f"type=bind,src={repo_root_path}/benchmarking,dst={user_home_path}/app/benchmarking",
            "--mount", f"type=bind,src={repo_root_path}/evals,dst={user_home_path}/app/evals",
            "--mount", f"type=bind,src={repo_root_path}/locust,dst={user_home_path}/app/locust",
            "--mount", f"type=bind,src={repo_root_path}/utils,dst={user_home_path}/app/utils",
            "--mount", f"type=bind,src={repo_root_path}/tests,dst={user_home_path}/app/tests",
        ]
        # fmt: on

    # add docker image at end
    docker_command.append(docker_image)
    if args.interactive:
        docker_command.extend(["bash", "-c", "sleep infinity"])
    logger.info(f"Docker run command:\n{shlex.join(docker_command)}\n")

    docker_log_file = open(docker_log_file_path, "w", buffering=1)
    logger.info(f"Running docker container with log file: {docker_log_file_path}")
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
        logger.error(f"Docker image: {docker_image}")
        logger.error("Check logs for more information.")
        logger.error(f"Docker logs are streamed to: {docker_log_file_path}")
        raise RuntimeError("Docker container failed to start.")

    skip_workflows = {WorkflowType.SERVER, WorkflowType.REPORTS}
    if WorkflowType.from_string(args.workflow) not in skip_workflows:

        def teardown_docker():
            logger.info("atexit: Stopping inference server Docker container ...")
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
