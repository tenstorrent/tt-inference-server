# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
import shlex
import atexit
import time
import logging
from datetime import datetime

from workflows.utils import (
    get_repo_root_path,
)
from workflows.model_config import MODEL_CONFIGS
from workflows.utils import get_default_workflow_root_log_dir, ensure_readwriteable_dir

logger = logging.getLogger("run_log")


def run_docker_server(args, setup_config):
    model_name = args.model
    repo_root_path = get_repo_root_path()
    model_config = MODEL_CONFIGS[model_name]
    model_volume = setup_config.model_volume_root
    cache_root = setup_config.cache_root
    env_file = setup_config.env_file
    service_port = args.service_port
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    docker_log_file_dir = get_default_workflow_root_log_dir() / "docker_server"
    ensure_readwriteable_dir(docker_log_file_dir)
    docker_log_file_path = (
        docker_log_file_dir / f"{args.model}_{args.workflow}_{timestamp}.log"
    )
    docker_image = model_config.docker_image
    # fmt: off
    # TODO: replace --volume with --mount commands
    docker_command = [
        "docker",
        "run",
        "--rm",
        "-e", f"SERVICE_PORT={service_port}",
        "--env-file", str(env_file),
        "--cap-add", "ALL",
        "--device", "/dev/tenstorrent:/dev/tenstorrent",
        "--volume", "/dev/hugepages-1G:/dev/hugepages-1G:rw",
        "--volume", f"{model_volume}:{cache_root}:rw",
        "--shm-size", "32G",
        "--publish", f"{service_port}:{service_port}",  # map host port 8000 to container port 8000
    ]
    # fmt: on
    if args.dev_mode:
        # use dev image
        docker_image = docker_image.replace("-release-", "-dev-")
        # development mounts
        # Define the environment file path for the container.
        user_home_path = "/home/container_app_user"
        # fmt: off
        docker_command += [
            "--volume", f"{repo_root_path}/vllm-tt-metal-llama3/src:{user_home_path}/app/src",
            "--volume", f"{repo_root_path}/benchmarking:{user_home_path}/app/benchmarking",
            "--volume", f"{repo_root_path}/evals:{user_home_path}/app/evals",
            "--volume", f"{repo_root_path}/locust:{user_home_path}/app/locust",
            "--volume", f"{repo_root_path}/utils:{user_home_path}/app/utils",
            "--volume", f"{repo_root_path}/tests:{user_home_path}/app/tests",
        ]
        # fmt: on

    # add docker image at end
    docker_command.append(docker_image)
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
    # wait for container to start
    time.sleep(5)

    container_id = subprocess.check_output(
        ["docker", "ps", "-l", "--format", "{{.ID}}"], text=True
    ).strip()

    def teardown_docker():
        logger.info("Stopping inference server Docker container ...")
        subprocess.run(["docker", "stop", container_id])
        docker_log_file.close()
        logger.info("run_docker cleanup finished.")

    atexit.register(teardown_docker)

    return
