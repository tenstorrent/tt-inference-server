# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess

from workflows.utils import (
    get_logger,
    get_repo_root_path,
)
from workflows.model_config import MODEL_CONFIGS

logger = get_logger()


def run_docker(args, setup_config):
    model_name = args.model
    repo_root_path = get_repo_root_path()
    model_config = MODEL_CONFIGS[model_name]
    model_volume = setup_config.model_volume_root
    cache_root = setup_config.cache_root
    env_file = setup_config.env_file
    service_port = args.service_port
    # fmt: off
    # TODO: replace --volume with --mount commands
    docker_command = [
        "docker",
        "run",
        "--rm",
        "-it",
        "-e", f"SERVICE_PORT={service_port}",
        # "-e", f"JWT_SECRET={args.jwt_secret}",
        # "-e", f"HF_TOKEN={args.hf_token}",
        "--env-file", str(env_file),
        "--cap-add", "ALL",
        "--device", "/dev/tenstorrent:/dev/tenstorrent",
        "--volume", "/dev/hugepages-1G:/dev/hugepages-1G:rw",
        "--volume", f"{model_volume}:{cache_root}:rw",
        "--shm-size", "32G",
        "--publish", f"{service_port}:{service_port}",  # map host port 8000 to container port 8000
    ]
    # fmt: on
    dev = True
    if dev:
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
    docker_command += [model_config.docker_image]
    cmd = " ".join(docker_command)
    logger.info(f"Executing command: {cmd}")

    try:
        # Execute the command, streaming output to the terminal.
        subprocess.run(docker_command, check=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the docker container:")
        print(e)
