# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


import os
import subprocess

from workflows.utils import (
    get_logger,
    get_repo_root_path,
)
from workflows.model_config import MODEL_CONFIGS

logger = get_logger()


def run_docker(args):
    model_name = args.model
    repo_root_path = get_repo_root_path()
    model_config = MODEL_CONFIGS[model_name]

    # Get MODEL_VOLUME from the environment; if not set, default to a constructed path.
    MODEL_VOLUME = os.environ.get("MODEL_VOLUME")
    if not MODEL_VOLUME:
        MODEL_VOLUME = (
            repo_root_path
            / f"persistent_volume/volume_id_tt-metal-{model_name}-v0.0.1/"
        )

    # Define the environment file path for the container.
    env_file = repo_root_path / "persistent_volume" / "model_envs" / f"{model_name}.env"
    user_home_path = "/home/container_app_user"
    # development mounts
    docker_command = [
        "docker",
        "run",
        "--rm",
        "-it",
        "--env-file",
        str(env_file),
        "--cap-add",
        "ALL",
        "--device",
        "/dev/tenstorrent:/dev/tenstorrent",
        "--volume",
        "/dev/hugepages-1G:/dev/hugepages-1G:rw",
        "--volume",
        f"{MODEL_VOLUME}:{user_home_path}/cache_root:rw",
        "--shm-size",
        "32G",
    ]
    dev = True
    if dev:
        # development mounts
        docker_command += [
            "--volume",
            f"{repo_root_path}/vllm-tt-metal-llama3/src:{user_home_path}/app/src",
            "--volume",
            f"{repo_root_path}/benchmarking:{user_home_path}/app/benchmarking",
            "--volume",
            f"{repo_root_path}/evals:{user_home_path}/app/evals",
            "--volume",
            f"{repo_root_path}/locust:{user_home_path}/app/locust",
            "--volume",
            f"{repo_root_path}/utils:{user_home_path}/app/utils",
            "--volume",
            f"{repo_root_path}/tests:{user_home_path}/app/tests",
        ]

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
