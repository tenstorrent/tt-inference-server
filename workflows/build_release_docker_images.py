# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging

from workflows.model_config import MODEL_CONFIGS
from workflows.utils import run_command, get_repo_root_path
from workflows.log_setup import setup_workflow_script_logger

logger = logging.getLogger(__file__)


def build_docker_images(model_configs):
    """
    Builds all Docker images required by the provided ModelConfigs.
    """
    script_path = get_repo_root_path() / "workflows" / "build_docker.sh"
    for model_name, config in model_configs.items():
        # fmt: off
        command = [
            str(script_path),
            "--build",
            "--tt-metal-commit", config.tt_metal_commit,
            "--vllm-commit", config.vllm_commit,
            "--ubuntu-version", "20.04",
            "--container-uid", "1000"
        ]
        # fmt: on

        logger.info(f"Building Docker image for {model_name}...")
        logger.info(f"{config.docker_image}")
        run_command(command, logger=logger)
        logger.info("Done building Docker image.")


if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    build_docker_images(MODEL_CONFIGS)
