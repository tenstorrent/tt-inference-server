# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys
from pathlib import Path
import argparse

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import MODEL_CONFIGS
from workflows.utils import run_command, get_repo_root_path
from workflows.log_setup import setup_workflow_script_logger

logger = logging.getLogger(__file__)


def build_docker_images(model_configs, force_build=False, release=False, push=False):
    """
    Builds all Docker images required by the provided ModelConfigs.
    """
    script_path = get_repo_root_path() / "workflows" / "build_docker.sh"

    unique_sha_combinations = {
        (config.tt_metal_commit, config.vllm_commit)
        for config in model_configs.values()
    }
    logger.info(f"Unique SHA combinations: {unique_sha_combinations}")

    for tt_metal_commit, vllm_commit in unique_sha_combinations:
        # fmt: off
        command = [
            str(script_path),
            "--build",
            "--tt-metal-commit", tt_metal_commit,
            "--vllm-commit", vllm_commit,
            "--ubuntu-version", "20.04",
            "--container-uid", "1000",
        ]
        # fmt: on

        # optional arguments
        if force_build:
            command.append("--force-build")
        if release:
            command.append("--release")
        if push:
            command.append("--push")

        logger.info(
            f"Building Docker image for: tt_metal_commit:={tt_metal_commit}, vllm_commit:={vllm_commit} ..."
        )
        run_command(command, logger=logger)
        logger.info("Done building Docker image.")


if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    parser = argparse.ArgumentParser(
        description="Build Docker images for model configs."
    )
    parser.add_argument(
        "--force-build", action="store_true", help="Force rebuild even if image exists."
    )
    parser.add_argument("--release", action="store_true", help="Mark build as release.")
    parser.add_argument("--push", action="store_true", help="Push containers.")

    args = parser.parse_args()

    build_docker_images(
        MODEL_CONFIGS,
        force_build=args.force_build,
        release=args.release,
        push=args.push,
    )
