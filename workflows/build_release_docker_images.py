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


def build_docker_images(
    model_configs,
    force_build=False,
    release=False,
    push=False,
    ubuntu_version="20.04",
    build_metal_commit=None,
):
    """
    Builds all Docker images required by the provided ModelConfigs.
    """
    script_path = get_repo_root_path() / "workflows" / "build_docker.sh"

    # Filter combinations if build_metal_commit is specified
    unique_sha_combinations = {
        (config.tt_metal_commit, config.vllm_commit)
        for config in model_configs.values()
    }
    if build_metal_commit:
        unique_sha_combinations = {
            combo for combo in unique_sha_combinations if combo[0] == build_metal_commit
        }

    if not unique_sha_combinations:
        logger.warning(
            f"No configurations found with tt_metal_commit={build_metal_commit}"
        )
        return
    logger.info(f"Unique SHA combinations to check: {unique_sha_combinations}")

    for tt_metal_commit, vllm_commit in unique_sha_combinations:
        # fmt: off
        command = [
            str(script_path),
            "--build",
            "--tt-metal-commit", tt_metal_commit,
            "--ubuntu-version", ubuntu_version,
            "--container-uid", "1000",
        ]
        # fmt: on

        # optional arguments
        if vllm_commit is not None:
            command.extend(("--vllm-commit", vllm_commit))
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
    parser.add_argument(
        "--ubuntu-version",
        type=str,
        default="22.04",
        help="Ubuntu version to use for the base image.",
        choices={"20.04", "22.04"},
    )
    parser.add_argument(
        "--build-metal-commit",
        type=str,
        help="Only build containers with this exact tt-metal commit",
    )
    args = parser.parse_args()

    build_docker_images(
        MODEL_CONFIGS,
        force_build=args.force_build,
        release=args.release,
        push=args.push,
        ubuntu_version=args.ubuntu_version,
        build_metal_commit=args.build_metal_commit,
    )
