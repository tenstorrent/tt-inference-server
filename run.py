#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse

from workflows.configs import (
    model_config,
    get_default_workflow_root_log_dir,
)
from workflows.setup_host import setup_host
from workflows.utils import ensure_readwriteable_dir
from workflows.logger import get_logger
from workflows.run_local import run_local
from workflows.run_docker import run_docker

logger = get_logger()


def parse_arguments():
    valid_workflows = {"benchmarks", "evals", "server", "release", "report"}
    valid_devices = {"N150", "N300", "T3K"}
    valid_models = model_config.keys()
    # required
    parser = argparse.ArgumentParser(
        description="A CLI for running workflows with optional docker, device, and workflow-args.",
        epilog="\nAvailable models:\n  " + "\n  ".join(model_config.keys()),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=valid_models, help="Model to run"
    )
    parser.add_argument(
        "--workflow",
        required=True,
        choices=valid_workflows,
        help=f"Workflow to run (choices: {', '.join(valid_workflows)})",
    )
    # optional
    parser.add_argument("--docker", action="store_true", help="Enable docker mode")
    parser.add_argument(
        "--device",
        choices=valid_devices,
        help=f"Device option (choices: {', '.join(valid_devices)})",
    )
    parser.add_argument(
        "--workflow-args",
        help="Additional workflow arguments (e.g., 'param1=value1 param2=value2')",
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=None,
    )

    args = parser.parse_args()
    logger.info(f"model:          {args.model}")
    logger.info(f"workflow:       {args.workflow}")
    logger.info(f"docker:         {args.docker}")
    logger.info(f"device:         {args.device}")
    logger.info(f"workflow_args:  {args.workflow_args}")
    logger.info("-------------------------------")

    return args


def find_tt_metal_vllm_env():
    # PYTHON_ENV_DIR
    # TT_METAL_HOME
    pass


def detect_local_setup(model: str):
    # tt_metal_venv_path = find_tt_metal_vllm_env()
    # TODO:
    # check if tt_metal_venv_path has valid python environment
    # check ttnn exists

    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)
    pass


def main():
    # wrap in try / except to logg errors to file
    try:
        args = parse_arguments()
        if args.docker:
            logger.info("Docker mode enabled")
            setup_host(model=args.model)
            run_docker(args)
        else:
            # run outside docks user existing dev env
            logger.info("Running on host without Docker ...")
            detect_local_setup(model=args.model)
            run_local(args)

    except Exception:
        logger.error("An error occurred, stack trace:", exc_info=True)
        # TODO: output the log file path


if __name__ == "__main__":
    main()
