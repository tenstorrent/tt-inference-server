#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import argparse
import getpass
from pathlib import Path

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import WorkflowType
from evals.eval_config import EVAL_CONFIGS
from workflows.setup_host import setup_host
from workflows.utils import (
    ensure_readwriteable_dir,
    get_logger,
    get_default_workflow_root_log_dir,
    load_dotenv,
    write_dotenv,
)
from workflows.run_workflows import run_workflows
from workflows.run_docker_server import run_docker_server

logger = get_logger()


def parse_arguments():
    valid_workflows = {w.name.lower() for w in WorkflowType}
    valid_devices = {"N150", "N300", "T3K"}
    valid_models = MODEL_CONFIGS.keys()
    # required
    parser = argparse.ArgumentParser(
        description="A CLI for running workflows with optional docker, device, and workflow-args.",
        epilog="\nAvailable models:\n  " + "\n  ".join(valid_models),
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
    parser.add_argument(
        "--local-server", action="store_true", help="Run inference server on localhost"
    )
    parser.add_argument(
        "--docker-server",
        action="store_true",
        help="Run inference server in Docker container",
    )
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
        "--service-port",
        type=str,
        help="SERVICE_PORT",
        default=os.getenv("SERVICE_PORT", "8000"),
    )
    parser.add_argument("--dev-mode", action="store_true", help="Enable developer mode")

    args = parser.parse_args()

    logger.info(f"model:            {args.model}")
    logger.info(f"workflow:         {args.workflow}")
    logger.info(f"device:           {args.device}")
    logger.info(f"local-server:     {args.local_server}")
    logger.info(f"docker-server:    {args.docker_server}")
    logger.info(f"workflow_args:    {args.workflow_args}")

    return args


def handle_secrets(args):
    # note: can enable a path for offline without huggingface access
    # this requires pre-downloading the tokenizers and configs as well as weights
    huggingface_required = True
    required_env_vars = ["JWT_SECRET"]
    if huggingface_required:
        required_env_vars += ["HF_TOKEN"]

    # load secrets from env file or prompt user to enter them once
    if not load_dotenv():
        env_vars = {}
        for key in required_env_vars:
            _val = os.getenv(key)
            if not _val:
                _val = getpass.getpass(f"Enter your {key}: ").strip()
            env_vars[key] = _val

        assert all([env_vars[k] for k in required_env_vars])
        write_dotenv(env_vars)


def find_tt_metal_vllm_env():
    # TODO: check
    # PYTHON_ENV_DIR
    # TT_METAL_HOME
    pass


def detect_local_setup(model_name: str):
    # tt_metal_venv_path = find_tt_metal_vllm_env()
    # TODO:
    # check if tt_metal_venv_path has valid python environment
    # check ttnn exists
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)
    pass


def validate_runtime_args(args):
    workflow_type = WorkflowType.from_string(args.workflow)
    model_config = MODEL_CONFIGS[args.model]
    if workflow_type == WorkflowType.EVALS:
        assert (
            model_config.hf_model_repo in EVAL_CONFIGS
        ), f"Model:={model_config.hf_model_repo} not found in EVAL_CONFIGS"
    if workflow_type == WorkflowType.BENCHMARKS:
        print("testing ...")
        # raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")
    if workflow_type == WorkflowType.TESTS:
        raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")
    if workflow_type == WorkflowType.REPORTS:
        raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")
    if workflow_type == WorkflowType.SERVER:
        raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")

    assert not (
        args.docker_server and args.local_server
    ), "Cannot run --docker-server and --local-server"


def main():
    # wrap in try / except to log errors to file
    try:
        args = parse_arguments()
        validate_runtime_args(args)
        handle_secrets(args)
        version = Path("VERSION").read_text().strip()
        logger.info(f"tt-inference-server version: {version}")
        # optionally run inference server
        if args.docker_server:
            logger.info("Running inference server in Docker container ...")
            setup_config = setup_host(
                model_name=args.model,
                jwt_secret=os.getenv("JWT_SECRET"),
                hf_token=os.getenv("HF_TOKEN"),
            )
            run_docker_server(args, setup_config)
        elif args.local_server:
            logger.info("Running inference server on localhost ...")
            raise NotImplementedError("TODO")

        # run workflow
        detect_local_setup(model_name=args.model)
        run_workflows(args)

    except Exception:
        logger.error("An error occurred, stack trace:", exc_info=True)
        # TODO: output the log file path

    logger.info("✅ Completed run.py")
    logger.info("Running cleaning up using atexit ...")


if __name__ == "__main__":
    main()
