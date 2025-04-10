#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import argparse
import getpass
import logging
from datetime import datetime
from pathlib import Path

from workflows.model_config import MODEL_CONFIGS
from evals.eval_config import EVAL_CONFIGS
from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from workflows.setup_host import setup_host
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    load_dotenv,
    write_dotenv,
    get_run_id,
)
from workflows.run_workflows import run_workflows
from workflows.run_docker_server import run_docker_server
from workflows.log_setup import setup_run_logger
from workflows.workflow_types import DeviceTypes, WorkflowType

logger = logging.getLogger("run_log")


def parse_arguments():
    valid_workflows = {w.name.lower() for w in WorkflowType}
    valid_devices = {device.name.lower() for device in DeviceTypes}
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
    parser.add_argument(
        "--device",
        required=True,
        choices=valid_devices,
        help=f"Device option (choices: {', '.join(valid_devices)})",
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
        "--workflow-args",
        help="Additional workflow arguments (e.g., 'param1=value1 param2=value2')",
    )
    parser.add_argument(
        "--service-port",
        type=str,
        help="SERVICE_PORT",
        default=os.getenv("SERVICE_PORT", "8000"),
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already runnning and traces captured.",
    )

    parser.add_argument("--dev-mode", action="store_true", help="Enable developer mode")

    args = parser.parse_args()

    return args


def handle_secrets(args):
    # note: can enable a path for offline without huggingface access
    # this requires pre-downloading the tokenizers and configs as well as weights
    # currently requiring HF authentication
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


def validate_local_setup(model_name: str):
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)


def validate_runtime_args(args):
    workflow_type = WorkflowType.from_string(args.workflow)
    model_config = MODEL_CONFIGS[args.model]
    if workflow_type == WorkflowType.EVALS:
        assert (
            model_config.model_name in EVAL_CONFIGS
        ), f"Model:={model_config.model_name} not found in EVAL_CONFIGS"
    if workflow_type == WorkflowType.BENCHMARKS:
        assert (
            model_config.model_name in BENCHMARK_CONFIGS
        ), f"Model:={model_config.model_name} not found in BENCHMARKS_CONFIGS"
    if workflow_type == WorkflowType.TESTS:
        raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")
    if workflow_type == WorkflowType.REPORTS:
        pass
    if workflow_type == WorkflowType.SERVER:
        pass
    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and benchmarks
        # today this will stop models defined in MODEL_CONFIGS
        # but not in EVAL_CONFIGS or BENCHMARK_CONFIGS, e.g. non-instruct models
        # a run_*.log fill will be made for the failed combination indicating this
        assert (
            model_config.model_name in EVAL_CONFIGS
        ), f"Model:={model_config.model_name} not found in EVAL_CONFIGS"
        assert (
            model_config.model_name in BENCHMARK_CONFIGS
        ), f"Model:={model_config.model_name} not found in BENCHMARKS_CONFIGS"

    if not args.device:
        # TODO: detect phy device
        raise NotImplementedError("TODO")

    assert DeviceTypes.from_string(args.device) in model_config.device_configurations

    assert not (
        args.docker_server and args.local_server
    ), "Cannot run --docker-server and --local-server"


def main():
    args = parse_arguments()
    # step 1: validate runtime
    validate_runtime_args(args)
    handle_secrets(args)
    validate_local_setup(model_name=args.model)

    # step 2: setup logging
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = get_run_id(
        timestamp=run_timestamp,
        model=args.model,
        device=args.device,
        workflow=args.workflow,
    )
    run_log_path = (
        get_default_workflow_root_log_dir() / "run_logs" / f"run_{run_id}.log"
    )
    setup_run_logger(logger=logger, run_id=run_id, run_log_path=run_log_path)
    logger.info(f"model:            {args.model}")
    logger.info(f"workflow:         {args.workflow}")
    logger.info(f"device:           {args.device}")
    logger.info(f"local-server:     {args.local_server}")
    logger.info(f"docker-server:    {args.docker_server}")
    logger.info(f"workflow_args:    {args.workflow_args}")
    version = Path("VERSION").read_text().strip()
    logger.info(f"tt-inference-server version: {version}")

    # step 3: optionally run inference server
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

    # step 4: run workflows
    skip_workflows = {WorkflowType.SERVER}
    if WorkflowType.from_string(args.workflow) not in skip_workflows:
        run_workflows(args)
        logger.info("✅ Completed run.py")
    else:
        logger.info(f"Completed {args.workflow} workflow, skipping run_workflows().")

    logger.info(
        "The output of the workflows is not checked and any errors will be in the logs above and in the saved log file."
    )
    logger.info(
        "If you encounter any issues please share the log file in a GitHuB issue and server log if available."
    )
    logger.info(f"This log file is saved on local machine at: {run_log_path}")


if __name__ == "__main__":
    main()
