#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import argparse
import getpass
import logging
import subprocess
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
    get_model_id,
)
from workflows.run_workflows import run_workflows
from workflows.run_docker_server import run_docker_server
from workflows.log_setup import setup_run_logger
from workflows.workflow_types import DeviceTypes, WorkflowType

logger = logging.getLogger("run_log")


def parse_arguments():
    valid_workflows = {w.name.lower() for w in WorkflowType}
    valid_devices = {device.name.lower() for device in DeviceTypes}
    valid_models = {config.model_name for _, config in MODEL_CONFIGS.items()}
    valid_impls = {config.impl.impl_name for _, config in MODEL_CONFIGS.items()}
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
    parser.add_argument(
        "--impl",
        required=False,
        choices=valid_impls,
        help=f"Implementation option (choices: {', '.join(valid_impls)})",
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
        "-it",
        "--interactive",
        action="store_true",
        help="Run docker in interactive mode",
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
    parser.add_argument(
        "--override-docker-image",
        type=str,
        help="Override the Docker image used by --docker-server, ignoring the model config",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        help="Tenstorrent device ID (e.g. '0' for /dev/tenstorrent/0)",
    )
    parser.add_argument(
        "--override-tt-config",
        type=str,
        help="Override TT config as JSON string (e.g., '{\"data_parallel\": 16}')",
    )

    args = parser.parse_args()

    return args


def handle_secrets(args):
    # JWT_SECRET is only required for --workflow server --docker-server
    workflow_type = WorkflowType.from_string(args.workflow)
    jwt_secret_required = workflow_type == WorkflowType.SERVER and args.docker_server
    # if interactive, user can enter secrets manually or it should not be a production deployment
    jwt_secret_required = jwt_secret_required and not args.interactive

    # HF_TOKEN is optional for client-side scripts workflows
    client_side_workflows = {WorkflowType.BENCHMARKS, WorkflowType.EVALS}
    huggingface_required = workflow_type not in client_side_workflows
    huggingface_required = huggingface_required and not args.interactive

    required_env_vars = []
    if jwt_secret_required:
        required_env_vars.append("JWT_SECRET")
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
        # read back secrets to current process env vars
        check = load_dotenv()
        assert check, "load_dotenv() failed after write_dotenv(env_vars)."
    else:
        logger.info("Using secrets from .env file.")
        for key in required_env_vars:
            assert os.getenv(
                key
            ), f"Required environment variable {key} is not set in .env file."


def infer_args(args):
    # TODO:infer hardware
    # infer the impl from the default for given model_name
    if not args.impl:
        device_type = DeviceTypes.from_string(args.device)
        for _, model_config in MODEL_CONFIGS.items():
            if (
                model_config.model_name == args.model
                and model_config.device_type == device_type
                and model_config.device_model_spec.default_impl
            ):
                args.impl = model_config.impl.impl_name
                logger.info(f"Inferred impl:={args.impl} for model:={args.model}")
                break
    if not args.impl:
        raise ValueError(
            f"Model:={args.model} does not have a default impl, you must pass --impl"
        )

    logger.info(f"Using impl:={args.impl} for model:={args.model}")


def get_current_commit_sha() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return (
        subprocess.check_output(["git", "-C", script_dir, "rev-parse", "HEAD"])
        .decode()
        .strip()
    )


def validate_local_setup(model_name: str):
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)


def validate_runtime_args(args):
    workflow_type = WorkflowType.from_string(args.workflow)

    if not args.device:
        # TODO: detect phy device
        raise NotImplementedError("Device detection not implemented yet")

    model_id = get_model_id(args.impl, args.model, args.device)

    # Check if the model_id exists in MODEL_CONFIGS (this validates device support)
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"model:={args.model} does not support device:={args.device}")

    model_config = MODEL_CONFIGS[model_id]

    if workflow_type == WorkflowType.EVALS:
        assert (
            model_config.model_name in EVAL_CONFIGS
        ), f"Model:={model_config.model_name} not found in EVAL_CONFIGS"
    if workflow_type == WorkflowType.BENCHMARKS:
        if os.getenv("OVERRIDE_BENCHMARKS"):
            logger.warning("OVERRIDE_BENCHMARKS is active, using override benchmarks")
        assert (
            model_config.model_id in BENCHMARK_CONFIGS
        ), f"Model:={model_config.model_name} not found in BENCHMARKS_CONFIGS"
    if workflow_type == WorkflowType.TESTS:
        raise NotImplementedError(f"--workflow {args.workflow} not implemented yet")
    if workflow_type == WorkflowType.REPORTS:
        pass
    if workflow_type == WorkflowType.SERVER:
        if args.local_server:
            raise NotImplementedError(
                f"Workflow {args.workflow} not implemented for --local-server"
            )
        if not (args.docker_server or args.local_server):
            raise ValueError(
                f"Workflow {args.workflow} requires --docker-server argument"
            )
    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and benchmarks
        # today this will stop models defined in MODEL_CONFIGS
        # but not in EVAL_CONFIGS or BENCHMARK_CONFIGS, e.g. non-instruct models
        # a run_*.log fill will be made for the failed combination indicating this
        assert (
            model_config.model_name in EVAL_CONFIGS
        ), f"Model:={model_config.model_name} not found in EVAL_CONFIGS"
        assert (
            model_config.model_id in BENCHMARK_CONFIGS
        ), f"Model:={model_config.model_name} not found in BENCHMARKS_CONFIGS"

    if DeviceTypes.from_string(args.device) == DeviceTypes.GPU:
        if args.docker_server or args.local_server:
            raise NotImplementedError(
                "GPU support for running inference server not implemented yet"
            )

    assert not (
        args.docker_server and args.local_server
    ), "Cannot run --docker-server and --local-server"


def main():
    args = parse_arguments()
    # step 1: infer impl from model name
    infer_args(args)

    # step 2: validate runtime
    validate_runtime_args(args)
    handle_secrets(args)
    validate_local_setup(model_name=args.model)
    model_id = get_model_id(args.impl, args.model, args.device)
    model_config = MODEL_CONFIGS[model_id]
    tt_inference_server_sha = get_current_commit_sha()

    # step 3: setup logging
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = get_run_id(
        timestamp=run_timestamp,
        model_id=model_id,
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
    logger.info(f"interactive:      {args.interactive}")
    logger.info(f"workflow_args:    {args.workflow_args}")
    if args.override_docker_image:
        logger.info(f"docker_image:     {args.override_docker_image}")
    version = Path("VERSION").read_text().strip()
    logger.info(f"tt-inference-server version: {version}")
    logger.info(f"tt-inference-server commit: {tt_inference_server_sha}")
    logger.info(f"tt-metal commit: {model_config.tt_metal_commit}")
    logger.info(f"vllm commit: {model_config.vllm_commit}")

    # step 4: optionally run inference server
    if args.docker_server:
        logger.info("Running inference server in Docker container ...")
        setup_config = setup_host(
            model_id=model_id,
            jwt_secret=os.getenv("JWT_SECRET"),
            hf_token=os.getenv("HF_TOKEN"),
            automatic_setup=os.getenv("AUTOMATIC_HOST_SETUP"),
        )
        run_docker_server(args, setup_config)
    elif args.local_server:
        logger.info("Running inference server on localhost ...")
        raise NotImplementedError("TODO")

    # step 5: run workflows
    main_return_code = 0

    skip_workflows = {WorkflowType.SERVER}
    if WorkflowType.from_string(args.workflow) not in skip_workflows:
        args.run_id = run_id
        return_codes = run_workflows(args)
        if all(return_code == 0 for return_code in return_codes):
            logger.info("✅ Completed run.py successfully.")
        else:
            main_return_code = 1
            logger.error(
                f"⛔ run.py failed with return codes: {return_codes}. See logs above for details."
            )
    else:
        logger.info(f"Completed {args.workflow} workflow, skipping run_workflows().")

    logger.info(
        "The output of the workflows is not checked and any errors will be in the logs above and in the saved log file."
    )
    logger.info(
        "If you encounter any issues please share the log file in a GitHuB issue and server log if available."
    )
    logger.info(f"This log file is saved on local machine at: {run_log_path}")

    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
