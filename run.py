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
import shutil
from datetime import datetime
from pathlib import Path

from workflows.model_spec import MODEL_SPECS, ModelSpec, get_runtime_model_spec
from evals.eval_config import EVAL_CONFIGS
from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from tests.test_config import TEST_CONFIGS
from workflows.setup_host import setup_host
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    load_dotenv,
    run_command,
    write_dotenv,
    get_run_id,
)
from workflows.run_workflows import run_workflows, WorkflowSetup
from workflows.run_docker_server import run_docker_server
from workflows.log_setup import setup_run_logger
from workflows.workflow_types import DeviceTypes, WorkflowType
from workflows.workflow_venvs import create_local_setup_venv

logger = logging.getLogger("run_log")


def parse_device_ids(value):
    try:
        # Split input by commas
        parts = value.split(",")
        # Convert to int and ensure all are non-negative
        device_ids = [int(p) for p in parts]
        if any(d < 0 for d in device_ids):
            raise ValueError
        return device_ids
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid device-id list: '{value}'. Must be comma-separated non-negative integers (e.g. '0' or '0,1,2')"
        )


def parse_arguments():
    valid_workflows = {w.name.lower() for w in WorkflowType}
    valid_devices = {device.name.lower() for device in DeviceTypes}

    # Build valid models set, including full HF repo names for whisper models
    valid_models = set()
    for _, config in MODEL_SPECS.items():
        valid_models.add(config.model_name)

    valid_impls = {config.impl.impl_name for _, config in MODEL_SPECS.items()}
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
        type=parse_device_ids,
        help="Tenstorrent device IDs, integer or comma-separated list of non-negative PCI indices (e.g. '0' for /dev/tenstorrent/0)",
    )
    parser.add_argument(
        "--override-tt-config",
        type=str,
        help="Override TT config as JSON string (e.g., '{\"data_parallel\": 16}')",
    )
    parser.add_argument(
        "--vllm-override-args",
        type=str,
        help='Override vLLM arguments as JSON string (e.g., \'{"max_model_len": 4096, "enable_chunked_prefill": true}\')',
    )
    parser.add_argument(
        "--reset-venvs",
        action="store_true",
        help="If there are Python dependency issues, remove .workflow_venvs/ directory so it can be automatically recreated.",
    )
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
    )
    parser.add_argument(
        "--tt-metal-python-venv-dir",
        type=str,
        help="[for --local-server] TT-Metal python venv directory, PYTHON_ENV_DIR in tt-metal usage, must be pre-built with python_env setup and vLLM installed.",
    )
    parser.add_argument(
        "--limit-samples-mode",
        type=str,
        help="Predefined eval dataset limit mappings: ['ci-nightly', 'ci-long', 'ci-commit', 'smoke-test']",
    )
    parser.add_argument(
        "--skip-system-sw-validation",
        action="store_true",
        help="Skips the system software validation step (no tt-smi or tt-topology verification)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="Enables CI-mode, which indirectly sets other flags to facilitate CI environments",
    )
    parser.add_argument(
        "--streaming",
        type=str,
        help="Enable or disable streaming for evals and benchmarks (true/false). Default is false.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        help="Enable or disable preprocessing for evals and benchmarks (true/false). Default is false.",
    )
    parser.add_argument(
        "--sdxl_num_prompts",
        type=str,
        help="Number of prompts to use for SDXL (default: 1)",
        default="100",
    )
    parser.add_argument(
        "--tools",
        type=str,
        choices=["genai", "vllm"],
        default="vllm",
        help="Benchmarking tool to use: 'genai' for genai-perf (Triton SDK), 'vllm' for vLLM benchmark_serving.py (default)",
    )

    args = parser.parse_args()

    # indirectly set additional flags for CI-mode
    if args.ci_mode:
        if "--limit-samples-mode" not in args:
            args.limit_samples_mode = "ci-nightly"
        if "--skip-system-sw-validation" not in args:
            args.skip_system_sw_validation = True

    return args


def handle_secrets(model_spec):
    args = model_spec.cli_args
    # JWT_SECRET is only required for --workflow server --docker-server
    workflow_type = WorkflowType.from_string(args.workflow)
    jwt_secret_required = workflow_type == WorkflowType.SERVER and args.docker_server
    # if interactive, user can enter secrets manually or it should not be a production deployment
    jwt_secret_required = jwt_secret_required and not args.interactive

    # HF_TOKEN is optional for client-side scripts workflows
    client_side_workflows = {WorkflowType.BENCHMARKS, WorkflowType.EVALS}
    # --docker-server requires the HF_TOKEN env var to be available
    huggingface_required = (
        workflow_type not in client_side_workflows or args.docker_server
    )
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
            assert os.getenv(key), (
                f"Required environment variable {key} is not set in .env file."
            )


def get_current_commit_sha() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return (
        subprocess.check_output(["git", "-C", script_dir, "rev-parse", "HEAD"])
        .decode()
        .strip()
    )


def validate_local_setup(model_spec, json_fpath):
    logger.info("Starting local setup validation")
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)
    WorkflowSetup.bootstrap_uv()

    def _validate_system_software_deps():
        # check, and enforce if necessary, system software dependency versions
        venv_python = create_local_setup_venv(WorkflowSetup.uv_exec)

        # fmt: off
        cmd = [
            str(venv_python),
            str(get_repo_root_path() / "workflows" / "run_local_setup_validation.py"),
            "--model-spec-json", str(json_fpath),
        ]
        # fmt: on

        return_code = run_command(cmd, logger=logger)

        if return_code != 0:
            raise ValueError(
                "⛔ validating local setup failed. See ValueErrors above for required version, and System Info section above for current system versions."
            )
        else:
            logger.info("✅ validating local setup completed")

    if (
        WorkflowType.from_string(model_spec.cli_args.workflow)
        in (WorkflowType.SERVER, WorkflowType.RELEASE)
    ) and (not model_spec.cli_args.skip_system_sw_validation):
        _validate_system_software_deps()


def format_cli_args_summary(args, model_spec):
    """Format CLI arguments and runtime info in a clean, readable format."""
    lines = [
        "",
        "=" * 60,
        "tt-inference-server run.py CLI args summary",
        "=" * 60,
        "",
        "Model Options:",
        f"  model:                      {args.model}",
        f"  device:                     {args.device}",
        f"  impl:                       {args.impl}",
        f"  workflow:                   {args.workflow}",
        f"  tools:                      {args.tools}",
        "",
        "Optional args:",
        f"  dev_mode:                   {args.dev_mode}",
        f"  docker_server:              {args.docker_server}",
        f"  local_server:               {args.local_server}",
        f"  tt_metal_python_venv_dir:   {args.tt_metal_python_venv_dir}",
        f"  service_port:               {args.service_port}",
        f"  docker_override_image:      {args.override_docker_image}",
        f"  docker_interactive:         {args.interactive}",
        f"  device_id:                  {args.device_id}",
        f"  disable_trace_capture:      {args.disable_trace_capture}",
        f"  override_tt_config:         {args.override_tt_config}",
        f"  vllm_override_args:         {args.vllm_override_args}",
        f"  model_spec_json:            {args.model_spec_json}",
        f"  workflow_args:              {args.workflow_args}",
        f"  reset_venvs:                {args.reset_venvs}",
        f"  limit-samples-mode:         {args.limit_samples_mode}",
        f"  skip_system_sw_validation:  {args.skip_system_sw_validation}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


def validate_runtime_args(model_spec):
    args = model_spec.cli_args
    workflow_type = WorkflowType.from_string(args.workflow)

    if not args.device:
        # TODO: detect phy device
        raise NotImplementedError("Device detection not implemented yet")

    model_id = model_spec.model_id

    # Check if the model_id exists in MODEL_SPECS (this validates device support)
    if model_id not in MODEL_SPECS:
        raise ValueError(f"model:={args.model} does not support device:={args.device}")

    if workflow_type == WorkflowType.EVALS:
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
    if workflow_type == WorkflowType.BENCHMARKS:
        if os.getenv("OVERRIDE_BENCHMARKS"):
            logger.warning("OVERRIDE_BENCHMARKS is active, using override benchmarks")
        assert model_spec.model_id in BENCHMARK_CONFIGS, (
            f"Model:={model_spec.model_name} not found in BENCHMARKS_CONFIGS"
        )
    if workflow_type == WorkflowType.TESTS:
        assert model_spec.model_name in TEST_CONFIGS, (
            f"Model:={model_spec.model_name} not found in TEST_CONFIGS"
        )
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

        # For partitioning Galaxy per tray as T3K
        # TODO: Add a check to verify whether these devices belong to the same tray
        if DeviceTypes.from_string(args.device) == DeviceTypes.GALAXY_T3K:
            if not args.device_id or len(args.device_id) != 8:
                raise ValueError(
                    "Galaxy T3K requires exactly 8 device IDs specified with --device-id (e.g. '0,1,2,3,4,5,6,7'). These must be devices within the same tray."
                )

    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and benchmarks
        # today this will stop models defined in MODEL_SPECS
        # but not in EVAL_CONFIGS or BENCHMARK_CONFIGS, e.g. non-instruct models
        # a run_*.log fill will be made for the failed combination indicating this
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
        assert model_spec.model_id in BENCHMARK_CONFIGS, (
            f"Model:={model_spec.model_name} not found in BENCHMARKS_CONFIGS"
        )

    if DeviceTypes.from_string(args.device) == DeviceTypes.GPU:
        if args.docker_server or args.local_server:
            raise NotImplementedError(
                "GPU support for running inference server not implemented yet"
            )

    assert not (args.docker_server and args.local_server), (
        "Cannot run --docker-server and --local-server"
    )

    if "ENABLE_AUTO_TOOL_CHOICE" in os.environ:
        raise AssertionError(
            "Setting ENABLE_AUTO_TOOL_CHOICE has been deprecated, use the VLLM_OVERRIDE_ARGS env var directly or via --vllm-override-args in run.py CLI.\n"
            'Enable auto tool choice by adding --vllm-override-args \'{"enable-auto-tool-choice": true, "tool-call-parser": <parser-name>}\' when calling run.py'
        )


def handle_maintenance_args(args):
    if args.reset_venvs:
        venvs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".workflow_venvs"
        if venvs_dir.exists():
            logger.info(f"Removing {venvs_dir}...")
            shutil.rmtree(venvs_dir)
            logger.info(f"Successfully removed {venvs_dir}")
        else:
            logger.info(f"{venvs_dir} does not exist. NOP.")


def main():
    args = parse_arguments()
    # step 0: handle maintenance args
    handle_maintenance_args(args)

    # step 1: determine model spec
    if args.model_spec_json:
        logger.warning(
            f"No validation is done, model_spec loading from JSON file: {args.model_spec_json}"
        )
        model_spec = ModelSpec.from_json(args.model_spec_json)
    else:
        model_spec = get_runtime_model_spec(args)
    model_id = model_spec.model_id

    # step 2: validate runtime
    validate_runtime_args(model_spec)
    handle_secrets(model_spec)
    tt_inference_server_sha = get_current_commit_sha()

    # step 3: setup logging
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = get_run_id(
        timestamp=run_timestamp,
        model_id=model_id,
        workflow=model_spec.cli_args.workflow,
    )
    log_path = get_default_workflow_root_log_dir()
    run_logs_path = log_path / "run_logs"
    run_model_spec_path = log_path / "run_specs"
    ensure_readwriteable_dir(run_logs_path)
    ensure_readwriteable_dir(run_model_spec_path)
    run_log_path = run_logs_path / f"run_{run_id}.log"

    setup_run_logger(logger=logger, run_id=run_id, run_log_path=run_log_path)

    # Log CLI arguments and runtime info in a clean format
    version = Path("VERSION").read_text().strip()
    logger.info(f"TT-Inference version: {version}")
    logger.info(f"TT-Inference SHA: {tt_inference_server_sha[:12]}")
    logger.info(format_cli_args_summary(args, model_spec))

    # write model spec to json file
    json_fpath = model_spec.to_json(run_id, run_model_spec_path)
    logger.info(f"Model spec saved to: {json_fpath}")

    # validate local setup after run logger has been initialized
    # and ModelSpec JSON has been written
    validate_local_setup(model_spec, json_fpath)

    # step 4: optionally run inference server
    if model_spec.cli_args.docker_server:
        logger.info("Running inference server in Docker container ...")
        setup_config = setup_host(
            model_spec=model_spec,
            jwt_secret=os.getenv("JWT_SECRET"),
            hf_token=os.getenv("HF_TOKEN"),
            automatic_setup=os.getenv("AUTOMATIC_HOST_SETUP"),
        )
        run_docker_server(model_spec, setup_config, json_fpath)
    elif model_spec.cli_args.local_server:
        logger.info("Running inference server on localhost ...")
        raise NotImplementedError("TODO")

    # step 5: run workflows
    main_return_code = 0

    skip_workflows = {WorkflowType.SERVER}
    if WorkflowType.from_string(model_spec.cli_args.workflow) not in skip_workflows:
        model_spec.cli_args.run_id = run_id
        return_codes = run_workflows(model_spec, json_fpath)
        if all(return_code == 0 for return_code in return_codes):
            logger.info("✅ Completed run.py successfully.")
        else:
            main_return_code = 1
            logger.error(
                f"⛔ run.py failed with return codes: {return_codes}. See logs above for details."
            )
    else:
        logger.info(
            f"Completed {model_spec.cli_args.workflow} workflow, skipping run_workflows()."
        )

    logger.info(
        "The output of the workflows is not checked and any errors will be in the logs above and in the saved log file."
    )
    logger.info(
        "If you encounter any issues please share the log file in a GitHub issue and server log if available."
    )
    logger.info(f"This log file is saved on local machine at: {run_log_path}")

    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
