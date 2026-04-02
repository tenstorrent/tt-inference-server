#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import getpass
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from workflows.bootstrap_uv import bootstrap_uv
from workflows.device_utils import infer_default_device
from workflows.log_setup import setup_run_logger
from workflows.model_spec import (
    MODEL_SPECS,
    ModelSpec,
    export_model_specs_json,
    get_runtime_model_spec,
)
from workflows.run_docker_server import (
    format_docker_command,
    generate_docker_run_command,
    run_docker_server,
)
from workflows.run_local_server import run_local_server
from workflows.multihost_orchestrator import (
    MultiHostOrchestrator,
    get_expected_num_hosts,
    is_multihost_deployment,
    setup_multihost_config,
)
from workflows.validate_setup import run_multihost_validation_subprocess
from workflows.run_workflows import run_workflows
from workflows.runtime_config import RuntimeConfig
from workflows.setup_host import setup_host
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_hf_home_path,
    get_default_workflow_root_log_dir,
    get_run_id,
    load_dotenv,
    write_dotenv,
)
from workflows.validate_setup import validate_setup
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    WorkflowType,
)

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
    valid_engines = {engine.to_string() for engine in InferenceEngine}

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
        "--tt-device",
        required=False,
        choices=valid_devices,
        help=f"Tenstorrent device option (choices: {', '.join(valid_devices)}). "
        "Defaults to the largest supported device available on the host.",
    )
    parser.add_argument(
        "--device",
        required=False,
        choices=valid_devices,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--impl",
        required=False,
        choices=valid_impls,
        help=f"Implementation option (choices: {', '.join(valid_impls)})",
    )
    parser.add_argument(
        "--engine",
        required=False,
        choices=valid_engines,
        help=f"Inference engine override (choices: {', '.join(valid_engines)}). "
        "Defaults to the model spec default inference_engine.",
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
        "--bind-host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind published docker service port to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already runnning and traces captured.",
    )
    parser.add_argument(
        "--disable-metal-timeout",
        action="store_true",
        help="Disable tt-metal operation timeout and auto-triage on hang.",
    )
    parser.add_argument(
        "--percentile-report",
        action="store_true",
        help="Generate detailed percentile reports for stress tests (includes p05, p25, p50, p95, p99 for TTFT, TPOT, ITL, E2EL)",
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
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
    )
    parser.add_argument(
        "--tt-metal-python-venv-dir",
        type=str,
        help="[for --local-server] TT-Metal python venv directory, PYTHON_ENV_DIR in tt-metal usage, must be pre-built with python_env setup and vLLM installed.",
    )
    parser.add_argument(
        "--tt-metal-home",
        type=str,
        default=os.getenv("TT_METAL_HOME"),
        help="[for --local-server] Host path to a built tt-metal repo containing python_env/ and build/lib/. "
        "Defaults to TT_METAL_HOME from the environment when set.",
    )
    parser.add_argument(
        "--vllm-dir",
        type=str,
        default=os.getenv("vllm_dir"),
        help="[for --local-server] Host path to the vLLM source tree to export as vllm_dir "
        "and append to PYTHONPATH. Defaults to vllm_dir from the environment when set, "
        "otherwise tt-metal-home/vllm.",
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
        choices=["vllm", "genai", "aiperf"],
        default="vllm",
        help="Benchmarking tool to use: 'vllm' for vLLM benchmark_serving.py (default), 'genai' for genai-perf (Triton SDK), 'aiperf' for AIPerf (https://github.com/ai-dynamo/aiperf)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable vLLM API key authorization in the server (skips JWT_SECRET requirement)",
    )
    parser.add_argument(
        "--concurrency-sweeps",
        action="store_true",
        help="Expand benchmark sweep concurrencies to powers-of-2 up to model max.",
    )
    parser.add_argument(
        "--print-docker-cmd",
        action="store_true",
        help="Print simplified Docker run command and exit (does not start server)",
    )
    parser.add_argument(
        "--host-volume",
        nargs="?",
        const=str(Path(__file__).resolve().parent / "persistent_volume"),
        default=None,
        help="Host directory for persistent cache/log/tensor storage. "
        "If the flag is given without a path, defaults to persistent_volume/ in the repo root. "
        "For --docker-server, omitting it uses a Docker named volume. "
        "For --local-server, omitting it still uses the repo persistent_volume/ path.",
    )
    parser.add_argument(
        "--host-hf-cache",
        nargs="?",
        const=str(get_default_hf_home_path()),
        default=None,
        help="Host HuggingFace cache directory to reuse for model weights. "
        "If the flag is given without a path, defaults to HOST_HF_HOME, then HF_HOME, then ~/.cache/huggingface. "
        "For --local-server, tensor cache/logs still use the host volume path.",
    )
    parser.add_argument(
        "--host-weights-dir",
        type=str,
        default=None,
        help="Host directory containing pre-downloaded model weights. "
        "For --local-server, tensor cache/logs still use the host volume path.",
    )
    parser.add_argument(
        "--image-user",
        type=str,
        default="1000",
        help="UID to pass to docker run --user (default: 1000). "
        "Set to match the UID the image was built with. "
        "Default release images use UID 1000. "
        "Override only when using a custom image built with a different UID.",
    )

    args = parser.parse_args()

    if args.tt_device and args.device and args.tt_device != args.device:
        parser.error(
            "--tt-device and --device were both provided with different values. "
            "Use only one of them."
        )
    args.device = args.tt_device or args.device

    args.engine = (
        InferenceEngine.from_string(args.engine).value if args.engine else None
    )
    if not args.device:
        args.device = infer_default_device(args.model, args.engine)
    args.tt_device = args.device

    if not args.vllm_dir and args.tt_metal_home:
        args.vllm_dir = str(Path(args.tt_metal_home).expanduser() / "vllm")

    # indirectly set additional flags for CI-mode
    if args.ci_mode:
        if "--limit-samples-mode" not in args:
            args.limit_samples_mode = "ci-nightly"
        if "--skip-system-sw-validation" not in args:
            args.skip_system_sw_validation = True

    return args


def handle_secrets(runtime_config):
    # JWT_SECRET is only required for --workflow server --docker-server
    workflow_type = WorkflowType.from_string(runtime_config.workflow)
    jwt_secret_required = (
        workflow_type == WorkflowType.SERVER and runtime_config.docker_server
    )
    # if interactive, user can enter secrets manually or it should not be a production deployment
    jwt_secret_required = jwt_secret_required and not runtime_config.interactive
    # --no-auth disables authorization, so JWT_SECRET is not required
    jwt_secret_required = jwt_secret_required and not runtime_config.no_auth
    # HF_TOKEN is optional for client-side scripts workflows
    client_side_workflows = {WorkflowType.BENCHMARKS, WorkflowType.EVALS}
    # --docker-server requires the HF_TOKEN env var to be available
    huggingface_required = (
        workflow_type not in client_side_workflows or runtime_config.docker_server
    )
    huggingface_required = huggingface_required and not runtime_config.interactive

    required_env_vars = []
    if jwt_secret_required:
        required_env_vars.append("JWT_SECRET")
    if huggingface_required:
        required_env_vars += ["HF_TOKEN"]

    if (
        workflow_type == WorkflowType.SERVER
        and runtime_config.engine in ("media", "forge")
        and not runtime_config.no_auth
        and not runtime_config.interactive
        and not os.getenv("API_KEY")
    ):
        logger.warning(
            "API_KEY is not set. Using a default key for media/forge server auth. "
            "Set API_KEY in .env or as an environment variable."
        )

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


def format_cli_args_summary(runtime_config):
    """Format CLI arguments and runtime info in a clean, readable format."""
    lines = [
        "",
        "=" * 60,
        "tt-inference-server run.py CLI args summary",
        "=" * 60,
        "",
        "Model Options:",
        f"  model:                      {runtime_config.model}",
        f"  device:                     {runtime_config.device}",
        f"  impl:                       {runtime_config.impl}",
        f"  engine:                     {runtime_config.engine}",
        f"  workflow:                   {runtime_config.workflow}",
        f"  tools:                      {runtime_config.tools}",
        "",
        "Optional args:",
        f"  dev_mode:                   {runtime_config.dev_mode}",
        f"  docker_server:              {runtime_config.docker_server}",
        f"  local_server:               {runtime_config.local_server}",
        f"  no_auth:                    {runtime_config.no_auth}",
        f"  tt_metal_python_venv_dir:   {runtime_config.tt_metal_python_venv_dir}",
        f"  tt_metal_home:              {runtime_config.tt_metal_home}",
        f"  vllm_dir:                   {runtime_config.vllm_dir}",
        f"  service_port:               {runtime_config.service_port}",
        f"  bind_host:                  {runtime_config.bind_host}",
        f"  docker_override_image:      {runtime_config.override_docker_image}",
        f"  docker_interactive:         {runtime_config.interactive}",
        f"  device_id:                  {runtime_config.device_id}",
        f"  disable_trace_capture:      {runtime_config.disable_trace_capture}",
        f"  override_tt_config:         {runtime_config.override_tt_config}",
        f"  vllm_override_args:         {runtime_config.vllm_override_args}",
        f"  workflow_args:              {runtime_config.workflow_args}",
        f"  limit_samples_mode:         {runtime_config.limit_samples_mode}",
        f"  skip_system_sw_validation:  {runtime_config.skip_system_sw_validation}",
        "",
        "Host Storage Options:",
        f"  host_volume:                {runtime_config.host_volume}",
        f"  host_hf_cache:              {runtime_config.host_hf_cache}",
        f"  host_weights_dir:           {runtime_config.host_weights_dir}",
        f"  image_user:                 {runtime_config.image_user}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


def populate_model_spec_cli_args(model_spec, runtime_config):
    """Backfill model_spec.cli_args from RuntimeConfig for compatibility.

    TODO: Remove when tt-media-server no longer depends on model_spec.cli_args. #1767
    """
    if not hasattr(model_spec, "cli_args") or model_spec.cli_args is None:
        return

    cli_args = runtime_config.to_dict()
    # Legacy callers still look up tt_device from cli_args.
    cli_args["tt_device"] = runtime_config.device

    model_spec.cli_args.clear()
    model_spec.cli_args.update(cli_args)


def resolve_runtime(args):
    """Atomically build RuntimeConfig and resolve ModelSpec.

    Returns ``(runtime_config, model_spec)`` with impl/engine fully resolved.
    """
    if args.runtime_model_spec_json:
        logger.warning(
            f"No validation is done, loading runtime model spec from JSON: "
            f"{args.runtime_model_spec_json}"
        )
        model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
        runtime_config = RuntimeConfig.from_args(args)
    else:
        model_spec, resolved_impl, resolved_engine = get_runtime_model_spec(
            model=args.model,
            device=args.device,
            engine=args.engine,
            impl=args.impl,
        )
        runtime_config = RuntimeConfig.from_args(
            args, impl=resolved_impl, engine=resolved_engine
        )
        model_spec.apply_overrides(runtime_config)

    populate_model_spec_cli_args(model_spec, runtime_config)
    runtime_config.runtime_model_spec = model_spec.get_serialized_dict()

    return runtime_config, model_spec


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
    # step 00: handle maintenance args
    args = parse_arguments()
    handle_maintenance_args(args)
    # Export repo-root model_spec.json from pristine MODEL_SPECS
    repo_root = Path(__file__).resolve().parent
    export_model_specs_json(MODEL_SPECS, repo_root / "model_spec.json")

    # step 0: bootstrap uv
    bootstrap_uv()

    # step 1: build runtime config and resolve model spec atomically
    runtime_config, model_spec = resolve_runtime(args)
    model_id = model_spec.model_id

    # step 2: handle secrets
    handle_secrets(runtime_config)
    tt_inference_server_sha = get_current_commit_sha()

    # step 3: setup logging and finalize run_id
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = get_run_id(
        timestamp=run_timestamp,
        model_id=model_id,
        workflow=runtime_config.workflow,
    )
    runtime_config.run_id = run_id

    log_path = get_default_workflow_root_log_dir()
    run_logs_path = log_path / "run_logs"
    runtime_model_spec_dir = log_path / "runtime_model_specs"
    ensure_readwriteable_dir(run_logs_path)
    ensure_readwriteable_dir(runtime_model_spec_dir)
    run_log_path = run_logs_path / f"run_{run_id}.log"

    setup_run_logger(logger=logger, run_id=run_id, run_log_path=run_log_path)

    # Log CLI arguments and runtime info
    version = Path("VERSION").read_text().strip()
    logger.info(f"TT-Inference version: {version}")
    logger.info(f"TT-Inference SHA: {tt_inference_server_sha[:12]}")
    logger.info(format_cli_args_summary(runtime_config))

    # Write runtime model spec + runtime config for subprocess scripts
    json_fpath = runtime_config.to_json(
        model_spec, run_timestamp, model_id, runtime_model_spec_dir
    )
    logger.info(f"Runtime model spec saved to: {json_fpath}")

    # validate setup after run logger has been initialized
    validate_setup(model_spec, runtime_config, json_fpath)

    setup_config = None
    if runtime_config.docker_server or runtime_config.local_server:
        if runtime_config.docker_server:
            logger.info("Running inference server in Docker container ...")
        else:
            logger.info("Resolving local-server host storage ...")
        setup_config = setup_host(
            model_spec=model_spec,
            jwt_secret=os.getenv("JWT_SECRET"),
            hf_token=os.getenv("HF_TOKEN"),
            automatic_setup=os.getenv("AUTOMATIC_HOST_SETUP"),
            host_volume=runtime_config.host_volume,
            host_hf_cache=runtime_config.host_hf_cache,
            host_weights_dir=runtime_config.host_weights_dir,
            image_user=(
                runtime_config.image_user if runtime_config.docker_server else None
            ),
            local_server=runtime_config.local_server,
        )

    # step 4: optionally run inference server
    if runtime_config.docker_server:
        docker_json_fpath = None
        if runtime_config.dev_mode:
            docker_json_fpath = json_fpath
        if runtime_config.print_docker_cmd:
            if is_multihost_deployment(runtime_config):
                # Print multi-host docker commands
                expected_hosts = get_expected_num_hosts(runtime_config)
                multihost_config = setup_multihost_config(
                    model_spec, expected_hosts, dry_run=True
                )
                hosts = run_multihost_validation_subprocess(
                    multihost_config,
                    model_spec=model_spec,
                    json_fpath=json_fpath,
                    dry_run=True,
                )
                orchestrator = MultiHostOrchestrator(
                    hosts=hosts,
                    mpi_interface=multihost_config.mpi_interface,
                    shared_storage_root=multihost_config.shared_storage_root,
                    config_pkl_dir=multihost_config.config_pkl_dir,
                    docker_image=model_spec.docker_image,
                    runtime_config=runtime_config,
                    model_spec=model_spec,
                    setup_config=setup_config,
                    tt_smi_path=multihost_config.tt_smi_path,
                )
                orchestrator.prepare()

                print("\n=== Multi-Host Deployment Commands ===\n")
                for rank, host in enumerate(hosts):
                    worker_cmd, _ = orchestrator.generate_worker_docker_command(
                        host, rank
                    )
                    print(f"Worker {rank} on {host}:")
                    print(f"ssh {host} {format_docker_command(worker_cmd)}\n")

                controller_cmd, _ = orchestrator.generate_controller_docker_command()
                print("Controller (run on rank-0 host):")
                print(f"{format_docker_command(controller_cmd)}\n")
            else:
                docker_command, _ = generate_docker_run_command(
                    model_spec, runtime_config, setup_config, docker_json_fpath
                )
                print(
                    f"Docker run command:\n\n{format_docker_command(docker_command)}\n"
                )
            return 0
        run_docker_server(model_spec, runtime_config, setup_config, docker_json_fpath)
    elif runtime_config.local_server:
        logger.info("Running inference server on localhost ...")
        run_local_server(model_spec, runtime_config, json_fpath, setup_config)

    main_return_code = 0

    # step 5: run workflows
    skip_workflows = {WorkflowType.SERVER}
    if WorkflowType.from_string(runtime_config.workflow) not in skip_workflows:
        workflow_results = run_workflows(model_spec, runtime_config, json_fpath)
        if all(result.return_code == 0 for result in workflow_results):
            logger.info("Completed run.py.")
        else:
            failed_workflows = [
                f"{result.workflow_name} ({result.return_code})"
                for result in workflow_results
                if result.return_code != 0
            ]
            logger.error(
                f"run.py failed workflows: {failed_workflows}. "
                "See logs above for details."
            )
            main_return_code = 1
    else:
        logger.info(
            f"Completed {runtime_config.workflow} workflow, skipping run_workflows()."
        )

    logger.info(
        "The output of the workflows is not checked and any errors will be "
        "in the logs above and in the saved log file."
    )
    logger.info(
        "If you encounter any issues please share the log file in a GitHub "
        "issue and server log if available."
    )
    logger.info(f"This log file is saved on local machine at: {run_log_path}")
    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
