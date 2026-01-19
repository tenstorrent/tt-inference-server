# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import jwt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.media_clients.media_client_factory import MediaClientFactory, MediaTaskType

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from benchmarking.run_genai_benchmarks import run_genai_benchmarks
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec, ModelType
from workflows.utils import run_command
from workflows.workflow_config import (
    WORKFLOW_BENCHMARKS_CONFIG,
)
from workflows.workflow_types import DeviceTypes
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

# fmt: off
IMAGE_RESOLUTIONS = [
    (512, 512),
    (512, 1024),
    (1024, 512),
    (1024, 1024)
    ]
# fmt: on

BENCHMARKS_TASK_TYPES = [
    ModelType.IMAGE,
    ModelType.CNN,
    ModelType.AUDIO,
    ModelType.EMBEDDING,
]


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for benchmark output",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name",
        required=False,
    )

    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    ret_args = parser.parse_args()
    return ret_args


def build_benchmark_command(
    task,
    benchmark_script,
    params,
    output_path,
    service_port,
    benchmark_config,
    model_spec,
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    isl = params.isl
    osl = params.osl
    max_concurrency = params.max_concurrency
    num_prompts = params.num_prompts
    if params.task_type == "image":
        result_filename = (
            Path(output_path)
            / f"benchmark_{model_spec.model_id}_{run_timestamp}_isl-{isl}_osl-{osl}_maxcon-{max_concurrency}_n-{num_prompts}_images-{params.images_per_prompt}_height-{params.image_height}_width-{params.image_width}.json"
        )
    else:
        result_filename = (
            Path(output_path)
            / f"benchmark_{model_spec.model_id}_{run_timestamp}_isl-{isl}_osl-{osl}_maxcon-{max_concurrency}_n-{num_prompts}.json"
        )

    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    # fmt: off
    cmd = [
        str(task_venv_config.venv_python), str(benchmark_script),
        "--backend", ("vllm" if params.task_type == "text" else "openai-chat"),
        "--model", model_spec.hf_model_repo,
        "--port", str(service_port),
        "--dataset-name", "cleaned-random",
        "--max-concurrency", str(max_concurrency),
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--ignore-eos",  # Ignore EOS tokens to force max output length as set
        "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
        "--save-result",
        "--result-filename", str(result_filename),
    ]

    # Add multimodal parameters if the model supports it
    if params.task_type == "image":
        if params.image_height and params.image_width:
            cmd.extend([
                "--random-images-per-prompt", str(params.images_per_prompt),
                "--random-image-height", str(params.image_height),
                "--random-image-width", str(params.image_width),
                "--endpoint", "/v1/chat/completions"
            ])
    # fmt: on
    return cmd


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    jwt_secret = args.jwt_secret
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    device_str = cli_args.get("device")
    service_port = cli_args.get("service_port", os.getenv("SERVICE_PORT", "8000"))
    disable_trace_capture = cli_args.get("disable_trace_capture", False)

    # Automatically control trace capture based on has_builtin_warmup
    # Only apply automatic logic if user hasn't explicitly set --disable-trace-capture
    if not disable_trace_capture and hasattr(model_spec, "has_builtin_warmup"):
        if model_spec.has_builtin_warmup:
            # Model has builtin warmup - disable client-side trace capture
            disable_trace_capture = True
            logger.info(
                "Model has builtin warmup (has_builtin_warmup=True), "
                "automatically disabling trace capture for benchmarks workflow"
            )

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_BENCHMARKS_CONFIG
    # Check for tools selection (genai vs vllm)
    tools = cli_args.get("tools", "vllm")
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    logger.info(f"service_port=: {service_port}")
    logger.info(f"output_path=: {args.output_path}")
    logger.info(f"tools=: {tools}")

    # Route to genai-perf benchmarks if tools=genai
    if tools == "genai":
        logger.info("Using genai-perf (Triton SDK) for benchmarking")

        # Determine debug mode from limit_samples_mode
        limit_samples_mode_str = cli_args.get("limit_samples_mode")
        debug_mode = False
        if limit_samples_mode_str:
            from workflows.workflow_types import EvalLimitMode

            limit_mode = EvalLimitMode.from_string(limit_samples_mode_str)
            # Enable debug mode for quick test modes
            if limit_mode in (EvalLimitMode.SMOKE_TEST, EvalLimitMode.CI_COMMIT):
                debug_mode = True
                logger.info(
                    f"Enabling genai-perf debug mode (2 benchmarks) for limit_samples_mode={limit_samples_mode_str}"
                )

        return run_genai_benchmarks(
            model_spec=model_spec,
            output_path=args.output_path,
            jwt_secret=jwt_secret,
            service_port=service_port,
            debug=debug_mode,
        )

    # set environment vars
    if jwt_secret:
        # If jwt-secret is provided, generate the JWT and set OPENAI_API_KEY.
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    # copy env vars to pass to subprocesses
    os.environ["OPENAI_API_KEY"] = "your-secret-key"
    os.environ["VLLM_API_KEY"] = "your-secret-key"
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using BENCHMARK_CONFIGS.
    if model_spec.model_id not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"No benchmark tasks defined for model: {model_spec.model_name}"
        )
    benchmark_config = BENCHMARK_CONFIGS[model_spec.model_id]

    # check for any benchmarks to run for model on given device
    all_params = [
        param
        for task in benchmark_config.tasks
        if device in task.param_map
        for param in task.param_map[device]
    ]

    if model_spec.model_type in BENCHMARKS_TASK_TYPES:
        return run_benchmarks(
            all_params, model_spec, device, args.output_path, service_port
        )

    log_str = "Running benchmarks for:\n"
    log_str += f"  {'#':<3} {'isl':<10} {'osl':<10} {'max_concurrency':<15} {'num_prompts':<12}\n"
    log_str += (
        f"  {'-' * 3:<3} {'-' * 10:<10} {'-' * 10:<10} {'-' * 15:<15} {'-' * 12:<12}\n"
    )
    for i, param in enumerate(all_params, 1):
        if param.task_type == "text":
            log_str += f"  {i:<3} {param.isl:<10} {param.osl:<10} {param.max_concurrency:<15} {param.num_prompts:<12}\n"
    if "image" in model_spec.supported_modalities:
        log_str += "Running image benchmarks for:\n"
        log_str += f"  {'#':<3} {'isl':<10} {'osl':<10} {'max_concurrency':<15} {'images_per_prompt':<12} {'image_height':<12} {'image_width':<12} {'num_prompts':<12}\n"
        log_str += f"  {'-' * 3:<3} {'-' * 10:<10} {'-' * 10:<10} {'-' * 15:<15} {'-' * 12:<12} {'-' * 12:<12} {'-' * 12:<12} {'-' * 12:<12}\n"
        for i, param in enumerate(all_params, 1):
            if param.task_type == "image":
                log_str += f"  {i:<3} {param.isl:<10} {param.osl:<10} {param.max_concurrency:<15} {param.images_per_prompt:<12} {param.image_height:<12} {param.image_width:<12} {param.num_prompts:<12}\n"
    logger.info(log_str)

    assert all_params, (
        f"No benchmark tasks defined for model: {model_spec.model_name} on device: {device.name}"
    )

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = jwt_secret
    env_config.service_port = service_port
    env_config.vllm_model = model_spec.hf_model_repo

    # Use intelligent timeout - automatically determines 90 minutes for first run, 30 minutes for subsequent runs
    prompt_client = PromptClient(env_config, model_spec=model_spec)
    if not prompt_client.wait_for_healthy():
        logger.error("⛔️ vLLM server is not healthy. Aborting benchmarks. ")
        return 1

    # keep track of captured traces to avoid re-running requests
    captured_traces = set()

    # Run benchmarks
    return_codes = []
    for task in benchmark_config.tasks:
        venv_config = VENV_CONFIGS[task.workflow_venv_type]
        benchmark_script = venv_config.venv_path / "scripts" / "benchmark_serving.py"
        if device in task.param_map:
            params_list = task.param_map[device]
            context_lens = [(params.isl, params.osl) for params in params_list]
            # de-dupe
            context_lens_set = set(context_lens)
            context_lens_set.difference_update(captured_traces)
            # ascending order of input sequence length
            sorted_context_lens_set = sorted(context_lens_set)
            if not disable_trace_capture:
                if "image" in model_spec.supported_modalities:
                    prompt_client.capture_traces(
                        context_lens=list(sorted_context_lens_set),
                        timeout=1200.0,
                        image_resolutions=IMAGE_RESOLUTIONS,
                    )
                else:
                    prompt_client.capture_traces(
                        context_lens=list(sorted_context_lens_set), timeout=1200.0
                    )
                captured_traces.update(sorted_context_lens_set)
            for i, params in enumerate(params_list, 1):
                health_check = prompt_client.get_health()
                if health_check.status_code != 200:
                    logger.error("⛔️ vLLM server is not healthy. Aborting benchmarks.")
                    return 1

                logger.info(
                    f"Running benchmark {model_spec.model_name}: {i}/{len(params_list)}"
                )
                # Add a small delay between runs to ensure system stability
                time.sleep(2)
                cmd = build_benchmark_command(
                    task,
                    benchmark_script,
                    params=params,
                    output_path=args.output_path,
                    service_port=service_port,
                    benchmark_config=benchmark_config,
                    model_spec=model_spec,
                )
                return_code = run_command(command=cmd, logger=logger, env=env_vars)
                return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed benchmarks")
        main_return_code = 0
    else:
        logger.error(
            f"⛔ benchmarks failed with return codes: {return_codes}. See logs above for details."
        )
        main_return_code = 1

    return main_return_code


def run_benchmarks(all_params, model_spec, device, output_path, service_port):
    """
    Run benchmarks for the given model and device. Here we are running IMAGE, CNN
    and AUDIO benchmarks.
    """
    logger.info(
        f"Running benchmarks for model: {model_spec.model_name} on device: {device.name}"
    )
    return MediaClientFactory.run_media_task(
        model_spec,
        all_params,
        device,
        output_path,
        service_port,
        task_type=MediaTaskType.BENCHMARK,
    )


if __name__ == "__main__":
    sys.exit(main())
