# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

import argparse
import logging
import os
import sys
from pathlib import Path


# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.media_clients.media_client_factory import MediaClientFactory, MediaTaskType
from utils.url_helpers import resolve_deploy_url

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarking.benchmark_config import (
    BENCHMARK_CONFIGS,
    BenchmarkConfig,
    expand_concurrency_sweep_params,
    powers_of_two_up_to,
    select_smoke_test_benchmark_config,
)
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_config import (
    WORKFLOW_BENCHMARKS_CONFIG,
)
from workflows.workflow_types import (
    BenchmarkTaskType,
    DeviceTypes,
    EvalLimitMode,
    InferenceEngine,
    ModelType,
)


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
    ModelType.TEXT_TO_SPEECH,
    ModelType.VIDEO,
]


def _is_smoke_test_mode(runtime_config: RuntimeConfig) -> bool:
    if not runtime_config.limit_samples_mode:
        return False
    return (
        EvalLimitMode.from_string(runtime_config.limit_samples_mode)
        == EvalLimitMode.SMOKE_TEST
    )


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
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
    parser.add_argument(
        "--concurrency-sweeps",
        action="store_true",
        help="Expand benchmark sweep concurrencies to powers-of-2 up to model max.",
    )
    ret_args = parser.parse_args()
    return ret_args


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    device_str = runtime_config.device
    service_port = runtime_config.service_port
    deploy_url = resolve_deploy_url(runtime_config)

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_BENCHMARKS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    logger.info(f"service_port=: {service_port}")
    logger.info(f"output_path=: {args.output_path}")

    # set environment vars
    if (
        model_spec.inference_engine == InferenceEngine.MEDIA.value
        or model_spec.inference_engine == InferenceEngine.FORGE.value
    ):
        os.environ["OPENAI_API_KEY"] = "your-secret-key"
        os.environ["VLLM_API_KEY"] = "your-secret-key"
        logger.info("VLLM_API_KEY environment variable set to your-secret-key.")

    # Look up the evaluation configuration for the model using BENCHMARK_CONFIGS.
    if model_spec.model_id not in BENCHMARK_CONFIGS:
        message = f"No benchmark tasks defined for model: {model_spec.model_name}"
        raise ValueError(message)
    benchmark_config = BENCHMARK_CONFIGS[model_spec.model_id]
    smoke_test_mode = _is_smoke_test_mode(runtime_config)
    if smoke_test_mode:
        benchmark_config = select_smoke_test_benchmark_config(benchmark_config, device)
        logger.info("Smoke-test mode enabled; selected smoke-test benchmark config.")

    if os.getenv("ONLY_STRUCTURED_OUTPUT_BENCHMARKS"):
        benchmark_config = BenchmarkConfig(
            model_id=benchmark_config.model_id,
            tasks=[
                t
                for t in benchmark_config.tasks
                if t.task_type
                == BenchmarkTaskType.HTTP_CLIENT_VLLM_STRUCTURED_OUTPUT_API
            ],
        )
        logger.info(
            "ONLY_STRUCTURED_OUTPUT_BENCHMARKS set; running structured-output tasks only."
        )

    concurrency_sweeps = args.concurrency_sweeps or runtime_config.concurrency_sweeps
    if smoke_test_mode and concurrency_sweeps:
        logger.info("Ignoring concurrency sweeps in smoke-test mode.")
        concurrency_sweeps = False
    if concurrency_sweeps:
        max_context = model_spec.device_model_spec.max_context
        max_tokens_all_users = model_spec.device_model_spec.max_tokens_all_users
        model_max_concurrency = model_spec.device_model_spec.max_concurrency
        candidate_concurrencies = powers_of_two_up_to(model_max_concurrency)
        # TODO: get the number of perf targets from the model config instead of 1
        for task in benchmark_config.tasks[1:]:
            if device not in task.param_map:
                continue
            task.param_map[device] = expand_concurrency_sweep_params(
                task.param_map[device],
                max_context=max_context,
                max_tokens_all_users=max_tokens_all_users,
                model_max_concurrency=model_max_concurrency,
                model_name=model_spec.model_name,
                candidate_concurrencies=candidate_concurrencies,
            )

    # check for any benchmarks to run for model on given device
    all_params = [
        param
        for task in benchmark_config.tasks
        if device in task.param_map
        for param in task.param_map[device]
    ]

    if model_spec.model_type in BENCHMARKS_TASK_TYPES:
        return_code = run_benchmarks(
            all_params,
            model_spec,
            device,
            args.output_path,
            service_port,
            deploy_url=deploy_url,
        )
        return return_code

    # LLM benchmarks (incl. prefix-cache / spec-decode / serving-bench) are
    # served by the v2 engine (tt-inference-server-v2 via workflows/v2_bridge);
    # the v1 vLLM/genai-perf benchmark path has been retired. Reaching here means
    # can_route_to_v2 did not route this model to v2 (a routing regression).
    raise SystemExit(
        f"benchmarks for {model_spec.model_name!r} are served by the v2 engine; "
        "the v1 benchmark path has been retired. Reaching this branch means "
        "workflows/v2_bridge.can_route_to_v2 did not route this model to v2."
    )


def run_benchmarks(
    all_params, model_spec, device, output_path, service_port, deploy_url=None
):
    """
    Run benchmarks for the given model and device. Here we are running IMAGE, CNN, AUDIO, VIDEO benchmarks.
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
        deploy_url=deploy_url,
    )


if __name__ == "__main__":
    sys.exit(main())
