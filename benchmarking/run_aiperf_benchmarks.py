# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

"""
AIPerf Benchmark Runner for tt-inference-server.

This script runs performance benchmarks using the AIPerf tool
(https://github.com/ai-dynamo/aiperf) against a vLLM-compatible server.

AIPerf is a comprehensive benchmarking tool that measures the performance
of generative AI models served by inference solutions like vLLM.
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import jwt
import requests

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.utils import run_command
from workflows.workflow_types import DeviceTypes
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for a single benchmark run result."""

    isl: int
    osl: int
    concurrency: int
    num_prompts: int
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_tps: float = 0.0
    p99_latency: float = 0.0
    error: Optional[str] = None


class BenchmarkAggregator:
    """Aggregates benchmark results across multiple runs."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def to_dict(self) -> List[Dict]:
        return [
            {
                "isl": r.isl,
                "osl": r.osl,
                "concurrency": r.concurrency,
                "num_prompts": r.num_prompts,
                "avg_ttft_ms": r.avg_ttft_ms,
                "avg_tpot_ms": r.avg_tpot_ms,
                "avg_tps": r.avg_tps,
                "p99_latency": r.p99_latency,
                "error": r.error,
            }
            for r in self.results
        ]


def get_num_prompts(input_len: int, output_len: int, max_concurrency: int) -> int:
    """Calculate number of prompts based on sequence lengths and concurrency."""
    if output_len > 1024 or input_len > 4000:
        return 2 * max_concurrency
    if (output_len > 128 and output_len <= 1024) or (
        input_len > 128 and input_len <= 4000
    ):
        return 4 * max_concurrency
    if output_len <= 128:
        return 8 * max_concurrency
    raise ValueError(f"Invalid output_len: {output_len}")


def parse_aiperf_output(artifact_dir: str) -> Dict[str, float]:
    """
    Parse metrics from AIPerf profile export JSON file.

    AIPerf stores aggregated results in profile_export_aiperf.json in the artifact directory.
    This function extracts key metrics compatible with vLLM benchmark format.
    """
    # Prefer profile_export_aiperf.json as it contains aggregated summary metrics
    # The JSONL file contains per-request records which require different parsing
    json_candidates = [
        os.path.join(artifact_dir, "profile_export_aiperf.json"),
        os.path.join(artifact_dir, "profile_export.json"),
    ]

    # Also search in subdirectories
    for subdir in glob.glob(os.path.join(artifact_dir, "*")):
        if os.path.isdir(subdir):
            json_candidates.extend(
                [
                    os.path.join(subdir, "profile_export_aiperf.json"),
                    os.path.join(subdir, "profile_export.json"),
                ]
            )

    json_path = None
    for candidate in json_candidates:
        if os.path.exists(candidate):
            json_path = candidate
            logger.info(f"Found AIPerf output file: {json_path}")
            break

    if not json_path:
        logger.warning(f"AIPerf output not found in {artifact_dir}")
        return {}

    try:
        # Parse the JSON file containing aggregated metrics
        with open(json_path, "r") as f:
            summary = json.load(f)

        # Map AIPerf metrics to vLLM-compatible format
        # AIPerf summary format: {"metric_name": {"unit": "...", "avg": X, "p50": Y, ...}}
        metrics = {
            # TTFT metrics (Time To First Token)
            "mean_ttft_ms": summary.get("time_to_first_token", {}).get("avg", 0),
            "median_ttft_ms": summary.get("time_to_first_token", {}).get("p50", 0),
            "p99_ttft_ms": summary.get("time_to_first_token", {}).get("p99", 0),
            "std_ttft_ms": summary.get("time_to_first_token", {}).get("std", 0),
            # TPOT metrics (Time Per Output Token)
            "mean_tpot_ms": summary.get("inter_token_latency", {}).get("avg", 0),
            "median_tpot_ms": summary.get("inter_token_latency", {}).get("p50", 0),
            "p99_tpot_ms": summary.get("inter_token_latency", {}).get("p99", 0),
            "std_tpot_ms": summary.get("inter_token_latency", {}).get("std", 0),
            # ITL metrics (Inter-Token Latency)
            "mean_itl_ms": summary.get("inter_token_latency", {}).get("avg", 0),
            "median_itl_ms": summary.get("inter_token_latency", {}).get("p50", 0),
            "p99_itl_ms": summary.get("inter_token_latency", {}).get("p99", 0),
            "std_itl_ms": summary.get("inter_token_latency", {}).get("std", 0),
            # E2EL metrics (End-to-End Latency)
            "mean_e2el_ms": summary.get("request_latency", {}).get("avg", 0),
            "median_e2el_ms": summary.get("request_latency", {}).get("p50", 0),
            "p99_e2el_ms": summary.get("request_latency", {}).get("p99", 0),
            "std_e2el_ms": summary.get("request_latency", {}).get("std", 0),
            # Throughput metrics
            "output_token_throughput": summary.get("output_token_throughput", {}).get(
                "avg", 0
            ),
            "total_token_throughput": summary.get("total_token_throughput", {}).get(
                "avg", 0
            ),
            "request_throughput": summary.get("request_throughput", {}).get("avg", 0),
            # Request counts
            "completed": int(summary.get("request_count", {}).get("avg", 0)),
            "total_input_tokens": int(
                summary.get("input_sequence_length", {}).get("avg", 0)
                * summary.get("request_count", {}).get("avg", 0)
            ),
            "total_output_tokens": int(
                summary.get("output_sequence_length", {}).get("avg", 0)
                * summary.get("request_count", {}).get("avg", 0)
            ),
        }

        return metrics

    except Exception as e:
        logger.error(f"Error parsing AIPerf output: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {}


def print_detailed_results(
    isl: int, osl: int, concurrency: int, num_requests: int, metrics: Dict[str, float]
) -> None:
    """Print detailed benchmark results matching vLLM format."""
    logger.info("=" * 80)
    logger.info(
        f"Serving Benchmark Result: ISL={isl}, OSL={osl}, Concurrency={concurrency}"
    )
    logger.info("=" * 80)
    logger.info(
        f"Successful requests:                     {metrics.get('completed', num_requests)}"
    )
    logger.info(
        f"Request throughput (req/s):              {metrics.get('request_throughput', 0):.2f}"
    )
    logger.info(
        f"Output token throughput (tok/s):         {metrics.get('output_token_throughput', 0):.2f}"
    )
    logger.info(
        f"Total token throughput (tok/s):          {metrics.get('total_token_throughput', 0):.2f}"
    )
    logger.info(
        f"Total input tokens:                      {metrics.get('total_input_tokens', 0)}"
    )
    logger.info(
        f"Total output tokens:                     {metrics.get('total_output_tokens', 0)}"
    )
    logger.info(
        f"Mean TTFT (ms):                          {metrics.get('mean_ttft_ms', 0):.2f}"
    )
    logger.info(
        f"Mean TPOT (ms):                          {metrics.get('mean_tpot_ms', 0):.2f}"
    )
    logger.info(
        f"Mean E2EL (ms):                          {metrics.get('mean_e2el_ms', 0):.2f}"
    )
    logger.info("=" * 80)


def save_individual_result(
    metrics: Dict[str, float],
    isl: int,
    osl: int,
    concurrency: int,
    num_requests: int,
    model_name: str,
    model_id: str,
    output_dir: str,
    images: int = 0,
    image_height: int = 0,
    image_width: int = 0,
) -> str:
    """Save individual benchmark result in aiperf format."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use aiperf_ prefix to differentiate from vLLM benchmarks
    # Include image parameters in filename if it's an image benchmark
    if images > 0:
        filename = f"aiperf_benchmark_{model_id}_{timestamp}_isl-{isl}_osl-{osl}_maxcon-{concurrency}_n-{num_requests}_images-{images}_height-{image_height}_width-{image_width}.json"
    else:
        filename = f"aiperf_benchmark_{model_id}_{timestamp}_isl-{isl}_osl-{osl}_maxcon-{concurrency}_n-{num_requests}.json"
    filepath = os.path.join(output_dir, filename)

    # Build vLLM-compatible JSON structure
    result = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "aiperf",
        "model_id": model_name,
        "tokenizer_id": model_name,
        "num_prompts": num_requests,
        "max_concurrency": concurrency,
        **metrics,  # All harmonized metrics
    }

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result saved to: {filepath}")
    return filepath


def run_benchmark(
    isl: int,
    osl: int,
    concurrency: int,
    aggregator,
    model_name: str,
    model_id: str,
    tokenizer: str,
    url: str,
    auth_token: str,
    artifact_base: str,
    output_dir: str,
    venv_python: Path,
    verbose: bool = False,
    images: int = 0,
    image_height: int = 0,
    image_width: int = 0,
) -> int:
    """Run a single AIPerf benchmark with specified parameters.

    Args:
        images: Number of images per prompt (0 for text-only)
        image_height: Height of images in pixels
        image_width: Width of images in pixels
    """
    num_prompts = get_num_prompts(isl, osl, concurrency)

    run_id = f"bench_{isl}_{osl}_{concurrency}"
    if images > 0:
        run_id += f"_img{images}_{image_height}x{image_width}"
    artifact_dir = os.path.join(artifact_base, run_id)

    if images > 0:
        logger.info(
            f"Running: ISL={isl}, OSL={osl}, Concur={concurrency}, N={num_prompts}, Images={images}, Size={image_width}x{image_height}"
        )
    else:
        logger.info(
            f"Running: ISL={isl}, OSL={osl}, Concur={concurrency}, N={num_prompts}"
        )

    if not auth_token:
        logger.warning("No auth token provided, benchmark may fail")

    # Clean up previous artifact dir
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    # Build aiperf command
    # Format URL properly
    if not url.startswith("http"):
        url = f"http://{url}"

    cmd = [
        str(venv_python),
        "-m",
        "aiperf",
        "profile",
        "--model",
        model_name,
        "--tokenizer",
        tokenizer,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(num_prompts),
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(osl),
        "--output-tokens-stddev",
        "0",
        "--url",
        url,
        "--artifact-dir",
        artifact_dir,
    ]

    # Add image parameters if this is an image benchmark
    if images > 0:
        cmd.extend(
            [
                "--image-width-mean",
                str(image_width),
                "--image-width-stddev",
                "0",
                "--image-height-mean",
                str(image_height),
                "--image-height-stddev",
                "0",
                "--image-batch-size",
                str(images),
            ]
        )

    # Add auth token if available
    if auth_token:
        cmd.extend(["--api-key", auth_token])

    # Set environment
    env_vars = os.environ.copy()

    # Run the benchmark
    logger.info(f"Executing: {' '.join(cmd)}")
    return_code = run_command(command=cmd, logger=logger, env=env_vars)

    if return_code != 0:
        logger.error(f"AIPerf benchmark failed with return code: {return_code}")
        aggregator.add_result(
            BenchmarkResult(
                isl, osl, concurrency, num_prompts, error="Benchmark failed"
            )
        )
        return return_code

    # Parse results
    metrics = parse_aiperf_output(artifact_dir)

    if metrics:
        print_detailed_results(isl, osl, concurrency, num_prompts, metrics)
        save_individual_result(
            metrics,
            isl,
            osl,
            concurrency,
            num_prompts,
            model_name,
            model_id,
            output_dir,
            images,
            image_height,
            image_width,
        )
        aggregator.add_result(
            BenchmarkResult(
                isl,
                osl,
                concurrency,
                num_prompts,
                avg_ttft_ms=metrics.get("mean_ttft_ms", 0),
                avg_tpot_ms=metrics.get("mean_tpot_ms", 0),
                avg_tps=1000 / max(metrics.get("mean_tpot_ms", 1), 0.001),
            )
        )
    else:
        aggregator.add_result(
            BenchmarkResult(
                isl, osl, concurrency, num_prompts, error="Failed to parse output"
            )
        )

    return return_code


def send_warmup_requests(
    prompt_client: "PromptClient",
    model_spec: "ModelSpec",
    num_requests: int = 3,
) -> bool:
    """
    Send warm-up requests to the server to initialize CUDA kernels and KV cache.

    This prevents cold-start overhead from affecting the first benchmark.
    The warm-up requests use small input/output sizes to minimize time while
    still triggering all necessary initializations.

    Args:
        prompt_client: Client for sending requests to the server
        model_spec: Model specification
        num_requests: Number of warm-up requests to send

    Returns:
        True if all warm-up requests succeeded, False otherwise
    """
    warmup_prompts = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Say 'ready' if you can hear me.",
    ]

    success_count = 0
    for i in range(min(num_requests, len(warmup_prompts))):
        try:
            logger.info(f"Sending warm-up request {i + 1}/{num_requests}...")

            # Use the prompt client's URL and auth
            url = f"http://localhost:{prompt_client.env_config.service_port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            # Add auth if available
            if prompt_client.env_config.jwt_secret:
                import jwt as jwt_lib

                json_payload = {"team_id": "tenstorrent", "token_id": "warmup"}
                token = jwt_lib.encode(
                    json_payload, prompt_client.env_config.jwt_secret, algorithm="HS256"
                )
                headers["Authorization"] = f"Bearer {token}"

            payload = {
                "model": model_spec.hf_model_repo,
                "messages": [{"role": "user", "content": warmup_prompts[i]}],
                "max_tokens": 32,  # Small output to minimize time
                "stream": False,
            }

            response = requests.post(url, json=payload, headers=headers, timeout=120)

            if response.status_code == 200:
                success_count += 1
                logger.info(f"Warm-up request {i + 1} succeeded")
            else:
                logger.warning(
                    f"Warm-up request {i + 1} failed with status {response.status_code}: {response.text[:200]}"
                )

        except Exception as e:
            logger.warning(f"Warm-up request {i + 1} failed with exception: {e}")

    return success_count == num_requests


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run AIPerf benchmarks")
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
        help="Device name (passed for consistency, read from model spec JSON)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (passed for consistency, read from model spec JSON)",
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
    return parser.parse_args()


def main():
    """Main entry point for AIPerf benchmarks."""
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    jwt_secret = args.jwt_secret
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    device_str = cli_args.get("device")
    service_port = cli_args.get("service_port", os.getenv("SERVICE_PORT", "8000"))

    device = DeviceTypes.from_string(device_str)
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    logger.info(f"service_port=: {service_port}")
    logger.info(f"output_path=: {args.output_path}")

    # Set environment vars
    auth_token = ""
    if jwt_secret:
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        auth_token = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )

    # Get venv config for aiperf
    from workflows.workflow_types import WorkflowVenvType

    venv_config = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_AIPERF]

    # Look up the benchmark configuration for the model
    if model_spec.model_id not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"No benchmark tasks defined for model: {model_spec.model_name}"
        )
    benchmark_config = BENCHMARK_CONFIGS[model_spec.model_id]

    # Get all benchmark params for this device
    all_params = [
        param
        for task in benchmark_config.tasks
        if device in task.param_map
        for param in task.param_map[device]
    ]

    if not all_params:
        raise ValueError(
            f"No benchmark tasks defined for model: {model_spec.model_name} on device: {device.name}"
        )

    # Check for limit_samples_mode (smoke-test, ci-commit) to enable debug mode
    limit_samples_mode_str = cli_args.get("limit_samples_mode")
    if limit_samples_mode_str:
        from workflows.workflow_types import EvalLimitMode

        limit_mode = EvalLimitMode.from_string(limit_samples_mode_str)
        if limit_mode in (EvalLimitMode.SMOKE_TEST, EvalLimitMode.CI_COMMIT):
            # Limit to 2 benchmarks for quick testing (1 small, 1 medium)
            original_count = len(all_params)
            all_params = all_params[:2]
            logger.info(
                f"Enabling AIPerf debug mode (2 benchmarks) for limit_samples_mode={limit_samples_mode_str}"
            )
            logger.info(
                f"Reduced from {original_count} to {len(all_params)} benchmarks"
            )

    # Log benchmark parameters
    log_str = "Running AIPerf benchmarks for:\n"
    log_str += f"  {'#':<3} {'Type':<8} {'isl':<6} {'osl':<6} {'Concur':<8} {'N':<6} {'Images':<8}\n"
    log_str += f"  {'-' * 3:<3} {'-' * 8:<8} {'-' * 6:<6} {'-' * 6:<6} {'-' * 8:<8} {'-' * 6:<6} {'-' * 8:<8}\n"
    for i, param in enumerate(all_params, 1):
        img_str = ""
        if param.task_type in ("image", "vlm"):
            img_str = (
                f"{param.images_per_prompt}@{param.image_width}x{param.image_height}"
            )
        log_str += f"  {i:<3} {param.task_type:<8} {param.isl:<6} {param.osl:<6} {param.max_concurrency:<8} {param.num_prompts:<6} {img_str:<8}\n"
    logger.info(log_str)

    # Wait for server to be ready
    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = jwt_secret
    env_config.service_port = service_port
    env_config.vllm_model = model_spec.hf_model_repo

    prompt_client = PromptClient(env_config, model_spec=model_spec)
    if not prompt_client.wait_for_healthy():
        logger.error("vLLM server is not healthy. Aborting benchmarks.")
        return 1

    # Send warm-up requests to ensure server is fully initialized
    # This prevents cold-start overhead from affecting the first benchmark
    logger.info("Sending warm-up requests to initialize server...")
    warmup_success = send_warmup_requests(prompt_client, model_spec, num_requests=3)
    if not warmup_success:
        logger.warning("Warm-up requests failed, but continuing with benchmarks")
    else:
        logger.info("Warm-up completed successfully")

    # Create artifact directory
    artifact_base = venv_config.venv_path / "artifacts" / model_spec.model_id
    artifact_base.mkdir(parents=True, exist_ok=True)

    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)

    # Run benchmarks
    aggregator = BenchmarkAggregator()
    return_codes = []

    for i, params in enumerate(all_params, 1):
        # Health check
        health_check = prompt_client.get_health()
        if health_check.status_code != 200:
            logger.error("vLLM server is not healthy. Aborting benchmarks.")
            return 1

        logger.info(f"Running benchmark {model_spec.model_name}: {i}/{len(all_params)}")

        # Add delay between runs
        time.sleep(2)

        # Extract image parameters if this is an image benchmark
        images = 0
        image_height = 0
        image_width = 0
        if params.task_type in ("image", "vlm"):
            images = getattr(params, "images_per_prompt", 1)
            image_height = getattr(params, "image_height", 0)
            image_width = getattr(params, "image_width", 0)

        return_code = run_benchmark(
            isl=params.isl,
            osl=params.osl,
            concurrency=params.max_concurrency,
            aggregator=aggregator,
            model_name=model_spec.hf_model_repo,
            model_id=model_spec.model_id,
            tokenizer=model_spec.hf_model_repo,
            url=f"localhost:{service_port}",
            auth_token=auth_token,
            artifact_base=str(artifact_base),
            output_dir=args.output_path,
            venv_python=venv_config.venv_python,
            images=images,
            image_height=image_height,
            image_width=image_width,
        )
        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("Completed AIPerf benchmarks")
        main_return_code = 0
    else:
        logger.error(
            f"AIPerf benchmarks failed with return codes: {return_codes}. See logs above for details."
        )
        main_return_code = 1

    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
