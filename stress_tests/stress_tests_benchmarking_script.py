#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Stress Tests Benchmarking Script

This script provides the same functionality as benchmark_serving.py but uses
our homebrewed utilities from the utils/ modules. It produces identical JSON
output format for compatibility with the existing workflow infrastructure.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from utils.cleaned_prompt_generation import generate_stable_prompt_tokens
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig

# Note: capture_traces functionality is built into PromptClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class BenchmarkMetrics:
    """Metrics structure matching benchmark_serving.py output"""

    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


@dataclass
class RequestOutput:
    """Request output structure matching benchmark_serving.py"""

    success: bool = False
    generated_text: str = ""
    prompt_len: int = 0
    output_tokens: int = 0
    ttft: float = 0.0
    itl: List[float] = None
    latency: float = 0.0
    error: str = ""

    def __post_init__(self):
        if self.itl is None:
            self.itl = []


def remove_prefix(text: str, prefix: str) -> str:
    """Remove prefix from text if present."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


async def async_request_openai_completions(
    prompt: str,
    prompt_len: int,
    output_len: int,
    model_name: str,
    api_url: str,
    auth_headers: Dict[str, str],
    ignore_eos: bool = True,
) -> RequestOutput:
    """
    Make async HTTP request to OpenAI-compatible completions API.
    """
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": output_len,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }

        if ignore_eos:
            payload["ignore_eos"] = ignore_eos

        output = RequestOutput()
        output.prompt_len = prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        output_tokens = 0

        try:
            async with session.post(
                url=api_url, json=payload, headers=auth_headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            pass
                        else:
                            try:
                                data = json.loads(chunk)

                                # NOTE: Some completion API might have a last
                                # usage summary response without a token so we
                                # want to check a token was generated
                                if choices := data.get("choices"):
                                    # Note that text could be empty here
                                    # e.g. for special tokens
                                    text = choices[0].get("text")
                                    timestamp = time.perf_counter()
                                    # First token
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        ttft = time.perf_counter() - st
                                        output.ttft = ttft
                                    # Decoding phase
                                    else:
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    most_recent_timestamp = timestamp
                                    generated_text += text or ""
                                    output_tokens += 1

                                # Check for usage stats
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get(
                                        "completion_tokens"
                                    )
                            except json.JSONDecodeError:
                                # Skip malformed chunks
                                continue

                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                    output.output_tokens = output_tokens
                else:
                    output.error = (
                        f"HTTP {response.status}: {response.reason or 'Unknown error'}"
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        return output


def generate_cleaned_random_prompts_using_server(
    num_prompts: int,
    input_len: int,
    output_len: int,
    model_name: str,
    client: PromptClient,
    seed: Optional[int] = None,
    use_server_tokenizer: bool = False,
) -> List[Tuple[str, int, int, Optional[Dict]]]:
    """
    Generate cleaned random prompts using configurable tokenization (server-side or client-side).
    Returns list of (prompt_text, prompt_len, output_len, multi_modal_data) tuples.

    Args:
        num_prompts: Number of prompts to generate
        input_len: Target input sequence length
        output_len: Target output sequence length
        model_name: Model name for tokenization
        client: PromptClient instance
        seed: Random seed for reproducibility
        use_server_tokenizer: If True, use server-side tokenization; if False, use client-side
    """
    logger.info(
        f"Generating {num_prompts} cleaned random prompts using {'server' if use_server_tokenizer else 'client'}-side tokenizer..."
    )

    # Load tokenizer once outside the loop for efficiency (only if using client-side)
    tokenizer = None
    if not use_server_tokenizer:
        from utils.cleaned_prompt_generation import get_tokenizer

        tokenizer, actual_model = get_tokenizer(model_name, fallback_model="gpt2")

    prompt_tuples = []

    for i in range(num_prompts):
        # Generate stable prompt tokens using the specified tokenization method
        final_tokens = generate_stable_prompt_tokens(
            input_length=input_len,
            max_length=input_len,
            model_name=model_name,
            server_tokenizer=use_server_tokenizer,
            client=client,
            seed=seed + i if seed is not None else None,
            preloaded_tokenizer=tokenizer if not use_server_tokenizer else None,
        )

        # Convert tokens back to text using SAME tokenization method
        if use_server_tokenizer:
            detokenize_result = client.detokenize(final_tokens, model_name)
            if "error" in detokenize_result:
                raise RuntimeError(
                    f"Server detokenization failed: {detokenize_result['error']}"
                )
            prompt_text = detokenize_result["prompt"]
        else:
            # Use client-side detokenization to be consistent
            prompt_text = tokenizer.decode(final_tokens)
        actual_prompt_len = len(final_tokens)

        prompt_tuples.append((prompt_text, actual_prompt_len, output_len, None))

    logger.info(f"Generated {len(prompt_tuples)} cleaned random prompts")
    return prompt_tuples


async def run_concurrent_requests(
    prompts: List[Tuple[str, int, int, Optional[Dict]]],
    model_name: str,
    api_url: str,
    auth_headers: Dict[str, str],
    max_concurrency: int,
    ignore_eos: bool = True,
) -> List[RequestOutput]:
    """
    Run requests with controlled concurrency using async/await.
    """
    logger.info(
        f"Running {len(prompts)} requests with max concurrency {max_concurrency}"
    )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_request(
        prompt_data: Tuple[str, int, int, Optional[Dict]],
    ) -> RequestOutput:
        prompt_text, prompt_len, output_len, multi_modal_data = prompt_data

        async with semaphore:
            return await async_request_openai_completions(
                prompt=prompt_text,
                prompt_len=prompt_len,
                output_len=output_len,
                model_name=model_name,
                api_url=api_url,
                auth_headers=auth_headers,
                ignore_eos=ignore_eos,
            )

    # Create tasks for all requests
    tasks = [limited_request(prompt_data) for prompt_data in prompts]

    # Execute all requests concurrently
    outputs = await asyncio.gather(*tasks)

    return outputs


def calculate_metrics(
    prompts: List[Tuple[str, int, int, Optional[Dict]]],
    outputs: List[RequestOutput],
    duration_s: float,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    """
    Calculate benchmark metrics matching benchmark_serving.py format.
    """
    actual_output_lens = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls = []
    tpots = []
    all_tpots = []
    ttfts = []
    e2els = []

    for i, output in enumerate(outputs):
        if output.success:
            actual_output_lens.append(output.output_tokens)
            total_input += output.prompt_len

            # Calculate TPOT (Time Per Output Token)
            tpot = 0
            if output.output_tokens > 1:
                latency_minus_ttft = output.latency - output.ttft
                tpot = latency_minus_ttft / (output.output_tokens - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)

            itls.extend(output.itl)
            ttfts.append(output.ttft)
            e2els.append(output.latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    # Calculate goodput if configured
    if goodput_config_dict:
        valid_metrics = []
        slo_values = []

        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )

        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        logger.warning("All requests failed. This may indicate a configuration issue.")

    # Calculate percentiles for each metric
    def calc_percentiles(values, percentiles):
        if not values:
            return [(p, 0.0) for p in percentiles]
        return [(p, np.percentile(values, p)) for p in percentiles]

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / duration_s,
        request_goodput=good_completed / duration_s,
        output_throughput=sum(actual_output_lens) / duration_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / duration_s,
        mean_ttft_ms=np.mean(ttfts or [0]) * 1000,
        std_ttft_ms=np.std(ttfts or [0]) * 1000,
        median_ttft_ms=np.median(ttfts or [0]) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or [0], p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or [0]) * 1000,
        std_tpot_ms=np.std(tpots or [0]) * 1000,
        median_tpot_ms=np.median(tpots or [0]) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or [0], p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or [0]) * 1000,
        std_itl_ms=np.std(itls or [0]) * 1000,
        median_itl_ms=np.median(itls or [0]) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or [0], p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or [0]) * 1000,
        std_e2el_ms=np.std(e2els or [0]) * 1000,
        median_e2el_ms=np.median(e2els or [0]) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or [0], p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def run_benchmark(
    backend: str,
    model_name: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    max_concurrency: int,
    client: PromptClient,
    ignore_eos: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
    host: str,
    port: int,
    seed: int = 0,
    use_server_tokenizer: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete benchmark process and return results matching benchmark_serving.py format.
    """
    logger.info("Starting benchmark run...")

    # Set up API URL and headers
    api_url = f"http://{host}:{port}/v1/completions"
    auth_headers = client.headers

    # Generate cleaned random prompts with configurable tokenization
    prompts = generate_cleaned_random_prompts_using_server(
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        model_name=model_name,
        client=client,
        seed=seed,
        use_server_tokenizer=use_server_tokenizer,
    )

    logger.info("Starting main benchmark run...")

    # Display traffic information
    logger.info("Traffic request rate: inf")
    logger.info("Burstiness factor: 1.0 (Poisson process)")
    logger.info(f"Maximum request concurrency: {max_concurrency}")

    # Run the main benchmark
    start_time = time.perf_counter()
    outputs = await run_concurrent_requests(
        prompts, model_name, api_url, auth_headers, max_concurrency, ignore_eos
    )
    benchmark_duration = time.perf_counter() - start_time

    # Calculate metrics
    metrics, actual_output_lens = calculate_metrics(
        prompts=prompts,
        outputs=outputs,
        duration_s=benchmark_duration,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    # Print results matching benchmark_serving.py format
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    # Build result dictionary matching benchmark_serving.py format
    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput:": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    # Add percentile metrics
    def process_one_metric(
        metric_attribute_name: str, metric_name: str, metric_header: str
    ):
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def parse_goodput(goodput_args: List[str]) -> Dict[str, float]:
    """Parse goodput arguments into a dictionary."""
    goodput_config_dict = {}
    if not goodput_args:
        return goodput_config_dict

    VALID_NAMES = ["ttft", "tpot", "e2el"]
    try:
        for slo_pair in goodput_args:
            slo_name, slo_val = slo_pair.split(":")
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}."
                )
            if float(slo_val) < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be non-negative."
                )
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


async def main():
    parser = argparse.ArgumentParser(
        description="Stress Tests Benchmarking Script - homebrewed replacement for benchmark_serving.py"
    )

    # Core arguments matching benchmark_serving.py
    parser.add_argument(
        "--backend", type=str, default="vllm", help="Backend type (vllm)"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cleaned-random",
        help="Dataset name (only cleaned-random supported)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=8, help="Number of prompts to process"
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=128,
        help="Input sequence length for random prompts",
    )
    parser.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Output sequence length for random prompts",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS tokens to force max output length",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-separated list of metrics to report percentiles",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles to report",
    )
    parser.add_argument(
        "--save-result", action="store_true", help="Save benchmark results to JSON file"
    )
    parser.add_argument(
        "--result-filename", type=str, help="Filename to save benchmark results"
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help="Service level objectives for goodput as KEY:VALUE pairs",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disable trace capture (use when traces already captured)",
    )
    parser.add_argument(
        "--use-server-tokenizer",
        action="store_true",
        help="Use server-side tokenization instead of client-side (default: client-side)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset_name != "cleaned-random":
        raise ValueError("Only 'cleaned-random' dataset is supported by this script")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse percentile arguments
    selected_percentile_metrics = args.percentile_metrics.split(",")
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    goodput_config_dict = parse_goodput(args.goodput or [])

    # Set up environment and client
    env_config = EnvironmentConfig()
    env_config.service_port = str(args.port)
    env_config.deploy_url = f"http://{args.host}"
    env_config.vllm_model = args.model

    # Set JWT token from environment
    auth_token = os.environ.get("OPENAI_API_KEY")
    if not auth_token:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set for cleaned-random dataset"
        )
    env_config.jwt_secret = auth_token  # Set jwt_secret instead of authorization

    client = PromptClient(env_config)

    # Wait for server to be healthy
    logger.info("Checking server health...")
    if not client.wait_for_healthy(timeout=30):
        raise RuntimeError("Server is not healthy")

    # Capture traces if not disabled
    if not args.disable_trace_capture:
        logger.info("Capturing traces...")
        context_lens = [(args.random_input_len, args.random_output_len)]
        client.capture_traces(context_lens=context_lens)

    # Run the benchmark
    try:
        benchmark_result = await run_benchmark(
            backend=args.backend,
            model_name=args.model,
            num_prompts=args.num_prompts,
            input_len=args.random_input_len,
            output_len=args.random_output_len,
            max_concurrency=args.max_concurrency,
            client=client,
            ignore_eos=args.ignore_eos,
            selected_percentile_metrics=selected_percentile_metrics,
            selected_percentiles=selected_percentiles,
            goodput_config_dict=goodput_config_dict,
            host=args.host,
            port=args.port,
            seed=args.seed,
            use_server_tokenizer=args.use_server_tokenizer,
        )
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

    # Save results if requested
    if args.save_result:
        result_json = {}

        # Setup metadata matching benchmark_serving.py format
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = args.backend
        result_json["model_id"] = args.model
        result_json["tokenizer_id"] = args.model
        result_json["num_prompts"] = args.num_prompts

        # Traffic info
        result_json["request_rate"] = "inf"
        result_json["burstiness"] = 1.0
        result_json["max_concurrency"] = args.max_concurrency

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        if args.result_filename:
            file_name = args.result_filename
        else:
            base_model_id = args.model.split("/")[-1]
            file_name = f"vllm-inf-concurrency{args.max_concurrency}-{base_model_id}-{current_dt}.json"

        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(result_json, outfile, indent=2)

        logger.info(f"Results saved to {file_name}")


if __name__ == "__main__":
    asyncio.run(main())
