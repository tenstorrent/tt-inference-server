# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# genai_benchmark.py - In-container script for running genai-perf benchmarks
# This script runs inside the nvcr.io/nvidia/tritonserver container
# and uses the genai-perf CLI to profile LLM inference servers.

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# --- Default Configuration ---
DEFAULT_BATCH_1_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (1024, 128),
    (2048, 128),
    (3072, 128),
    (4096, 128),
    (8192, 128),
    (16384, 128),
    (32000, 128),
]

DEFAULT_MAX_CONCURRENCY_ISL_OSL_PAIRS = [
    (128, 128),
    (128, 1024),
    (2048, 128),
    (2048, 2048),
    (3000, 64),
    (4000, 64),
    (8000, 64),
    (16000, 64),
    (32000, 64),
]

# VLM/Image benchmark configurations (ISL, OSL, images, height, width)
DEFAULT_VLM_BATCH_1_CONFIGS = [
    (128, 128, 1, 896, 896),  # Large square
    (128, 128, 1, 512, 512),  # Small square
    (128, 128, 1, 1024, 1024),  # Extra large square
    (128, 128, 1, 512, 1024),  # Portrait
    (128, 128, 1, 1024, 512),  # Landscape
]

DEFAULT_VLM_MAX_CONCURRENCY_CONFIGS = [
    (128, 128, 1, 512, 512),  # Small square
    (128, 128, 1, 1024, 1024),  # Large square
    (128, 128, 1, 512, 1024),  # Portrait
    (128, 128, 1, 1024, 512),  # Landscape
]

DEFAULT_TOKENIZER = "hf-internal-testing/llama-tokenizer"


@dataclass
class BenchmarkResult:
    isl: int
    osl: int
    concurrency: int
    num_requests: int
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_tps: float = 0.0
    p99_latency: float = 0.0
    error: str = ""
    # VLM/image parameters
    images: int = 0
    image_height: int = 0
    image_width: int = 0


class ResultsAggregator:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.start_time = time.time()

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def print_summary_table(self, print_runtime=True):
        print("\n" + "=" * 100)
        print(f"{'FINAL BENCHMARK SUMMARY':^100}")
        print("=" * 100)

        header = f"{'ISL':<8} | {'OSL':<8} | {'Concur':<8} | {'Reqs':<6} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'TPS':<10} | {'P99(ms)':<10} | {'Status'}"
        print(header)
        print("-" * 100)

        for r in self.results:
            status = "OK" if not r.error else "FAIL"
            ttft = f"{r.avg_ttft_ms:.2f}" if r.avg_ttft_ms else "N/A"
            tpot = f"{r.avg_tpot_ms:.2f}" if r.avg_tpot_ms else "N/A"
            tps = f"{r.avg_tps:.2f}" if r.avg_tps else "N/A"
            p99 = f"{r.p99_latency:.2f}" if r.p99_latency else "N/A"

            row = f"{r.isl:<8} | {r.osl:<8} | {r.concurrency:<8} | {r.num_requests:<6} | {ttft:<10} | {tpot:<10} | {tps:<10} | {p99:<10} | {status}"
            print(row)

        print("-" * 100)
        if print_runtime:
            total_duration = time.time() - self.start_time
            print(f"Total Runtime: {total_duration:.2f} seconds")
        print("=" * 100 + "\n")

    def to_json(self) -> List[Dict[str, Any]]:
        """Export results as JSON-serializable list."""
        return [
            {
                "isl": r.isl,
                "osl": r.osl,
                "concurrency": r.concurrency,
                "num_requests": r.num_requests,
                "avg_ttft_ms": r.avg_ttft_ms,
                "avg_tpot_ms": r.avg_tpot_ms,
                "avg_tps": r.avg_tps,
                "p99_latency": r.p99_latency,
                "error": r.error,
            }
            for r in self.results
        ]


def get_num_prompts(input_len, output_len, max_concurrency):
    if output_len > 1024 or input_len > 4000:
        return 2 * max_concurrency
    if (output_len > 128 and output_len <= 1024) or (
        input_len > 128 and input_len <= 4000
    ):
        return 4 * max_concurrency
    if output_len <= 128:
        return 8 * max_concurrency
    raise ValueError(f"Invalid output_len: {output_len}")


def get_benchmark_max_concurrency(isl, osl, max_context, model_max_concurrency=32):
    total_seq_len = isl + osl
    if total_seq_len > max_context:
        return 1
    max_concurrency_by_context = max_context // total_seq_len
    return min(max_concurrency_by_context, model_max_concurrency)


def parse_metrics(artifact_dir) -> Dict[str, float]:
    """Parse full metrics from profile_export_genai_perf.json with percentile data."""

    # Find the JSON file in nested directory structure
    json_path = None
    for subdir in glob.glob(os.path.join(artifact_dir, "*")):
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, "profile_export_genai_perf.json")
            if os.path.exists(candidate):
                json_path = candidate
                break

    if not json_path:
        print(f" [Error: JSON not found in {artifact_dir}]", end="")
        return {}

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract metrics with vLLM-compatible naming
        metrics = {
            # TTFT metrics (Time To First Token)
            "mean_ttft_ms": data["time_to_first_token"]["avg"],
            "median_ttft_ms": data["time_to_first_token"]["p50"],
            "p99_ttft_ms": data["time_to_first_token"]["p99"],
            "p95_ttft_ms": data["time_to_first_token"]["p95"],
            "p90_ttft_ms": data["time_to_first_token"]["p90"],
            "p75_ttft_ms": data["time_to_first_token"]["p75"],
            "std_ttft_ms": data["time_to_first_token"]["std"],
            # TPOT metrics (Time Per Output Token = Inter Token Latency)
            "mean_tpot_ms": data["inter_token_latency"]["avg"],
            "median_tpot_ms": data["inter_token_latency"]["p50"],
            "p99_tpot_ms": data["inter_token_latency"]["p99"],
            "p95_tpot_ms": data["inter_token_latency"]["p95"],
            "p90_tpot_ms": data["inter_token_latency"]["p90"],
            "p75_tpot_ms": data["inter_token_latency"]["p75"],
            "std_tpot_ms": data["inter_token_latency"]["std"],
            # ITL metrics (Inter-Token Latency, same as TPOT for genai-perf)
            "mean_itl_ms": data["inter_token_latency"]["avg"],
            "median_itl_ms": data["inter_token_latency"]["p50"],
            "p99_itl_ms": data["inter_token_latency"]["p99"],
            "std_itl_ms": data["inter_token_latency"]["std"],
            # E2EL metrics (End-to-End Latency = Request Latency)
            "mean_e2el_ms": data["request_latency"]["avg"],
            "median_e2el_ms": data["request_latency"]["p50"],
            "p99_e2el_ms": data["request_latency"]["p99"],
            "std_e2el_ms": data["request_latency"]["std"],
            # Throughput metrics
            "output_token_throughput": data["output_token_throughput"]["avg"],
            "total_token_throughput": data.get("total_token_throughput", {}).get(
                "avg", "N/A"
            ),
            "request_throughput": data["request_throughput"]["avg"],
            # Calculated token counts (approximations)
            "total_input_tokens": int(
                data["input_sequence_length"]["avg"] * data["request_count"]["avg"]
            ),
            "total_output_tokens": int(
                data["output_sequence_length"]["avg"] * data["request_count"]["avg"]
            ),
            # Request count
            "completed": int(data["request_count"]["avg"]),
        }

        return metrics

    except Exception as e:
        print(f" [Error parsing JSON: {e}]", end="")
        return {}


def print_detailed_results(isl, osl, concurrency, num_requests, metrics):
    """Print detailed benchmark results matching vLLM format"""

    print("\n" + "=" * 80)
    print(f"Serving Benchmark Result: ISL={isl}, OSL={osl}, Concurrency={concurrency}")
    print("=" * 80)

    # Summary section
    print(
        f"\nSuccessful requests:                     {metrics.get('completed', num_requests)}"
    )
    print(
        f"Request throughput (req/s):              {metrics.get('request_throughput', 0):.2f}"
    )
    print(
        f"Output token throughput (tok/s):         {metrics.get('output_token_throughput', 0):.2f}"
    )
    total_token_tput = metrics.get("total_token_throughput", 0)
    if total_token_tput == "N/A":
        print("Total token throughput (tok/s):          N/A")
    else:
        print(f"Total token throughput (tok/s):          {total_token_tput:.2f}")
    print(
        f"Total input tokens:                      {metrics.get('total_input_tokens', 0)}"
    )
    print(
        f"Total output tokens:                     {metrics.get('total_output_tokens', 0)}"
    )

    # TTFT section
    print(f"\n{'-' * 40}Time to First Token{'-' * 40}")
    print(
        f"Mean TTFT (ms):                          {metrics.get('mean_ttft_ms', 0):.2f}"
    )
    print(
        f"Median TTFT (ms):                        {metrics.get('median_ttft_ms', 0):.2f}"
    )
    print(
        f"P99 TTFT (ms):                           {metrics.get('p99_ttft_ms', 0):.2f}"
    )
    print(
        f"P95 TTFT (ms):                           {metrics.get('p95_ttft_ms', 0):.2f}"
    )
    print(
        f"P90 TTFT (ms):                           {metrics.get('p90_ttft_ms', 0):.2f}"
    )
    print(
        f"P75 TTFT (ms):                           {metrics.get('p75_ttft_ms', 0):.2f}"
    )
    print(
        f"Std TTFT (ms):                           {metrics.get('std_ttft_ms', 0):.2f}"
    )

    # TPOT section
    print(f"\n{'-' * 40}Time per Output Token (excl. 1st token){'-' * 40}")
    print(
        f"Mean TPOT (ms):                          {metrics.get('mean_tpot_ms', 0):.2f}"
    )
    print(
        f"Median TPOT (ms):                        {metrics.get('median_tpot_ms', 0):.2f}"
    )
    print(
        f"P99 TPOT (ms):                           {metrics.get('p99_tpot_ms', 0):.2f}"
    )
    print(
        f"P95 TPOT (ms):                           {metrics.get('p95_tpot_ms', 0):.2f}"
    )
    print(
        f"P90 TPOT (ms):                           {metrics.get('p90_tpot_ms', 0):.2f}"
    )
    print(
        f"P75 TPOT (ms):                           {metrics.get('p75_tpot_ms', 0):.2f}"
    )
    print(
        f"Std TPOT (ms):                           {metrics.get('std_tpot_ms', 0):.2f}"
    )

    # ITL section
    print(f"\n{'-' * 40}Inter-token Latency{'-' * 40}")
    print(
        f"Mean ITL (ms):                           {metrics.get('mean_itl_ms', 0):.2f}"
    )
    print(
        f"Median ITL (ms):                         {metrics.get('median_itl_ms', 0):.2f}"
    )
    print(
        f"P99 ITL (ms):                            {metrics.get('p99_itl_ms', 0):.2f}"
    )
    print(
        f"Std ITL (ms):                            {metrics.get('std_itl_ms', 0):.2f}"
    )

    # E2EL section
    print(f"\n{'-' * 40}End-to-end Latency{'-' * 40}")
    print(
        f"Mean E2EL (ms):                          {metrics.get('mean_e2el_ms', 0):.2f}"
    )
    print(
        f"Median E2EL (ms):                        {metrics.get('median_e2el_ms', 0):.2f}"
    )
    print(
        f"P99 E2EL (ms):                           {metrics.get('p99_e2el_ms', 0):.2f}"
    )
    print(
        f"Std E2EL (ms):                           {metrics.get('std_e2el_ms', 0):.2f}"
    )

    print("=" * 80 + "\n")


def save_individual_result(
    metrics,
    isl,
    osl,
    concurrency,
    num_requests,
    model_name,
    model_id,
    output_dir,
    images=0,
    image_height=0,
    image_width=0,
):
    """Save individual benchmark result in genai-perf format"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use genai_ prefix to differentiate from vLLM benchmarks
    if images > 0:
        # VLM benchmark - include image parameters in filename
        filename = f"genai_benchmark_{model_id}_{timestamp}_isl-{isl}_osl-{osl}_maxcon-{concurrency}_n-{num_requests}_images-{images}_height-{image_height}_width-{image_width}.json"
    else:
        # Text-only benchmark
        filename = f"genai_benchmark_{model_id}_{timestamp}_isl-{isl}_osl-{osl}_maxcon-{concurrency}_n-{num_requests}.json"

    filepath = os.path.join(output_dir, filename)

    # Build vLLM-compatible JSON structure
    result = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "genai-perf",
        "model_id": model_name,
        "tokenizer_id": model_name,
        "num_prompts": num_requests,
        "max_concurrency": concurrency,
        **metrics,  # All harmonized metrics
    }

    # Add image metadata if this is a VLM benchmark
    if images > 0:
        result["images_per_prompt"] = images
        result["image_height"] = image_height
        result["image_width"] = image_width
        result["task_type"] = "vlm"

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result saved to: {filepath}")
    return filepath


def run_benchmark(
    isl,
    osl,
    concurrency,
    aggregator,
    model_name,
    model_id,
    tokenizer,
    url,
    auth_token,
    artifact_base,
    verbose=False,
    images=0,
    image_height=0,
    image_width=0,
):
    """Run a single GenAI-Perf benchmark.

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

    if verbose:
        print(f"\n{'=' * 80}")
        if images > 0:
            print(
                f"BENCHMARK: ISL={isl}, OSL={osl}, Concurrency={concurrency}, Requests={num_prompts}, Images={images}, Size={image_width}x{image_height}"
            )
        else:
            print(
                f"BENCHMARK: ISL={isl}, OSL={osl}, Concurrency={concurrency}, Requests={num_prompts}"
            )
        print(f"Artifact Dir: {artifact_dir}")
        print(f"{'=' * 80}")
    else:
        if images > 0:
            print(
                f"\nRunning: ISL={isl}, OSL={osl}, Concur={concurrency}, Images={images}, Size={image_width}x{image_height} ... ",
                end="",
                flush=True,
            )
        else:
            print(
                f"\nRunning: ISL={isl}, OSL={osl}, Concur={concurrency} ... ",
                end="",
                flush=True,
            )

    if not auth_token:
        print("FAILED (No AUTH_TOKEN)")
        aggregator.add_result(
            BenchmarkResult(
                isl,
                osl,
                concurrency,
                num_prompts,
                error="No Token",
                images=images,
                image_height=image_height,
                image_width=image_width,
            )
        )
        return

    # Clean up previous artifact dir
    if os.path.exists(artifact_dir):
        import shutil

        shutil.rmtree(artifact_dir)

    # Choose endpoint type based on whether this is VLM or text-only
    endpoint_type = "vision" if images > 0 else "chat"

    cmd = [
        "genai-perf",
        "profile",
        "-m",
        model_name,
        "--tokenizer",
        tokenizer,
        "--endpoint-type",
        endpoint_type,
        "--streaming",
        "--warmup-request-count",
        "0",
        "--concurrency",
        str(concurrency),
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--url",
        f"http://{url}" if not url.startswith("http") else url,
        "--request-count",
        str(num_prompts),
        "--artifact-dir",
        artifact_dir,
        "--num-dataset-entries",
        "96",
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "temperature:0.0",
        "--extra-inputs",
        "top_p:1.0",
    ]

    # Add image parameters for VLM benchmarks
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
                "--image-format",
                "png",
            ]
        )

    # Add auth header
    cmd.extend(
        [
            "--",
            "-H",
            f"Authorization: Bearer {auth_token}",
            "--stability-percentage",
            "999",
            "--max-trials",
            "1",
        ]
    )

    if verbose:
        print("\nExecuting command:")
        print(f"  {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        if verbose:
            if result.stdout:
                print("\n--- STDOUT ---")
                print(result.stdout)
            if result.stderr:
                print("\n--- STDERR ---")
                print(result.stderr)

        stats = parse_metrics(artifact_dir)

        if stats:
            # Print detailed results to terminal
            print_detailed_results(isl, osl, concurrency, num_prompts, stats)

            # Save individual result file
            # Get output directory from environment or config
            benchmarks_output_dir = os.environ.get(
                "BENCHMARKS_OUTPUT_DIR", "/workspace/benchmarks_output"
            )
            if not os.path.exists(benchmarks_output_dir):
                os.makedirs(benchmarks_output_dir, exist_ok=True)

            save_individual_result(
                stats,
                isl,
                osl,
                concurrency,
                num_prompts,
                model_name,
                model_id,
                benchmarks_output_dir,
                images=images,
                image_height=image_height,
                image_width=image_width,
            )

            # Still add to aggregator for final summary table
            bench_res = BenchmarkResult(
                isl=isl,
                osl=osl,
                concurrency=concurrency,
                num_requests=num_prompts,
                avg_ttft_ms=stats.get("mean_ttft_ms", 0),
                avg_tpot_ms=stats.get("mean_tpot_ms", 0),
                avg_tps=stats.get("output_token_throughput", 0),
                p99_latency=stats.get("p99_itl_ms", 0),
                images=images,
                image_height=image_height,
                image_width=image_width,
            )
            aggregator.add_result(bench_res)
            if verbose:
                print("\n[OK] BENCHMARK COMPLETED SUCCESSFULLY")
                sys.stdout.flush()
            else:
                print("DONE")
                sys.stdout.flush()
        else:
            if verbose:
                print("\n[FAIL] BENCHMARK FAILED: Empty Results")
                if result.stderr:
                    print(f"\nStderr output:\n{result.stderr}")
                sys.stdout.flush()
            else:
                print("FAILED (Empty Results)")
                sys.stdout.flush()
            aggregator.add_result(
                BenchmarkResult(
                    isl,
                    osl,
                    concurrency,
                    num_prompts,
                    error="Empty Results",
                    images=images,
                    image_height=image_height,
                    image_width=image_width,
                )
            )

    except subprocess.CalledProcessError as e:
        if verbose:
            print(
                f"\n[FAIL] BENCHMARK FAILED: Process Error (exit code {e.returncode})"
            )
            if e.stdout:
                print("\n--- STDOUT ---")
                print(e.stdout)
            if e.stderr:
                print("\n--- STDERR ---")
                print(e.stderr)
            sys.stdout.flush()
        else:
            print("FAILED")
            sys.stdout.flush()
        error_msg = f"Process Error (code {e.returncode})"
        aggregator.add_result(
            BenchmarkResult(
                isl,
                osl,
                concurrency,
                num_prompts,
                error=error_msg,
                images=images,
                image_height=image_height,
                image_width=image_width,
            )
        )
    except KeyboardInterrupt:
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run genai-perf benchmarks")
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print existing benchmark results without running new tests",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run only 2 benchmarks (1 small, 1 medium) with verbose logging",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        help="Path to JSON config file with benchmark parameters",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to output JSON file for results",
    )
    return parser.parse_args()


def load_config_from_env():
    """Load configuration from environment variables."""
    return {
        "model_name": os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"),
        "model_id": os.environ.get("MODEL_ID", ""),
        "tokenizer": os.environ.get("TOKENIZER", DEFAULT_TOKENIZER),
        "url": os.environ.get("URL", "localhost:8000"),
        "max_context": int(os.environ.get("MAX_CONTEXT", "131072")),
        "model_max_concurrency": int(os.environ.get("MODEL_MAX_CONCURRENCY", "32")),
        "auth_token": os.environ.get("AUTH_TOKEN", ""),
        "artifact_base": os.environ.get("ARTIFACT_BASE", "/workspace/artifacts"),
    }


def main():
    args = parse_args()
    aggregator = ResultsAggregator()

    # Load config from environment or file
    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, "r") as f:
            config = json.load(f)
    else:
        config = load_config_from_env()

    model_name = config["model_name"]
    model_id = config.get(
        "model_id", model_name.replace("/", "__")
    )  # Fallback if not provided
    tokenizer = config["tokenizer"]
    url = config["url"]
    max_context = config["max_context"]
    model_max_concurrency = config["model_max_concurrency"]
    auth_token = config["auth_token"]
    artifact_base = config["artifact_base"]

    # Use custom ISL/OSL pairs from config if provided
    batch_1_pairs = config.get("batch_1_isl_osl_pairs", DEFAULT_BATCH_1_ISL_OSL_PAIRS)
    max_concurrency_pairs = config.get(
        "max_concurrency_isl_osl_pairs", DEFAULT_MAX_CONCURRENCY_ISL_OSL_PAIRS
    )

    # Use custom VLM configs from config if provided
    vlm_batch_1_configs = config.get("vlm_batch_1_configs", DEFAULT_VLM_BATCH_1_CONFIGS)
    vlm_max_concurrency_configs = config.get(
        "vlm_max_concurrency_configs", DEFAULT_VLM_MAX_CONCURRENCY_CONFIGS
    )

    print(f"Model: {model_name}")
    print(f"URL: {url}")
    print(f"Max Context: {max_context}")
    print(f"Model Max Concurrency: {model_max_concurrency}")
    print(f"Artifact Base: {artifact_base}")

    if args.debug:
        print("=" * 80)
        print("DEBUG MODE: Running 2 benchmarks with verbose logging")
        print("=" * 80)

        # Small text benchmark
        print("\n[1/2] Small text benchmark: ISL=128, OSL=128, Concurrency=1")
        run_benchmark(
            128,
            128,
            1,
            aggregator,
            model_name,
            model_id,
            tokenizer,
            url,
            auth_token,
            artifact_base,
            verbose=True,
        )

        # Medium text benchmark
        print("\n[2/2] Medium text benchmark: ISL=2048, OSL=128")
        concurrency = get_benchmark_max_concurrency(
            2048, 128, max_context, model_max_concurrency
        )
        print(f"      Calculated concurrency: {concurrency}")
        run_benchmark(
            2048,
            128,
            concurrency,
            aggregator,
            model_name,
            model_id,
            tokenizer,
            url,
            auth_token,
            artifact_base,
            verbose=True,
        )

        aggregator.print_summary_table()

    else:
        print("Starting Text Benchmarks...")

        # 1. Text Batch 1 Sweeps
        for isl, osl in batch_1_pairs:
            run_benchmark(
                isl,
                osl,
                1,
                aggregator,
                model_name,
                model_id,
                tokenizer,
                url,
                auth_token,
                artifact_base,
            )

        # 2. Text Max Concurrency Sweeps
        for isl, osl in max_concurrency_pairs:
            concurrency = get_benchmark_max_concurrency(
                isl, osl, max_context, model_max_concurrency
            )
            if concurrency > 1:
                run_benchmark(
                    isl,
                    osl,
                    concurrency,
                    aggregator,
                    model_name,
                    model_id,
                    tokenizer,
                    url,
                    auth_token,
                    artifact_base,
                )

        print("\nStarting VLM/Image Benchmarks...")

        # 3. VLM Batch 1 Sweeps (low concurrency)
        for isl, osl, images, height, width in vlm_batch_1_configs:
            run_benchmark(
                isl,
                osl,
                1,  # Always concurrency 1 for VLM batch 1
                aggregator,
                model_name,
                model_id,
                tokenizer,
                url,
                auth_token,
                artifact_base,
                images=images,
                image_height=height,
                image_width=width,
            )

        # 4. VLM Max Concurrency Sweeps
        for isl, osl, images, height, width in vlm_max_concurrency_configs:
            concurrency = get_benchmark_max_concurrency(
                isl, osl, max_context, model_max_concurrency
            )
            if concurrency > 1:
                run_benchmark(
                    isl,
                    osl,
                    concurrency,
                    aggregator,
                    model_name,
                    model_id,
                    tokenizer,
                    url,
                    auth_token,
                    artifact_base,
                    images=images,
                    image_height=height,
                    image_width=width,
                )

        aggregator.print_summary_table()

    # NOTE: Individual result files are already saved per run
    # No longer need aggregated JSON output
    # if args.output_json:
    #     with open(args.output_json, "w") as f:
    #         json.dump(aggregator.to_json(), f, indent=2)
    #     print(f"Results saved to: {args.output_json}")

    # Return non-zero if any benchmarks failed
    failed = any(r.error for r in aggregator.results)
    return 1 if failed else 0


if __name__ == "__main__":
    exit(main())
