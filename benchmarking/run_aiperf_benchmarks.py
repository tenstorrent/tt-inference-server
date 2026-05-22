# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

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
from urllib.parse import urlparse

import jwt
import requests

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from benchmarking.prefix_cache_scenarios import (
    PrefixCacheRun,
    build_runs as build_prefix_cache_runs,
    summarize_runs as summarize_prefix_cache_runs,
)
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command
from workflows.workflow_types import DeviceTypes
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

PREFIX_CACHE_HITS_METRIC = "vllm:prefix_cache_hits_total"
PREFIX_CACHE_QUERIES_METRIC = "vllm:prefix_cache_queries_total"
# AIPerf 0.5 strips the canonical Prometheus `_total` suffix when it writes
# server_metrics_export.jsonl. Older AIPerf / raw scrapes keep it. Accept both
# spellings so we work across versions.
PREFIX_CACHE_HITS_METRIC_ALIASES = (
    PREFIX_CACHE_HITS_METRIC,
    "vllm:prefix_cache_hits",
)
PREFIX_CACHE_QUERIES_METRIC_ALIASES = (
    PREFIX_CACHE_QUERIES_METRIC,
    "vllm:prefix_cache_queries",
)


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
            "p95_ttft_ms": summary.get("time_to_first_token", {}).get("p95", 0),
            "p99_ttft_ms": summary.get("time_to_first_token", {}).get("p99", 0),
            "std_ttft_ms": summary.get("time_to_first_token", {}).get("std", 0),
            # TPOT metrics (Time Per Output Token)
            "mean_tpot_ms": summary.get("inter_token_latency", {}).get("avg", 0),
            "median_tpot_ms": summary.get("inter_token_latency", {}).get("p50", 0),
            "p95_tpot_ms": summary.get("inter_token_latency", {}).get("p95", 0),
            "p99_tpot_ms": summary.get("inter_token_latency", {}).get("p99", 0),
            "std_tpot_ms": summary.get("inter_token_latency", {}).get("std", 0),
            # ITL metrics (Inter-Token Latency)
            "mean_itl_ms": summary.get("inter_token_latency", {}).get("avg", 0),
            "median_itl_ms": summary.get("inter_token_latency", {}).get("p50", 0),
            "p95_itl_ms": summary.get("inter_token_latency", {}).get("p95", 0),
            "p99_itl_ms": summary.get("inter_token_latency", {}).get("p99", 0),
            "std_itl_ms": summary.get("inter_token_latency", {}).get("std", 0),
            # E2EL metrics (End-to-End Latency)
            "mean_e2el_ms": summary.get("request_latency", {}).get("avg", 0),
            "median_e2el_ms": summary.get("request_latency", {}).get("p50", 0),
            "p95_e2el_ms": summary.get("request_latency", {}).get("p95", 0),
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

    # Add image metadata if this is a VLM benchmark
    if images > 0:
        result["images_per_prompt"] = images
        result["image_height"] = image_height
        result["image_width"] = image_width
        result["task_type"] = "vlm"

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


def _collect_metric_samples(
    server_metrics_path: Path,
) -> Dict[str, List[float]]:
    """Collect series of `vllm:prefix_cache_*_total` samples from server_metrics_export.jsonl.

    AIPerf writes one JSONL line per Prometheus scrape snapshot. Each snapshot
    is a Prometheus-style dict mapping metric_name -> list of {labels, value}
    or a flat value. We tolerate both shapes by extracting any numeric value
    found beneath the metric key.
    """
    # Track every alias separately; the caller collapses them back into the
    # canonical metric name. This keeps `_extract_numeric` (and any future
    # additions) shape-agnostic.
    series: Dict[str, List[float]] = {
        alias: []
        for alias in (
            *PREFIX_CACHE_HITS_METRIC_ALIASES,
            *PREFIX_CACHE_QUERIES_METRIC_ALIASES,
        )
    }
    if not server_metrics_path.exists():
        return series

    def _extract_numeric(payload) -> Optional[float]:
        if isinstance(payload, (int, float)):
            return float(payload)
        if isinstance(payload, list):
            total = 0.0
            found = False
            for item in payload:
                v = _extract_numeric(item)
                if v is not None:
                    total += v
                    found = True
            return total if found else None
        if isinstance(payload, dict):
            for key in ("value", "val", "total", "sum", "count"):
                if key in payload:
                    v = _extract_numeric(payload[key])
                    if v is not None:
                        return v
            return None
        return None

    try:
        with open(server_metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    snapshot = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # AIPerf 0.5 nests the Prometheus scrape under a top-level
                # "metrics" key (each entry is a list of {labels, value}).
                # Older shapes put metrics at the top level; support both.
                payload = (
                    snapshot["metrics"]
                    if isinstance(snapshot, dict)
                    and isinstance(snapshot.get("metrics"), dict)
                    else snapshot
                )
                for metric_name in series.keys():
                    if metric_name in payload:
                        value = _extract_numeric(payload[metric_name])
                        if value is not None:
                            series[metric_name].append(value)
    except OSError as e:
        logger.warning(
            f"Could not read server-metrics file {server_metrics_path}: {e}"
        )
    return series


def parse_server_metrics_for_prefix_cache(
    artifact_dir: str,
) -> Dict[str, Optional[float]]:
    """Compute the prefix-cache hit rate from AIPerf's Prometheus scrape export.

    Returns a dict with at minimum:
      - prefix_cache_hit_rate: float in [0, 1] or None if not derivable
      - prefix_cache_hits_delta, prefix_cache_queries_delta: ints or None
      - prefix_cache_hits_final, prefix_cache_queries_final: final cumulative counters
    """
    out: Dict[str, Optional[float]] = {
        "prefix_cache_hit_rate": None,
        "prefix_cache_hits_delta": None,
        "prefix_cache_queries_delta": None,
        "prefix_cache_hits_final": None,
        "prefix_cache_queries_final": None,
    }

    candidates = [
        Path(artifact_dir) / "server_metrics_export.jsonl",
    ]
    for sub in glob.glob(os.path.join(artifact_dir, "*")):
        sub_path = Path(sub)
        if sub_path.is_dir():
            candidates.append(sub_path / "server_metrics_export.jsonl")

    server_metrics_path: Optional[Path] = None
    for candidate in candidates:
        if candidate.exists():
            server_metrics_path = candidate
            break

    if server_metrics_path is None:
        logger.warning(
            "server_metrics_export.jsonl not found under "
            f"{artifact_dir}; prefix cache hit-rate will be unavailable. "
            "Check that the vLLM server exposes a Prometheus /metrics endpoint "
            "and that AIPerf's --server-metrics is enabled (default on)."
        )
        return out

    samples = _collect_metric_samples(server_metrics_path)

    def _first_populated(aliases):
        for alias in aliases:
            series = samples.get(alias) or []
            if series:
                return series
        return []

    hits = _first_populated(PREFIX_CACHE_HITS_METRIC_ALIASES)
    queries = _first_populated(PREFIX_CACHE_QUERIES_METRIC_ALIASES)
    if not hits or not queries:
        logger.warning(
            "vLLM prefix cache counters not present in "
            f"{server_metrics_path}. Hit rate unavailable for this run."
        )
        return out

    hits_delta = max(hits[-1] - hits[0], 0.0)
    queries_delta = max(queries[-1] - queries[0], 0.0)
    out["prefix_cache_hits_delta"] = hits_delta
    out["prefix_cache_queries_delta"] = queries_delta
    out["prefix_cache_hits_final"] = hits[-1]
    out["prefix_cache_queries_final"] = queries[-1]
    if queries_delta > 0:
        out["prefix_cache_hit_rate"] = hits_delta / queries_delta
    else:
        # Server reported queries but no delta inside the window - rare but
        # surface 0.0 so it's distinguishable from "metric missing".
        out["prefix_cache_hit_rate"] = 0.0
    return out


def build_aiperf_cmd_for_prefix_cache_run(
    run: "PrefixCacheRun",
    *,
    venv_python: Path,
    model_name: str,
    tokenizer: str,
    url: str,
    artifact_dir: str,
    auth_token: str,
) -> List[str]:
    """Construct the AIPerf CLI command for one prefix-cache run.

    Two modes:

    1. **Synthetic** (``shared_system`` / ``prefix_pool`` / ``multi_turn`` /
       ``baseline``): aiperf generates prompts using ``--synthetic-input-tokens-*``,
       ``--output-tokens-*`` plus a prefix knob
       (``--shared-system-prompt-length`` / ``--num-prefix-prompts``).

    2. **Trace-driven** (``mooncake_trace``, when ``run.uses_trace`` is
       ``True``): aiperf reads a JSONL mooncake trace via
       ``--custom-dataset-type mooncake-trace --input-file <trace>`` and
       optionally scales it with the ``--synthesis-*`` multipliers from
       https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/prefix-synthesis.md
       In this mode the synthetic ISL/OSL flags are intentionally omitted
       (the trace itself supplies sequence lengths).
    """
    if not url.startswith("http"):
        url = f"http://{url}"

    cmd: List[str] = [
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
        str(run.concurrency),
        "--request-count",
        str(run.request_count),
        "--url",
        url,
        "--artifact-dir",
        artifact_dir,
        # Server metrics collection (already on by default but pin it for clarity).
        "--server-metrics-formats",
        "jsonl",
    ]

    if run.uses_trace:
        # Trace-driven mode (mooncake_trace scenario).
        cmd.extend(
            [
                "--custom-dataset-type",
                run.custom_dataset_type or "mooncake_trace",
                "--input-file",
                str(run.trace_input_file),
            ]
        )
        if run.fixed_schedule:
            cmd.extend(["--fixed-schedule", "--fixed-schedule-auto-offset"])
        else:
            # Synthetic arrival pattern only applies when we don't take the
            # trace timestamps verbatim.
            cmd.extend(["--arrival-pattern", run.arrival_pattern])
            if run.arrival_smoothness is not None and run.arrival_pattern == "gamma":
                cmd.extend(["--arrival-smoothness", str(run.arrival_smoothness)])
            if run.request_rate is not None:
                cmd.extend(["--request-rate", str(run.request_rate)])

        # Block size controls how prefix groups are formed in the radix tree.
        if run.block_size is not None:
            cmd.extend(
                ["--prompt-input-tokens-block-size", str(run.block_size)]
            )

        # Synthesis multipliers — all optional, only emit when set.
        if run.synthesis_speedup_ratio is not None:
            cmd.extend(
                ["--synthesis-speedup-ratio", str(run.synthesis_speedup_ratio)]
            )
        if run.synthesis_prefix_len_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prefix-len-multiplier",
                    str(run.synthesis_prefix_len_multiplier),
                ]
            )
        if run.synthesis_prefix_root_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prefix-root-multiplier",
                    str(run.synthesis_prefix_root_multiplier),
                ]
            )
        if run.synthesis_prompt_len_multiplier is not None:
            cmd.extend(
                [
                    "--synthesis-prompt-len-multiplier",
                    str(run.synthesis_prompt_len_multiplier),
                ]
            )
        if run.synthesis_max_isl is not None:
            cmd.extend(["--synthesis-max-isl", str(run.synthesis_max_isl)])
        if run.synthesis_max_osl is not None:
            cmd.extend(["--synthesis-max-osl", str(run.synthesis_max_osl)])
    else:
        # Synthetic mode (shared_system / prefix_pool / multi_turn / baseline).
        cmd.extend(
            [
                "--synthetic-input-tokens-mean",
                str(run.isl_mean),
                "--synthetic-input-tokens-stddev",
                str(run.isl_stddev),
                "--output-tokens-mean",
                str(run.osl_mean),
                "--output-tokens-stddev",
                str(run.osl_stddev),
                "--arrival-pattern",
                run.arrival_pattern,
            ]
        )
        if run.arrival_smoothness is not None and run.arrival_pattern == "gamma":
            cmd.extend(["--arrival-smoothness", str(run.arrival_smoothness)])
        if run.request_rate is not None:
            cmd.extend(["--request-rate", str(run.request_rate)])

        # Prefix knobs (mutually exclusive between shared-system / pool).
        if run.shared_system_prompt_length is not None:
            cmd.extend(
                [
                    "--shared-system-prompt-length",
                    str(run.shared_system_prompt_length),
                ]
            )
        elif run.num_prefix_prompts is not None:
            cmd.extend(
                [
                    "--num-prefix-prompts",
                    str(run.num_prefix_prompts),
                    "--prefix-prompt-length",
                    str(run.prefix_prompt_length or 512),
                ]
            )

        # Multi-turn knobs.
        if run.conversation_num is not None:
            cmd.extend(
                [
                    "--conversation-num",
                    str(run.conversation_num),
                    "--conversation-turn-mean",
                    str(run.conversation_turn_mean or 1),
                    "--conversation-turn-stddev",
                    str(run.conversation_turn_stddev or 0),
                    "--conversation-turn-delay-mean",
                    str(run.conversation_turn_delay_mean_ms or 0),
                ]
            )

    if auth_token:
        cmd.extend(["--api-key", auth_token])
    return cmd


def analyze_trace(
    *,
    trace_path: Path,
    venv_python: Path,
    artifact_base: Path,
    block_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Run ``aiperf analyze-trace`` once for *trace_path* and load the result.

    The analysis JSON is cached under ``artifact_base / "prefix_cache" /
    "trace_analysis" / <trace_stem>.json`` so repeat invocations re-use the
    same on-disk file. Failures are logged and return ``None`` (analysis is
    optional — it only enriches the per-run JSON / report).
    """
    if not trace_path.exists():
        logger.warning(
            "Mooncake trace not found: %s. Skipping analyze-trace.", trace_path
        )
        return None

    analysis_dir = Path(artifact_base) / "prefix_cache" / "trace_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / f"{trace_path.stem}.json"

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            output_path.unlink(missing_ok=True)

    cmd = [
        str(venv_python),
        "-m",
        "aiperf",
        "analyze-trace",
        str(trace_path),
        "--output-file",
        str(output_path),
    ]
    if block_size is not None:
        cmd.extend(["--block-size", str(block_size)])

    logger.info("[prefix-cache] Analyzing trace: %s", " ".join(cmd))
    rc = run_command(command=cmd, logger=logger, env=os.environ.copy())
    if rc != 0 or not output_path.exists():
        logger.warning(
            "aiperf analyze-trace returned %s for %s; continuing without "
            "trace analysis enrichment.",
            rc,
            trace_path,
        )
        return None
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read analyze-trace output %s: %s", output_path, exc)
        return None


def save_prefix_cache_result(
    *,
    run: "PrefixCacheRun",
    metrics: Dict[str, float],
    cache_metrics: Dict[str, Optional[float]],
    model_name: str,
    model_id: str,
    output_dir: str,
    trace_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the per-run JSON for a prefix-cache benchmark.

    Filename pattern:
        aiperf_prefix_cache_<scenario>_<filesafe_label>_<timestamp>.json
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = (
        f"aiperf_prefix_cache_{model_id}_{timestamp}"
        f"_{run.scenario}_{run.filesafe_label()}.json"
    )
    filepath = os.path.join(output_dir, filename)

    payload = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": "aiperf",
        "task_type": "prefix_cache",
        "scenario": run.scenario,
        "label": run.label,
        "model_id": model_name,
        "tokenizer_id": model_name,
        "isl_mean": run.isl_mean,
        "isl_stddev": run.isl_stddev,
        "osl_mean": run.osl_mean,
        "osl_stddev": run.osl_stddev,
        "concurrency": run.concurrency,
        "max_concurrency": run.concurrency,
        "request_count": run.request_count,
        "num_prompts": run.request_count,
        "arrival_pattern": run.arrival_pattern,
        "arrival_smoothness": run.arrival_smoothness,
        "request_rate": run.request_rate,
        "shared_system_prompt_length": run.shared_system_prompt_length,
        "num_prefix_prompts": run.num_prefix_prompts,
        "prefix_prompt_length": run.prefix_prompt_length,
        "conversation_num": run.conversation_num,
        "conversation_turn_mean": run.conversation_turn_mean,
        "conversation_turn_stddev": run.conversation_turn_stddev,
        "conversation_turn_delay_mean_ms": run.conversation_turn_delay_mean_ms,
        # Trace / synthesis provenance (None for synthetic scenarios).
        "trace_input_file": run.trace_input_file,
        "custom_dataset_type": run.custom_dataset_type,
        "fixed_schedule": run.fixed_schedule,
        "block_size": run.block_size,
        "synthesis_speedup_ratio": run.synthesis_speedup_ratio,
        "synthesis_prefix_len_multiplier": run.synthesis_prefix_len_multiplier,
        "synthesis_prefix_root_multiplier": run.synthesis_prefix_root_multiplier,
        "synthesis_prompt_len_multiplier": run.synthesis_prompt_len_multiplier,
        "synthesis_max_isl": run.synthesis_max_isl,
        "synthesis_max_osl": run.synthesis_max_osl,
        "trace_analysis": trace_analysis,
        "metadata": run.metadata,
        **metrics,
        **cache_metrics,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Prefix-cache result saved to: {filepath}")
    return filepath


def run_prefix_cache_benchmark(
    *,
    run: "PrefixCacheRun",
    venv_python: Path,
    model_name: str,
    model_id: str,
    tokenizer: str,
    url: str,
    auth_token: str,
    artifact_base: str,
    output_dir: str,
    trace_analysis: Optional[Dict[str, Any]] = None,
) -> int:
    """Execute a single prefix-cache run and persist its result."""
    artifact_dir = os.path.join(
        artifact_base, "prefix_cache", run.scenario, run.filesafe_label()
    )
    if os.path.exists(artifact_dir):
        shutil.rmtree(artifact_dir)
    os.makedirs(artifact_dir, exist_ok=True)

    cmd = build_aiperf_cmd_for_prefix_cache_run(
        run,
        venv_python=venv_python,
        model_name=model_name,
        tokenizer=tokenizer,
        url=url,
        artifact_dir=artifact_dir,
        auth_token=auth_token,
    )
    if run.uses_trace:
        logger.info(
            f"[prefix-cache] {run.scenario}/{run.label}: "
            f"trace={run.trace_input_file} "
            f"concurrency={run.concurrency} requests={run.request_count} "
            f"synthesis(speedup={run.synthesis_speedup_ratio}, "
            f"prefix_len={run.synthesis_prefix_len_multiplier}, "
            f"prefix_root={run.synthesis_prefix_root_multiplier}, "
            f"prompt_len={run.synthesis_prompt_len_multiplier}) "
            f"fixed_schedule={run.fixed_schedule}"
        )
    else:
        logger.info(
            f"[prefix-cache] {run.scenario}/{run.label}: "
            f"isl_mean={run.isl_mean} osl_mean={run.osl_mean} "
            f"concurrency={run.concurrency} requests={run.request_count} "
            f"arrival={run.arrival_pattern} rate={run.request_rate}"
        )
    logger.info(f"Executing: {' '.join(cmd)}")
    return_code = run_command(command=cmd, logger=logger, env=os.environ.copy())
    if return_code != 0:
        logger.error(
            f"[prefix-cache] AIPerf failed for {run.scenario}/{run.label} "
            f"with return code {return_code}"
        )
        return return_code

    metrics = parse_aiperf_output(artifact_dir)
    cache_metrics = parse_server_metrics_for_prefix_cache(artifact_dir)

    if not metrics:
        logger.error(
            f"[prefix-cache] No metrics parsed from {artifact_dir}; "
            "skipping result save."
        )
        return 1

    # Persist combined result JSON for the report layer.
    save_prefix_cache_result(
        run=run,
        metrics=metrics,
        cache_metrics=cache_metrics,
        model_name=model_name,
        model_id=model_id,
        output_dir=output_dir,
        trace_analysis=trace_analysis,
    )

    # Console summary so the run log shows the cache hit rate immediately.
    hit_rate = cache_metrics.get("prefix_cache_hit_rate")
    hit_rate_str = (
        f"{hit_rate * 100:.2f}%" if isinstance(hit_rate, (int, float)) else "n/a"
    )
    logger.info("=" * 80)
    logger.info(
        f"[prefix-cache] {run.scenario}/{run.label} "
        f"hit_rate={hit_rate_str} "
        f"TTFT mean/p95/p99 = "
        f"{metrics.get('mean_ttft_ms', 0):.1f}/"
        f"{metrics.get('p95_ttft_ms', 0):.1f}/"
        f"{metrics.get('p99_ttft_ms', 0):.1f} ms; "
        f"TPOT mean/p95/p99 = "
        f"{metrics.get('mean_tpot_ms', 0):.1f}/"
        f"{metrics.get('p95_tpot_ms', 0):.1f}/"
        f"{metrics.get('p99_tpot_ms', 0):.1f} ms; "
        f"E2EL mean/p95/p99 = "
        f"{metrics.get('mean_e2el_ms', 0):.1f}/"
        f"{metrics.get('p95_e2el_ms', 0):.1f}/"
        f"{metrics.get('p99_e2el_ms', 0):.1f} ms"
    )
    logger.info("=" * 80)
    return 0


def run_prefix_cache_suite(
    *,
    runtime_config: RuntimeConfig,
    model_spec: ModelSpec,
    prompt_client: PromptClient,
    venv_python: Path,
    auth_token: str,
    artifact_base: Path,
    output_dir: str,
    service_port: str,
) -> int:
    """Plan + execute the full prefix-cache scenario set for a model."""
    manifest_path = (
        Path(runtime_config.prefix_cache_scenarios_json)
        if runtime_config.prefix_cache_scenarios_json
        else None
    )
    runs = build_prefix_cache_runs(
        preset=runtime_config.prefix_cache_preset,
        scenarios=runtime_config.prefix_cache_scenarios,
        arrival_pattern=runtime_config.prefix_cache_arrival,
        request_rate=runtime_config.prefix_cache_request_rate,
        manifest_path=manifest_path,
        trace_path_override=getattr(runtime_config, "prefix_cache_trace", None),
    )
    if not runs:
        logger.error("No prefix-cache runs produced by the selected preset/scenarios.")
        return 1
    logger.info(summarize_prefix_cache_runs(runs))

    # Pre-compute the trace analysis once per unique (trace, block_size) so
    # repeat scenarios using the same trace don't re-run aiperf analyze-trace.
    trace_analyses: Dict[tuple, Optional[Dict[str, Any]]] = {}
    for r in runs:
        if not r.uses_trace:
            continue
        key = (r.trace_input_file, r.block_size)
        if key in trace_analyses:
            continue
        trace_analyses[key] = analyze_trace(
            trace_path=Path(r.trace_input_file),
            venv_python=venv_python,
            artifact_base=Path(artifact_base),
            block_size=r.block_size,
        )

    # Run all baseline scenarios first so subsequent reuse runs benefit from a
    # consistent, warm-but-cold-for-the-cache starting point. Within each
    # group, sort by concurrency ascending for predictable logs.
    # Order: baseline first (so reuse scenarios have a fresh-but-warm
    # starting point), then synthetic reuse scenarios, then trace-driven
    # mooncake_trace runs (largest variance, run last).
    scenario_order = {
        "baseline": 0,
        "shared_system": 1,
        "prefix_pool": 2,
        "multi_turn": 3,
        "mooncake_trace": 4,
    }
    runs_sorted = sorted(
        runs,
        key=lambda r: (
            scenario_order.get(r.scenario, 99),
            r.scenario,
            r.concurrency,
            r.label,
        ),
    )

    return_codes: List[int] = []
    for i, run in enumerate(runs_sorted, 1):
        try:
            health_check = prompt_client.get_health()
        except requests.exceptions.RequestException as error:
            logger.error("Health check request failed: %s", error)
            return 1
        if health_check.status_code != 200:
            logger.error("vLLM server is not healthy. Aborting prefix-cache suite.")
            return 1

        logger.info(
            f"[prefix-cache] Running {i}/{len(runs_sorted)}: "
            f"{run.scenario}/{run.label}"
        )
        time.sleep(2)

        trace_analysis = (
            trace_analyses.get((run.trace_input_file, run.block_size))
            if run.uses_trace
            else None
        )
        rc = run_prefix_cache_benchmark(
            run=run,
            venv_python=venv_python,
            model_name=model_spec.hf_model_repo,
            model_id=model_spec.model_id,
            tokenizer=model_spec.hf_model_repo,
            url=f"localhost:{service_port}",
            auth_token=auth_token,
            artifact_base=str(artifact_base),
            output_dir=output_dir,
            trace_analysis=trace_analysis,
        )
        return_codes.append(rc)

    if all(rc == 0 for rc in return_codes):
        logger.info("Completed AIPerf prefix-cache benchmarks.")
        return 0
    logger.error(
        f"AIPerf prefix-cache benchmarks failed with return codes: {return_codes}. "
        "See logs above for details."
    )
    return 1


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

            # Use the prompt client's URL and auth. Honor an explicit port on
            # env_config.deploy_url to avoid double-port URLs when --server-url
            # already carries one.
            _parsed = urlparse(prompt_client.env_config.deploy_url.rstrip("/"))
            _base = (
                prompt_client.env_config.deploy_url.rstrip("/")
                if _parsed.port is not None
                else f"{prompt_client.env_config.deploy_url.rstrip('/')}:{prompt_client.env_config.service_port}"
            )
            url = f"{_base}/v1/chat/completions"
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


def _run_prefix_cache_mode(
    *,
    runtime_config: RuntimeConfig,
    model_spec: ModelSpec,
    jwt_secret: str,
    auth_token: str,
    service_port: str,
    venv_config,
    output_path: str,
) -> int:
    """Bootstrap a server-healthy AIPerf prefix-cache suite for `model_spec`.

    Mirrors the setup the default sweep path uses (health-check, warm-up,
    artifact directory) and then delegates to :func:`run_prefix_cache_suite`.
    """
    logger.info("=" * 80)
    logger.info(
        f"AIPerf prefix-cache benchmark suite for {model_spec.model_name} "
        f"(preset={runtime_config.prefix_cache_preset})"
    )
    logger.info("=" * 80)

    env_config = EnvironmentConfig()
    env_config.jwt_secret = jwt_secret
    env_config.service_port = service_port
    env_config.vllm_model = model_spec.hf_model_repo

    prompt_client = PromptClient(
        env_config,
        model_spec=model_spec,
        runtime_config=runtime_config,
    )
    if not prompt_client.wait_for_healthy():
        logger.error("vLLM server is not healthy. Aborting prefix-cache suite.")
        return 1

    logger.info("Sending warm-up requests to initialize server...")
    if not send_warmup_requests(prompt_client, model_spec, num_requests=3):
        logger.warning("Warm-up requests failed, but continuing with prefix-cache suite")

    artifact_base = venv_config.venv_path / "artifacts" / model_spec.model_id
    artifact_base.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    return run_prefix_cache_suite(
        runtime_config=runtime_config,
        model_spec=model_spec,
        prompt_client=prompt_client,
        venv_python=venv_config.venv_python,
        auth_token=auth_token,
        artifact_base=artifact_base,
        output_dir=output_path,
        service_port=service_port,
    )


def main():
    """Main entry point for AIPerf benchmarks."""
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    jwt_secret = args.jwt_secret
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    device_str = runtime_config.device
    service_port = runtime_config.service_port
    deploy_url = getattr(runtime_config, "server_url", None) or os.environ.get(
        "DEPLOY_URL", "http://127.0.0.1"
    )
    # An explicit port on deploy_url wins over service_port so AIPerf doesn't
    # connect to host:service_port when --server-url already carries a port.
    _parsed = urlparse(deploy_url.rstrip("/"))
    aiperf_host = _parsed.hostname or "localhost"
    aiperf_port = str(_parsed.port) if _parsed.port is not None else str(service_port)
    aiperf_url = f"{aiperf_host}:{aiperf_port}"

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

    # Branch for the prefix-caching scenario suite. We skip the default
    # benchmark sweep entirely and run the dedicated scenario set instead.
    if runtime_config.prefix_cache:
        return _run_prefix_cache_mode(
            runtime_config=runtime_config,
            model_spec=model_spec,
            jwt_secret=jwt_secret,
            auth_token=auth_token,
            service_port=service_port,
            venv_config=venv_config,
            output_path=args.output_path,
        )

    # Look up the benchmark configuration for the model
    if model_spec.model_id not in BENCHMARK_CONFIGS:
        message = f"No benchmark tasks defined for model: {model_spec.model_name}"
        raise ValueError(message)
    benchmark_config = BENCHMARK_CONFIGS[model_spec.model_id]

    # Get all benchmark params for this device
    all_params = [
        param
        for task in benchmark_config.tasks
        if device in task.param_map
        for param in task.param_map[device]
    ]

    if not all_params:
        message = f"No benchmark tasks defined for model: {model_spec.model_name} on device: {device.name}"
        raise ValueError(message)

    # Check for limit_samples_mode (smoke-test, ci-commit) to enable debug mode
    limit_samples_mode_str = runtime_config.limit_samples_mode
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
        if param.task_type == "vlm":
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
    env_config.deploy_url = deploy_url

    prompt_client = PromptClient(
        env_config,
        model_spec=model_spec,
        runtime_config=runtime_config,
    )
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
        try:
            health_check = prompt_client.get_health()
        except requests.exceptions.RequestException as error:
            logger.error("Health check request failed: %s", error)
            return 1
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
        if params.task_type == "vlm":
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
            url=aiperf_url,
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
