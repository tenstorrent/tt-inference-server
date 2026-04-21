# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Shared parsing utilities for report_module strategies."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from report_module.types import NOT_MEASURED_STR

logger = logging.getLogger(__name__)

BENCHMARK_SIG_DIGITS_MAP: Dict[str, int] = {
    "mean_ttft_ms": 1,
    "mean_tpot_ms": 1,
    "mean_tps": 2,
    "mean_e2el_ms": 1,
    "tps_decode_throughput": 1,
    "tps_prefill_throughput": 1,
    "request_throughput": 3,
    "total_token_throughput": 2,
}

STRESS_TEST_SIG_DIGITS_MAP: Dict[str, int] = {
    **BENCHMARK_SIG_DIGITS_MAP,
    "mean_itl_ms": 1,
    "p5_ttft_ms": 1,
    "p25_ttft_ms": 1,
    "p50_ttft_ms": 1,
    "p95_ttft_ms": 1,
    "p99_ttft_ms": 1,
    "p5_tpot_ms": 1,
    "p25_tpot_ms": 1,
    "p50_tpot_ms": 1,
    "p95_tpot_ms": 1,
    "p99_tpot_ms": 1,
    "p5_itl_ms": 1,
    "p25_itl_ms": 1,
    "p50_itl_ms": 1,
    "p95_itl_ms": 1,
    "p99_itl_ms": 1,
    "p5_e2el_ms": 1,
    "p25_e2el_ms": 1,
    "p50_e2el_ms": 1,
    "p95_e2el_ms": 1,
    "p99_e2el_ms": 1,
}

CONFIG_PARAM_REGEX = re.compile(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)")
IMAGE_PARAM_REGEX = re.compile(r"images-(\d+)_height-(\d+)_width-(\d+)")


def format_metrics(
    metrics: Dict[str, Any],
    sig_digits: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Round floats and replace None with NOT_MEASURED_STR."""
    sig_map = sig_digits or BENCHMARK_SIG_DIGITS_MAP
    formatted: Dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None or value == NOT_MEASURED_STR:
            formatted[key] = NOT_MEASURED_STR
        elif isinstance(value, float):
            formatted[key] = round(value, sig_map.get(key, 2))
        else:
            formatted[key] = value
    return formatted


def deduplicate_by_config(files: List[str]) -> List[str]:
    """Keep only the latest file per unique benchmark configuration."""
    config_to_file: Dict = {}
    for filepath in sorted(files, reverse=True):
        filename = Path(filepath).name
        match = CONFIG_PARAM_REGEX.search(filename)
        if match:
            isl, osl, con, n = map(int, match.groups())
            img_match = IMAGE_PARAM_REGEX.search(filename)
            if img_match:
                images, height, width = map(int, img_match.groups())
                config_key = (isl, osl, con, n, images, height, width)
            else:
                config_key = (isl, osl, con, n, 0, 0, 0)
            if config_key not in config_to_file:
                config_to_file[config_key] = filepath
        else:
            config_to_file[filepath] = filepath
    return list(config_to_file.values())


def parse_benchmark_params_from_filename(filename: str) -> Dict[str, Any]:
    """Extract ISL/OSL/concurrency/N and optional image params from a benchmark filename.

    Returns a dict with keys: isl, osl, concurrency, num_requests, and optionally
    images, image_height, image_width.  Returns None values when the filename
    doesn't match the standard pattern (caller must handle fallback).
    """
    match = CONFIG_PARAM_REGEX.search(filename)
    if not match:
        return {}
    isl, osl, concurrency, num_requests = map(int, match.groups())
    params: Dict[str, Any] = {
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "num_requests": num_requests,
    }
    img_match = IMAGE_PARAM_REGEX.search(filename)
    if img_match:
        images, height, width = map(int, img_match.groups())
        params.update(images=images, image_height=height, image_width=width)
    return params


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load and return a JSON file's contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_percentile_result(
    filepath: str, source: str, is_vlm: bool = False
) -> Dict[str, Any] | None:
    """Parse a single benchmark JSON into the percentile result dict used
    by AiPerf/GenAiPerf strategies.

    Returns None if the file cannot be parsed.
    """
    try:
        data = load_json_file(filepath)
        filename = Path(filepath).name
        params = parse_benchmark_params_from_filename(filename)
        if not params:
            data_prompts = max(data.get("num_prompts", 1), 1)
            params = {
                "isl": data.get("total_input_tokens", 0) // data_prompts,
                "osl": data.get("total_output_tokens", 0) // data_prompts,
                "concurrency": data.get("max_concurrency", 1),
                "num_requests": data.get("num_prompts", 0),
            }

        result: Dict[str, Any] = {
            "source": source,
            "isl": params["isl"],
            "osl": params["osl"],
            "concurrency": params["concurrency"],
            "num_requests": params["num_requests"],
            "mean_ttft_ms": data.get("mean_ttft_ms", 0),
            "median_ttft_ms": data.get("median_ttft_ms", 0),
            "p99_ttft_ms": data.get("p99_ttft_ms", 0),
            "std_ttft_ms": data.get("std_ttft_ms", 0),
            "mean_tpot_ms": data.get("mean_tpot_ms", 0),
            "median_tpot_ms": data.get("median_tpot_ms", 0),
            "p99_tpot_ms": data.get("p99_tpot_ms", 0),
            "std_tpot_ms": data.get("std_tpot_ms", 0),
            "mean_e2el_ms": data.get("mean_e2el_ms", 0),
            "median_e2el_ms": data.get("median_e2el_ms", 0),
            "p99_e2el_ms": data.get("p99_e2el_ms", 0),
            "std_e2el_ms": data.get("std_e2el_ms", 0),
            "output_token_throughput": data.get(
                "output_token_throughput", data.get("output_throughput", 0)
            ),
            "total_token_throughput": data.get("total_token_throughput", 0),
            "request_throughput": data.get("request_throughput", 0),
            "completed": data.get("completed", 0),
            "total_input_tokens": data.get("total_input_tokens", 0),
            "total_output_tokens": data.get("total_output_tokens", 0),
            "model_id": data.get("model_id", ""),
            "backend": source,
        }

        if is_vlm:
            if "images" not in params:
                logger.warning(
                    f"Could not parse image parameters from {filename}, skipping"
                )
                return None
            result.update(
                images=params["images"],
                images_per_prompt=params["images"],
                image_height=params["image_height"],
                image_width=params["image_width"],
            )
            if source == "vLLM":
                mean_tpot = data.get("mean_tpot_ms", 0)
                mean_tps = 1000.0 / mean_tpot if mean_tpot and mean_tpot > 0 else 0
                actual_con = min(params["concurrency"], params["num_requests"])
                result["mean_tps"] = mean_tps
                result["tps_decode_throughput"] = mean_tps * actual_con
                result["task_type"] = "vlm"
                result["max_con"] = params["concurrency"]
                result["backend"] = data.get("backend", "vllm")

        return result
    except Exception as e:
        logger.warning(f"Error processing {filepath}: {e}")
        return None
