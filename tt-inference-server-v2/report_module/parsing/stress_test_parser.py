# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Stress test JSON file parsing — extracted from stress_tests/stress_tests_summary_report.py."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

from report_module.parsing.common import STRESS_TEST_SIG_DIGITS_MAP, format_metrics

logger = logging.getLogger(__name__)

_STRESS_IMAGE_PATTERN = re.compile(
    r"^stress_test_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"_images-(?P<images_per_prompt>\d+)_height-(?P<image_height>\d+)_width-(?P<image_width>\d+)"
    r"\.json$"
)

_STRESS_TEXT_PATTERN = re.compile(
    r"^stress_test_"
    r"(?P<model>.+?)"
    r"(?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?"
    r"_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"_isl-(?P<isl>\d+)_osl-(?P<osl>\d+)_maxcon-(?P<maxcon>\d+)_n-(?P<n>\d+)"
    r"\.json$"
)


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    match = _STRESS_IMAGE_PATTERN.search(filename)
    if match:
        return {
            "model_name": match.group("model"),
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "images_per_prompt": int(match.group("images_per_prompt")),
            "image_height": int(match.group("image_height")),
            "image_width": int(match.group("image_width")),
            "task_type": "image",
        }

    match = _STRESS_TEXT_PATTERN.search(filename)
    if match:
        return {
            "model_name": match.group("model"),
            "timestamp": match.group("timestamp"),
            "device": match.group("device"),
            "input_sequence_length": int(match.group("isl")),
            "output_sequence_length": int(match.group("osl")),
            "max_con": int(match.group("maxcon")),
            "num_requests": int(match.group("n")),
            "task_type": "text",
        }

    raise ValueError(f"Could not extract parameters from filename: {filename}")


def process_stress_test_file(filepath: str) -> Dict[str, Any]:
    """Process a single stress test file and extract metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)
    params = extract_params_from_filename(filename)

    mean_tpot_ms = data.get("mean_tpot_ms")
    if mean_tpot_ms:
        mean_tpot = max(mean_tpot_ms, 1e-6)
        mean_tps = 1000.0 / mean_tpot
        std_tps = (
            (mean_tps - (1000.0 / (mean_tpot + data["std_tpot_ms"])))
            if data.get("std_tpot_ms")
            else None
        )
    else:
        mean_tps = None
        std_tps = None

    actual_max_con = min(params["max_con"], params["num_requests"])
    tps_decode = mean_tps * actual_max_con if mean_tps else None
    tps_prefill = (params["input_sequence_length"] * actual_max_con) / (
        data.get("mean_ttft_ms") / 1000
    )

    metrics: Dict[str, Any] = {
        "timestamp": params["timestamp"],
        "model_name": params["model_name"],
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
        "device": params.get("device", ""),
        "input_sequence_length": params["input_sequence_length"],
        "output_sequence_length": params["output_sequence_length"],
        "max_con": actual_max_con,
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "std_ttft_ms": data.get("std_ttft_ms"),
        "p5_ttft_ms": data.get("p5_ttft_ms"),
        "p25_ttft_ms": data.get("p25_ttft_ms"),
        "p50_ttft_ms": data.get("p50_ttft_ms"),
        "p95_ttft_ms": data.get("p95_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "mean_tpot_ms": mean_tpot_ms,
        "std_tpot_ms": data.get("std_tpot_ms"),
        "p5_tpot_ms": data.get("p5_tpot_ms"),
        "p25_tpot_ms": data.get("p25_tpot_ms"),
        "p50_tpot_ms": data.get("p50_tpot_ms"),
        "p95_tpot_ms": data.get("p95_tpot_ms"),
        "p99_tpot_ms": data.get("p99_tpot_ms"),
        "mean_tps": mean_tps,
        "std_tps": std_tps,
        "tps_decode_throughput": tps_decode,
        "tps_prefill_throughput": tps_prefill,
        "mean_itl_ms": data.get("mean_itl_ms"),
        "std_itl_ms": data.get("std_itl_ms"),
        "p5_itl_ms": data.get("p5_itl_ms"),
        "p25_itl_ms": data.get("p25_itl_ms"),
        "p50_itl_ms": data.get("p50_itl_ms"),
        "p95_itl_ms": data.get("p95_itl_ms"),
        "p99_itl_ms": data.get("p99_itl_ms"),
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "p5_e2el_ms": data.get("p5_e2el_ms"),
        "p25_e2el_ms": data.get("p25_e2el_ms"),
        "p50_e2el_ms": data.get("p50_e2el_ms"),
        "p95_e2el_ms": data.get("p95_e2el_ms"),
        "p99_e2el_ms": data.get("p99_e2el_ms"),
        "request_throughput": data.get("request_throughput"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "num_prompts": data.get("num_prompts", ""),
        "num_requests": params["num_requests"],
        "filename": filename,
        "task_type": params["task_type"],
    }

    if params["task_type"] == "image":
        metrics["images_per_prompt"] = params["images_per_prompt"]
        metrics["image_height"] = params["image_height"]
        metrics["image_width"] = params["image_width"]

    return format_metrics(metrics, sig_digits=STRESS_TEST_SIG_DIGITS_MAP)


def process_stress_test_files(files: List[str]) -> List[Dict[str, Any]]:
    """Process multiple stress test files, returning sorted metric dicts."""
    results: List[Dict[str, Any]] = []
    logger.info(f"Processing {len(files)} stress test files")
    for filepath in files:
        try:
            metrics = process_stress_test_file(filepath)
            results.append(metrics)
        except Exception:
            logger.exception(f"Error processing stress test file {filepath}")
    if not results:
        raise ValueError("No stress test files were successfully processed")
    return sorted(results, key=lambda x: x.get("timestamp", ""))
