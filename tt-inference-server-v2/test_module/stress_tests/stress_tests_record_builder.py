# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SCHEMA_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
RUNNER_DATE_FORMAT = "%Y%m%d-%H%M%S"
NOT_MEASURED_STR = "n/a"
RECORD_KIND = "stress_tests"


def _normalize_timestamp(text: str) -> str:
    """Convert the runner's ``data["date"]`` to the schema's display format."""
    try:
        return datetime.strptime(text, RUNNER_DATE_FORMAT).strftime(
            SCHEMA_TIMESTAMP_FORMAT
        )
    except (ValueError, TypeError):
        return text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process vLLM stress test results from multiple files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="One or more files containing stress test files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to write aggregated JSON results to (omit to print to stdout)",
    )
    return parser.parse_args()


_REQUIRED_PARAM_KEYS = (
    "model_id",
    "input_sequence_length",
    "output_sequence_length",
    "max_concurrency",
    "num_prompts",
)


def extract_params_from_data(
    data: Dict[str, Any], filename: str
) -> Dict[str, Any]:
    """Pull sweep params out of the result-JSON body.

    Replaces the old filename-regex approach: the runner now writes every
    sweep param into the body, so the filename is just a label. The
    aggregator only needs the body. ``filename`` is kept for error
    messages and for the trailing ``filename`` field on the metric row.
    """
    missing = [k for k in _REQUIRED_PARAM_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"{filename}: result JSON is missing required keys: {missing}. "
            "If this came from an older runner, rerun with the updated "
            "stress_tests_benchmarking_script.py which writes these into "
            "the body."
        )

    return {
        "model_name": str(data["model_id"]),
        "timestamp": str(data.get("date", "")),
        "device": str(data.get("device", "")),
        "input_sequence_length": int(data["input_sequence_length"]),
        "output_sequence_length": int(data["output_sequence_length"]),
        "max_con": int(data["max_concurrency"]),
        "num_requests": int(data["num_prompts"]),
        "task_type": str(data.get("task_type", "text")),
    }


def format_metrics(metrics):
    formatted_metrics = {}
    sig_digits_map = {
        "mean_ttft_ms": 1,
        "mean_tpot_ms": 1,
        "mean_tps": 2,
        "mean_e2el_ms": 1,
        "mean_itl_ms": 1,
        "tps_decode_throughput": 1,
        "tps_prefill_throughput": 1,
        "request_throughput": 3,
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

    for key, value in metrics.items():
        # Skip None values and NOT_MEASURED_STR
        if value is None or value == NOT_MEASURED_STR:
            formatted_metrics[key] = NOT_MEASURED_STR
        elif isinstance(value, float):
            # Format numeric values to 2 decimal places
            formatted_metrics[key] = round(float(value), sig_digits_map.get(key, 2))
        else:
            formatted_metrics[key] = value

    return formatted_metrics


def process_benchmark_file(filepath: str) -> Dict[str, Any]:
    """Process a single stress test file and extract relevant metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)
    params = extract_params_from_data(data, filename)

    # Calculate statistics for text/image stress tests
    mean_tpot_ms = data.get("mean_tpot_ms")
    if data.get("mean_tpot_ms"):
        mean_tpot = max(data.get("mean_tpot_ms"), 1e-6)  # Avoid division by zero
        mean_tps = 1000.0 / mean_tpot
        if data.get("std_tpot_ms"):
            std_tps = mean_tps - (1000.0 / (mean_tpot + data.get("std_tpot_ms")))
        else:
            std_tps = None
    else:
        mean_tps = None
        std_tps = None
    actual_max_con = min(params["max_con"], params["num_requests"])
    tps_decode_throughput = mean_tps * actual_max_con if mean_tps else None
    tps_prefill_throughput = (params["input_sequence_length"] * actual_max_con) / (
        data.get("mean_ttft_ms") / 1000
    )

    metrics = {
        "kind": RECORD_KIND,
        "model": params["model_name"],
        "device": params.get("device", ""),
        "timestamp": _normalize_timestamp(params["timestamp"]),
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
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
        "tps_decode_throughput": tps_decode_throughput,
        "tps_prefill_throughput": tps_prefill_throughput,
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

    # Image-specific params are read straight from the JSON body when an
    # image stress-test runner emits them.
    if params["task_type"] == "image":
        metrics.update(
            {
                "images_per_prompt": data.get("images_per_prompt"),
                "image_height": data.get("image_height"),
                "image_width": data.get("image_width"),
            }
        )

    metrics = format_metrics(metrics)

    return metrics


LATENCY_METRICS: List[str] = ["ttft", "tpot", "itl", "e2el"]
LATENCY_STATS: List[str] = ["mean", "std", "p5", "p25", "p50", "p95", "p99"]


def _config_label(s: Dict[str, Any]) -> str:
    return (
        f"isl={s['input_sequence_length']}"
        f"/osl={s['output_sequence_length']}"
        f"/c={s['max_con']}"
    )


def _config_row(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": _config_label(s),
        "isl": s["input_sequence_length"],
        "osl": s["output_sequence_length"],
        "concurrency": s["max_con"],
        "n_requests": s["num_requests"],
        "timestamp": s["timestamp"],
    }


def _latency_row(s: Dict[str, Any], metric: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"config": _config_label(s)}
    for stat in LATENCY_STATS:
        key = f"{stat}_{metric}_ms"
        if key in s:
            row[stat] = s[key]
    return row


def _throughput_row(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": _config_label(s),
        "request_tput": s.get("request_throughput"),
        "user_tps": s.get("mean_tps"),
        "decode_tps": s.get("tps_decode_throughput"),
        "prefill_tps": s.get("tps_prefill_throughput"),
    }


def _token_totals_row(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": _config_label(s),
        "total_input_tokens": s.get("total_input_tokens"),
        "total_output_tokens": s.get("total_output_tokens"),
        "num_prompts": s.get("num_prompts"),
    }


def _image_params_row(s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": _config_label(s),
        "images_per_prompt": s.get("images_per_prompt"),
        "image_height": s.get("image_height"),
        "image_width": s.get("image_width"),
    }


def to_report_record(files: List[str]) -> Dict[str, Any]:
    """Aggregate per-sweep stress-test JSONs into a single schema-conformant record.

    The returned dict has the universal envelope (`kind`/`model`/`device`/`timestamp`)
    plus one nested sub-table per metric category. Designed so the renderer's
    single-record path produces one heading-and-table per sub-section instead of
    a single 40-column row-per-sweep table.
    """
    print(f"Processing {len(files)} files")
    per_sweep: List[Dict[str, Any]] = []
    for filepath in files:
        print(f"Processing: {filepath} ...")
        try:
            per_sweep.append(process_benchmark_file(filepath))
        except Exception as e:
            print(f"Error processing file {filepath}: {str(e)}")

    if not per_sweep:
        raise ValueError("No stress test files were successfully processed")

    per_sweep.sort(
        key=lambda r: (
            r.get("input_sequence_length", 0),
            r.get("output_sequence_length", 0),
            r.get("max_con", 0),
        )
    )

    first = per_sweep[0]
    record: Dict[str, Any] = {
        "kind": RECORD_KIND,
        "model": first["model"],
        "device": first.get("device", ""),
        "timestamp": first["timestamp"],
        "model_id": first.get("model_id", ""),
        "backend": first.get("backend", ""),
        "Configurations": [_config_row(s) for s in per_sweep],
    }
    for metric in LATENCY_METRICS:
        record[f"{metric.upper()} (ms)"] = [_latency_row(s, metric) for s in per_sweep]
    record["Throughput"] = [_throughput_row(s) for s in per_sweep]
    record["Token Totals"] = [_token_totals_row(s) for s in per_sweep]

    image_sweeps = [s for s in per_sweep if s.get("task_type") == "image"]
    if image_sweeps:
        record["Image Parameters"] = [_image_params_row(s) for s in image_sweeps]
    return record


def main():
    args = parse_args()
    print("\nStress Test Summary:")
    print(f"Total files processed: {len(args.files)}")
    record = to_report_record(args.files)

    if args.output_dir:
        out_path = Path(args.output_dir) / "stress_test_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Wrote record to {out_path}")
    else:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
