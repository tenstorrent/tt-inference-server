# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import json
import glob
import os
from datetime import datetime
import re
from typing import Dict, List, Any
import argparse
from pathlib import Path


DATE_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process vLLM benchmark results from multiple directories."
    )
    parser.add_argument(
        "directories",
        nargs="+",
        type=str,
        help="One or more directories containing benchmark files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_benchmark_*.json",
        help="File pattern to match (default: vllm_online_benchmark_*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output CSV file name",
    )
    return parser.parse_args()


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    pattern = r"""
        benchmark_
        (?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})  # Timestamp
        _isl-(?P<isl>\d+)                                    # Input sequence length
        _osl-(?P<osl>\d+)                                    # Output sequence length
        _bsz-(?P<bsz>\d+)                                    # Batch size
        _n-(?P<n>\d+)                                        # Number of requests
    """

    match = re.search(pattern, filename, re.VERBOSE)
    if not match:
        raise ValueError(f"Could not extract parameters from filename: {filename}")

    # Convert timestamp string to datetime
    timestamp_str = match.group("timestamp")
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

    # Extract and convert numeric parameters
    params = {
        "timestamp": timestamp,
        "input_sequence_length": int(match.group("isl")),
        "output_sequence_length": int(match.group("osl")),
        "batch_size": int(match.group("bsz")),
        "num_requests": int(match.group("n")),
    }

    return params


def format_metrics(metrics):
    NOT_MEASURED_STR = "n/a"
    formatted_metrics = {}

    for key, value in metrics.items():
        # Skip None values and NOT_MEASURED_STR
        if value is None or value == NOT_MEASURED_STR:
            formatted_metrics[key] = NOT_MEASURED_STR
        elif isinstance(value, float):
            # Format numeric values to 2 decimal places
            formatted_metrics[key] = round(float(value), 2)
        else:
            formatted_metrics[key] = value

    return formatted_metrics


def process_benchmark_file(filepath: str) -> Dict[str, Any]:
    """Process a single benchmark file and extract relevant metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)

    params = extract_params_from_filename(filename)

    # Calculate statistics

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

    metrics = {
        "timestamp": params["timestamp"],
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
        "input_sequence_length": params["input_sequence_length"],
        "output_sequence_length": params["output_sequence_length"],
        "batch_size": params["batch_size"],
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "std_ttft_ms": data.get("std_ttft_ms"),
        "mean_tpot_ms": mean_tpot_ms,
        "std_tpot_ms": data.get("std_tpot_ms"),
        "mean_tps": mean_tps,
        "std_tps": std_tps,
        "mean_e2el_ms": data.get("mean_e2el_ms"),
        "request_throughput": data.get("request_throughput"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "num_prompts": data.get("num_prompts", ""),
        "num_requests": params["num_requests"],
        "filename": filename,
    }
    metrics = format_metrics(metrics)

    return metrics


def process_benchmark_files(
    directories: List[str], pattern: str
) -> List[Dict[str, Any]]:
    """Process benchmark files from multiple directories matching the given pattern."""
    results = []

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Warning: Directory not found: {directory}")
            continue

        file_pattern = str(dir_path / pattern)
        files = glob.glob(file_pattern)

        if not files:
            print(
                f"Warning: No files found matching pattern '{pattern}' in {directory}"
            )
            continue

        print(f"Processing {len(files)} files from {directory}")

        for filepath in files:
            print(f"Processing: {filepath} ...")
            try:
                metrics = process_benchmark_file(filepath)
                results.append(metrics)
            except Exception as e:
                print(f"Error processing file {filepath}: {str(e)}")

    if not results:
        raise ValueError("No benchmark files were successfully processed")

    # Sort by timestamp
    return sorted(results, key=lambda x: x["timestamp"])


def save_to_csv(
    results: List[Dict[str, Any]], output_dir: str, timestamp_str: str
) -> None:
    """Save results to a CSV file."""
    if not results:
        return

    file_path = Path(output_dir) / f"benchmark_results_{timestamp_str}.csv"

    # Get all unique keys from all dictionaries
    headers = list(results[0].keys())

    with open(file_path, "w") as f:
        # Write headers
        f.write(",".join(headers) + "\n")

        # Write data
        for result in results:
            row = [str(result.get(header, "")) for header in headers]
            f.write(",".join(row) + "\n")
    print(f"\nResults saved to: {file_path}")


def format_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Format results as a Markdown table."""
    if not results:
        return ""

    # Define columns to display and their headers
    display_cols = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("batch_size", "Batch Size"),
        ("num_requests", "Num Requests"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "TPS (user)"),
        ("mean_e2el_ms", "Request latency (ms)"),
        ("request_throughput", "Request Throughput (RPS)"),
    ]

    # Create header row
    header = " | ".join(header for _, header in display_cols)
    separator = "|".join(["---"] * len(display_cols))

    # Create data rows
    rows = []
    for result in results:
        row_values = []
        for col, _ in display_cols:
            value = result.get(col, "")
            # Format floats to 2 decimal places
            if isinstance(value, float):
                value = f"{value:.2f}"
            row_values.append(str(value))
        rows.append(" | ".join(row_values))

    # Combine all parts
    markdown_table = f"| {header} |\n| {separator} |\n"
    markdown_table += "\n".join(f"| {row} |" for row in rows)

    return markdown_table


def extract_timestamp(directories):
    pattern = r"""
        results_
        (?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})  # Timestamp
    """
    first_dir = directories[0]
    match = re.search(pattern, first_dir, re.VERBOSE)
    if not match:
        raise ValueError(f"Could not extract parameters from: {first_dir}")

    # Convert timestamp string to datetime
    timestamp_str = match.group("timestamp")

    return timestamp_str


def main():
    args = parse_args()

    results = process_benchmark_files(args.directories, args.pattern)
    timestamp_str = extract_timestamp(args.directories)

    # Display basic statistics
    print("\nBenchmark Summary:")
    print(f"Total files processed: {len(results)}")

    # Save to CSV
    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(os.environ.get("CACHE_ROOT", ""), "benchmark_results")
        os.makedirs(output_dir, exist_ok=True)

    save_to_csv(results, output_dir, timestamp_str)

    # Generate and print Markdown table
    print("\nMarkdown Table:\n")

    print(f"Model ID: {results[0].get('model_id')}")
    print(f"Backend: {results[0].get('backend')}")
    print(format_markdown_table(results))
    print("Note: all metrics are means across benchmark run unless otherwise stated.\n")


if __name__ == "__main__":
    main()
