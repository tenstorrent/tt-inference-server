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
        "--output",
        type=str,
        default=f"benchmark_results_{datetime.now().strftime(DATE_STR_FORMAT)}.csv",
        help="Output CSV file name",
    )
    return parser.parse_args()


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract all parameters from benchmark filename using regex.
    Example: vllm_online_benchmark_2024-12-17_13-24-17_isl-128_osl-128_bsz-32_n-32.json

    Returns:
        Dictionary containing timestamp and numeric parameters
    """
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


def process_benchmark_file(filepath: str) -> Dict[str, Any]:
    """Process a single benchmark file and extract relevant metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    filename = os.path.basename(filepath)

    params = extract_params_from_filename(filename)
    timestamp = params.pop("timestamp")  # Remove timestamp from params dict

    metrics = {
        "filepath": filepath,
        "filename": filename,
        "timestamp": timestamp,
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
        "num_prompts": data.get("num_prompts", ""),
        "mean_tpot_ms": data.get("mean_tpot_ms", "n/a"),
        "std_tpot_ms": data.get("std_tpot_ms", "n/a"),
        "mean_ttft_ms": data.get("mean_ttft_ms", "n/a"),
        "std_ttft_ms": data.get("std_ttft_ms", "n/a"),
        "total_input_tokens": data.get("total_input_tokens", "n/a"),
        "total_output_tokens": data.get("total_output_tokens", "n/a"),
        "mean_e2el_ms": data.get("mean_e2el_ms", "n/a"),
        "request_throughput": data.get("request_throughput", "n/a"),
        **params,  # Unpack the extracted parameters
    }

    # Calculate statistics
    mean_tpot = max(metrics["mean_tpot_ms"], 1e-6)  # Avoid division by zero
    mean_tps = 1000.0 / mean_tpot
    std_tps = mean_tps - (1000.0 / (mean_tpot + metrics["std_tpot_ms"]))
    metrics["mean_tps"] = mean_tps
    metrics["std_tps"] = std_tps
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


def save_to_csv(results: List[Dict[str, Any]], filename: str) -> None:
    """Save results to a CSV file."""
    if not results:
        return

    # Get all unique keys from all dictionaries
    headers = list(results[0].keys())

    with open(filename, "w") as f:
        # Write headers
        f.write(",".join(headers) + "\n")

        # Write data
        for result in results:
            row = [str(result.get(header, "")) for header in headers]
            f.write(",".join(row) + "\n")


def format_markdown_table(results: List[Dict[str, Any]]) -> str:
    """Format results as a Markdown table."""
    if not results:
        return ""

    # Define columns to display and their headers
    display_cols = [
        ("model_id", "Model ID"),
        ("backend", "Backend"),
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


def main():
    args = parse_args()

    results = process_benchmark_files(args.directories, args.pattern)

    # Display basic statistics
    print("\nBenchmark Summary:")
    print(f"Total files processed: {len(results)}")

    # Save to CSV
    save_to_csv(results, args.output)
    print(f"\nResults saved to: {args.output}")

    # Generate and print Markdown table
    print("\nMarkdown Table:\n")
    print(format_markdown_table(results))
    print("Note: all metrics are means across benchmark run unless otherwise stated.\n")


if __name__ == "__main__":
    main()
