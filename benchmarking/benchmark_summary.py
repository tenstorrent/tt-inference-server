# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import json
import glob
import os
import csv
from datetime import datetime
import re
from typing import Dict, List, Any, Union, Tuple
import argparse
from pathlib import Path


DATE_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"
NOT_MEASURED_STR = "n/a"


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
        .*?benchmark_                                       # Any prefix before benchmark_
        (?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})  # Timestamp
        (_(?P<mesh_device>N150|N300|T3K|T3K_LINE|T3K_RING|TG))? # MESH_DEVICE
        _isl-(?P<isl>\d+)                                   # Input sequence length
        _osl-(?P<osl>\d+)                                   # Output sequence length
        _maxcon-(?P<maxcon>\d+)                            # Max concurrency
        _n-(?P<n>\d+)                                      # Number of requests
        \.json$
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
        "mesh_device": match.group("mesh_device"),
        "input_sequence_length": int(match.group("isl")),
        "output_sequence_length": int(match.group("osl")),
        "max_con": int(match.group("maxcon")),
        "num_requests": int(match.group("n")),
    }

    return params


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


def format_metrics(metrics):
    formatted_metrics = {}
    sig_digits_map = {
        "mean_ttft_ms": 1,
        "mean_tpot_ms": 1,
        "mean_tps": 2,
        "mean_e2el_ms": 1,
        "tps_decode_throughput": 1,
        "tps_prefill_throughput": 1,
        "request_throughput": 3,
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

    tps_decode_throughput = mean_tps * params["max_con"] if mean_tps else None
    tps_prefill_throughput = (params["input_sequence_length"] * params["max_con"]) / (
        data.get("mean_ttft_ms") / 1000
    )

    metrics = {
        "timestamp": params["timestamp"],
        "model_id": data.get("model_id", ""),
        "backend": data.get("backend", ""),
        "mesh_device": params.get("mesh_device", ""),
        "input_sequence_length": params["input_sequence_length"],
        "output_sequence_length": params["output_sequence_length"],
        "max_con": params["max_con"],
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "std_ttft_ms": data.get("std_ttft_ms"),
        "mean_tpot_ms": mean_tpot_ms,
        "std_tpot_ms": data.get("std_tpot_ms"),
        "mean_tps": mean_tps,
        "std_tps": std_tps,
        "tps_decode_throughput": tps_decode_throughput,
        "tps_prefill_throughput": tps_prefill_throughput,
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


def save_to_csv(results: List[Dict[str, Any]], file_path: Union[Path, str]) -> None:
    if not results:
        return

    # Get headers from first result (assuming all results have same structure)
    headers = list(results[0].keys())

    try:
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write headers
            writer.writerow(headers)
            # Write data rows
            for result in results:
                row = [str(result.get(header, NOT_MEASURED_STR)) for header in headers]
                writer.writerow(row)

        print(f"\nResults saved to: {file_path}")

    except Exception as e:
        print(f"Error saving CSV file: {e}")


def create_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    # Define display columns mapping
    display_cols: List[Tuple[str, str]] = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Concurrency"),
        ("num_requests", "N Req"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("mean_tps", "Tput User (TPS)"),
        ("tps_decode_throughput", "Tput Decode (TPS)"),
        ("tps_prefill_throughput", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    display_dict = {}
    for col_name, display_header in display_cols:
        value = result.get(col_name, NOT_MEASURED_STR)
        display_dict[display_header] = str(value)

    return display_dict


def get_markdown_table(display_dicts: List[Dict[str, str]], metadata: str = "") -> str:
    if not display_dicts:
        return ""

    def sanitize_cell(text: str) -> str:
        """Sanitize cell content for Markdown compatibility"""
        # Replace problematic characters
        text = str(text)
        text = text.replace("|", "\\|")  # Escape pipe characters
        text = text.replace("\n", " ")  # Replace newlines with spaces
        text = re.sub(r"[^\x00-\x7F]+", "", text)  # Remove non-ASCII characters
        return text.strip()

    # Get headers from first dictionary
    headers = list(display_dicts[0].keys())

    # Calculate column widths based on all values including headers
    col_widths = {}
    for header in headers:
        # Include header length in width calculation
        width = len(header)
        # Check all values for this column
        for d in display_dicts:
            width = max(width, len(str(d.get(header, ""))))
        # Add minimum width of 3
        col_widths[header] = max(width, 3)

    # Create header row with proper padding
    header_row = (
        "| "
        + " | ".join(
            sanitize_cell(header).ljust(col_widths[header]) for header in headers
        )
        + " |"
    )

    # Create separator row with proper alignment indicators
    separator_row = (
        "|"
        + "|".join(":" + "-" * (col_widths[header]) + ":" for header in headers)
        + "|"
    )

    # Create value rows with proper padding
    value_rows = []
    for d in display_dicts:
        row = (
            "| "
            + " | ".join(
                sanitize_cell(str(d.get(header, ""))).ljust(col_widths[header])
                for header in headers
            )
            + " |"
        )
        value_rows.append(row)

    # add notes
    end_notes = (
        "\nNote: all metrics are means across benchmark run unless otherwise stated.\n"
    )
    # Combine all rows
    md_str = (
        metadata
        + f"\n{header_row}\n{separator_row}\n"
        + "\n".join(value_rows)
        + end_notes
    )
    return md_str


def save_markdown_table(
    markdown_str: str, filepath: str, add_title: str = None, add_notes: List[str] = None
) -> None:
    # Convert string path to Path object and ensure .md extension
    path = Path(filepath)
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")

    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare content
    content = []
    if add_title:
        # Add title with markdown h1 formatting and blank line
        content.extend([f"# {add_title}", ""])
    content.append(markdown_str)
    if add_notes:
        content.extend(add_notes)

    # Write to file with UTF-8 encoding
    try:
        path.write_text("\n".join(content), encoding="utf-8")
        print(f"Successfully saved markdown table to: {path}")
    except Exception as e:
        print(f"Error saving markdown table: {str(e)}")


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

    # save stats
    stats_file_path = Path(output_dir) / f"benchmark_stats_{timestamp_str}.csv"
    save_to_csv(results, stats_file_path)

    display_results = [create_display_dict(res) for res in results]
    disp_file_path = Path(output_dir) / f"benchmark_display_{timestamp_str}.csv"
    save_to_csv(display_results, disp_file_path)
    # Generate and print Markdown table
    print("\nMarkdown Table:\n")
    metadata = (
        f"Model ID: {results[0].get('model_id')}\n"
        f"Backend: {results[0].get('backend')}\n"
        f"mesh_device: {results[0].get('mesh_device')}\n"
    )
    display_md_str = get_markdown_table(display_results, metadata=metadata)
    print(display_md_str)
    disp_md_path = Path(output_dir) / f"benchmark_display_{timestamp_str}.md"
    save_markdown_table(display_md_str, disp_md_path)


if __name__ == "__main__":
    main()
