# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import json
import os
import csv
import re
from typing import Dict, List, Any, Union, Tuple
import argparse
from pathlib import Path


DATE_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"
NOT_MEASURED_STR = "n/a"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process vLLM benchmark results from multiple files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="One or more files containing benchmark files",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="benchmark_*.json",
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
        ^benchmark_
        (?P<model>.+?)                            # Model name (non-greedy, allows everything)
        (?:_(?P<device>N150|N300|T3K|TG|n150|n300|t3k|tg))?  # Optional device
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        \.json$
    """
    match = re.search(pattern, filename, re.VERBOSE)
    if not match:
        raise ValueError(f"Could not extract parameters from filename: {filename}")

    # Extract and convert numeric parameters
    params = {
        "timestamp": match.group("timestamp"),
        "device": match.group("device"),
        "input_sequence_length": int(match.group("isl")),
        "output_sequence_length": int(match.group("osl")),
        "max_con": int(match.group("maxcon")),
        "num_requests": int(match.group("n")),
    }

    return params


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
        "device": params.get("device", ""),
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


def process_benchmark_files(files: List[str], pattern: str) -> List[Dict[str, Any]]:
    """Process benchmark files from multiple files matching the given pattern."""
    results = []

    print(f"Processing {len(files)} files")

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


def get_markdown_table(display_dicts: List[Dict[str, str]]) -> str:
    if not display_dicts:
        return ""

    def sanitize_cell(text: str) -> str:
        text = str(text).replace("|", "\\|").replace("\n", " ")
        return re.sub(r"[^\x00-\x7F]+", "", text).strip()

    headers = list(display_dicts[0].keys())

    numeric_cols = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(d.get(header, "")).strip())
            for d in display_dicts
        )
        for header in headers
    }

    max_left, max_right = {}, {}
    for header in headers:
        max_left[header], max_right[header] = 0, 0
        if numeric_cols[header]:
            for d in display_dicts:
                val = str(d.get(header, "")).strip()
                left, _, right = val.partition(".")
                max_left[header] = max(max_left[header], len(left))
                max_right[header] = max(max_right[header], len(right))

    def format_numeric(val: str, header: str) -> str:
        left, _, right = val.partition(".")
        left = left.rjust(max_left[header])
        if max_right[header] > 0:
            right = right.ljust(max_right[header])
            return f"{left}.{right}"
        return left

    col_widths = {}
    for header in headers:
        if numeric_cols[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            col_widths[header] = max(len(header), numeric_width)
        else:
            max_content_width = max(
                len(sanitize_cell(str(d.get(header, "")))) for d in display_dicts
            )
            col_widths[header] = max(len(header), max_content_width)

    header_row = (
        "| "
        + " | ".join(
            sanitize_cell(header).center(col_widths[header]) for header in headers
        )
        + " |"
    )

    separator_row = (
        "|" + "|".join("-" * (col_widths[header] + 2) for header in headers) + "|"
    )

    value_rows = []
    for d in display_dicts:
        row = []
        for header in headers:
            cell = sanitize_cell(str(d.get(header, "")).strip())
            if numeric_cols[header]:
                cell = format_numeric(cell, header).rjust(col_widths[header])
            else:
                cell = cell.ljust(col_widths[header])
            row.append(cell)
        value_rows.append("| " + " | ".join(row) + " |")

    end_notes = "\n\nNote: all metrics are means across benchmark run unless otherwise stated.\n"

    md_str = f"\n{header_row}\n{separator_row}\n" + "\n".join(value_rows) + end_notes
    return md_str


def save_markdown_table(
    markdown_str: str, filepath: str, add_title: str = None, add_notes: List[str] = None
) -> None:
    # Convert string path to Path object and ensure .md extension
    path = Path(filepath)
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")

    # Create file if it doesn't exist
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


def generate_report(files, output_dir, metadata={}):
    assert len(files) > 0, "No benchmark files found."
    results = process_benchmark_files(files, pattern="benchmark_*.json")

    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and print Markdown table
    model_name = metadata["model_name"]
    device = results[0].get("device")
    if "device" in metadata:
        assert metadata["device"] == device, "Device mismatch in metadata"

    run_id = f"{model_name}_{device}"
    # save stats
    data_file_path = output_dir / "data" / f"benchmark_stats_{run_id}.csv"
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(results, data_file_path)

    display_results = [create_display_dict(res) for res in results]
    markdown_str = get_markdown_table(display_results)
    display_md_str = (
        f"### Performance Benchmarks for {model_name} on {device}\n\n{markdown_str}"
    )
    disp_md_path = Path(output_dir) / f"benchmark_display_{run_id}.md"
    save_markdown_table(display_md_str, disp_md_path)
    # TODO: add release report for benchmarks
    release_str = display_md_str
    release_raw = results
    return release_str, release_raw, disp_md_path, data_file_path


def main():
    args = parse_args()
    # Display basic statistics
    print("\nBenchmark Summary:")
    print(f"Total files processed: {len(args.files)}")
    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(os.environ.get("CACHE_ROOT", ""), "benchmark_results")
    release_str, release_raw, disp_md_path, data_file_path = generate_report(
        args.files, output_dir, metadata={}
    )
    print("Markdown Table:")
    print(release_str)


if __name__ == "__main__":
    main()
