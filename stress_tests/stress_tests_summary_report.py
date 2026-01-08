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
import unicodedata


DATE_STR_FORMAT = "%Y-%m-%d_%H-%M-%S"
NOT_MEASURED_STR = "n/a"


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
        "--pattern",
        type=str,
        default="stress_test_*.json",
        help="File pattern to match (default: stress_test_*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output CSV file name",
    )
    return parser.parse_args()


def extract_params_from_filename(filename: str) -> Dict[str, Any]:
    # First try the image stress_test pattern
    image_pattern = r"""
        ^stress_test_
        (?P<model>.+?)                            # Model name (non-greedy, allows everything)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?  # Optional device
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        _images-(?P<images_per_prompt>\d+)
        _height-(?P<image_height>\d+)
        _width-(?P<image_width>\d+)
        \.json$
    """

    # Try image pattern first
    match = re.search(image_pattern, filename, re.VERBOSE)
    if match:
        # Extract and convert numeric parameters for image benchmarks
        params = {
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
        return params

    # Fall back to text stress_test pattern
    text_pattern = r"""
        ^stress_test_
        (?P<model>.+?)                            # Model name (non-greedy, allows everything)
        (?:_(?P<device>N150|N300|P100|P150|T3K|p150x4|n150x4|TG|GALAXY|n150|n300|p100|p150|t3k|tg|galaxy))?  # Optional device
        _(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})
        _isl-(?P<isl>\d+)
        _osl-(?P<osl>\d+)
        _maxcon-(?P<maxcon>\d+)
        _n-(?P<n>\d+)
        \.json$
    """
    match = re.search(text_pattern, filename, re.VERBOSE)

    if match:
        # Extract and convert numeric parameters for text benchmarks
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

    # If no patterns match, raise error
    raise ValueError(f"Could not extract parameters from filename: {filename}")


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
    params = extract_params_from_filename(filename)

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

    # Add image-specific parameters if this is an image benchmark
    if params["task_type"] == "image":
        metrics.update(
            {
                "images_per_prompt": params["images_per_prompt"],
                "image_height": params["image_height"],
                "image_width": params["image_width"],
            }
        )

    metrics = format_metrics(metrics)

    return metrics


def process_benchmark_files(files: List[str], pattern: str) -> List[Dict[str, Any]]:
    """Process stress test files from multiple files matching the given pattern."""
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
        raise ValueError("No stress test files were successfully processed")

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


def create_image_display_dict(result: Dict[str, Any]) -> Dict[str, str]:
    # Define display columns mapping for image benchmarks
    display_cols: List[Tuple[str, str]] = [
        ("input_sequence_length", "ISL"),
        ("output_sequence_length", "OSL"),
        ("max_con", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
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


def sanitize_cell(text: str) -> str:
    text = str(text).replace("|", "\\|").replace("\n", " ")
    return text.strip()


def _cell_width(ch: str) -> int:
    # Combining characters take zero width
    if unicodedata.combining(ch):
        return 0
    # East Asian Fullwidth or Wide → 2 columns
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    # Everything else → 1 column
    return 1


def wcswidth(text: str) -> int:
    """Return the number of monospace columns `text` will occupy."""
    return sum(_cell_width(ch) for ch in text)


def pad_right(text: str, width: int) -> str:
    return text + " " * max(width - wcswidth(text), 0)


def pad_left(text: str, width: int) -> str:
    return " " * max(width - wcswidth(text), 0) + text


def pad_center(text: str, width: int) -> str:
    total = width - wcswidth(text)
    left = total // 2
    return " " * max(left, 0) + text + " " * max(total - left, 0)


def get_markdown_table(display_dicts: List[Dict[str, str]]) -> str:
    if not display_dicts:
        return ""

    headers = list(display_dicts[0].keys())

    # Detect which columns are purely numeric
    numeric_cols = {
        header: all(
            re.match(r"^-?\d+(\.\d+)?$", str(d.get(header, "")).strip())
            for d in display_dicts
        )
        for header in headers
    }

    # Precompute numeric left/right widths
    max_left, max_right = {}, {}
    for header in headers:
        max_left[header] = max_right[header] = 0
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

    # Compute final column widths (in display cells)
    col_widths = {}
    for header in headers:
        if numeric_cols[header]:
            numeric_width = (
                max_left[header]
                + (1 if max_right[header] > 0 else 0)
                + max_right[header]
            )
            col_widths[header] = max(wcswidth(header), numeric_width)
        else:
            max_content = max(
                wcswidth(sanitize_cell(str(d.get(header, "")))) for d in display_dicts
            )
            col_widths[header] = max(wcswidth(header), max_content)

    # Build header row
    header_row = (
        "| "
        + " | ".join(
            pad_center(sanitize_cell(header), col_widths[header]) for header in headers
        )
        + " |"
    )

    # Build separator row
    separator_row = (
        "|" + "|".join("-" * (col_widths[header] + 2) for header in headers) + "|"
    )

    # Build value rows
    value_rows = []
    for d in display_dicts:
        cells = []
        for header in headers:
            raw = sanitize_cell(str(d.get(header, "")).strip())
            if numeric_cols[header]:
                num = format_numeric(raw, header)
                cell = pad_left(num, col_widths[header])
            else:
                cell = pad_right(raw, col_widths[header])
            cells.append(cell)
        value_rows.append("| " + " | ".join(cells) + " |")

    end_notes = "\n\nNote: all metrics are means across stress test run unless otherwise stated.\n"

    # (Optional) header descriptions
    def clean_header(h: str) -> str:
        return re.sub(r"\s*\(.*?\)", "", h).strip()

    def describe_headers_from_keys(keys: List[str]) -> str:
        EXPLANATION_MAP = {
            "ISL": "Input Sequence Length (tokens)",
            "OSL": "Output Sequence Length (tokens)",
            "Concurrency": "number of concurrent requests (batch size)",
            "N Req": "total number of requests (sample size, N)",
            "TTFT": "Time To First Token (ms)",
            "TPOT": "Time Per Output Token (ms)",
            "Tput User": "Throughput per user (TPS)",
            "Tput Decode": "Throughput for decode tokens, across all users (TPS)",
            "Tput Prefill": "Throughput for prefill tokens (TPS)",
            "E2EL": "End-to-End Latency (ms)",
            "Req Tput": "Request Throughput (RPS)",
        }
        return "\n".join(
            f"> {key}: {EXPLANATION_MAP[key]}" for key in keys if key in EXPLANATION_MAP
        )

    key_list = [clean_header(k) for k in headers]
    explain_str = describe_headers_from_keys(key_list)

    return "\n".join([header_row, separator_row] + value_rows) + end_notes + explain_str


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


def generate_report(files, output_dir, report_id, metadata={}):
    assert len(files) > 0, "No stress test files found."
    results = process_benchmark_files(files, pattern="stress_test_*.json")

    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate and print Markdown table
    model_name = metadata["model_name"]
    device = results[0].get("device")
    if "device" in metadata:
        assert metadata["device"] == device, "Device mismatch in metadata"

    # save stats
    data_file_path = output_dir / "data" / f"stress_test_stats_{report_id}.csv"
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(results, data_file_path)

    # Separate text and image stress tests
    text_results = [r for r in results if r.get("task_type") == "text"]
    image_results = [r for r in results if r.get("task_type") == "image"]

    markdown_sections = []

    # Generate text stress tests section if any exist
    if text_results:
        text_display_results = [create_display_dict(res) for res in text_results]
        text_markdown_str = get_markdown_table(text_display_results)
        text_section = f"#### Text-to-Text Performance Stress Tests for {model_name} on {device}\n\n{text_markdown_str}"
        markdown_sections.append(text_section)

    # Generate image stress tests section if any exist
    if image_results:
        image_display_results = [
            create_image_display_dict(res) for res in image_results
        ]
        image_markdown_str = get_markdown_table(image_display_results)
        image_section = f"#### Image Stress Tests for {model_name} on {device}\n\n{image_markdown_str}"
        markdown_sections.append(image_section)

    # Combine sections
    if markdown_sections:
        display_md_str = (
            f"### Performance Stress Tests for {model_name} on {device}\n\n"
            + "\n\n".join(markdown_sections)
        )
    else:
        # Fallback to original behavior if no task_type is found
        display_results = [create_display_dict(res) for res in results]
        markdown_str = get_markdown_table(display_results)
        display_md_str = f"### Performance Stress Tests for {model_name} on {device}\n\n{markdown_str}"

    disp_md_path = Path(output_dir) / f"stress_test_display_{report_id}.md"
    save_markdown_table(display_md_str, disp_md_path)

    release_str = display_md_str
    release_raw = results
    return release_str, release_raw, disp_md_path, data_file_path


def main():
    args = parse_args()
    # Display basic statistics
    print("\nStress Test Summary:")
    print(f"Total files processed: {len(args.files)}")
    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(os.environ.get("CACHE_ROOT", ""), "stress_test_results")
    release_str, release_raw, disp_md_path, data_file_path = generate_report(
        args.files, output_dir, metadata={}
    )
    print("Markdown Table:")
    print(release_str)


if __name__ == "__main__":
    main()
