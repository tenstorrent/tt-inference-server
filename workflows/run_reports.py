# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarking.benchmark_config import cap_benchmark_params
from benchmarking.summary_report import (
    generate_report as benchmark_generate_report_helper,
)
from benchmarking.summary_report import get_markdown_table
from evals.eval_config import EVAL_CONFIGS
from stress_tests.stress_tests_summary_report import (
    generate_report as stress_test_generate_report_helper,
)
from tests.utils.vllm_parameter_json_to_md import main as generate_vllm_parameter_report
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec, ModelType
from workflows.utils import (
    get_default_workflow_root_log_dir,
    is_preprocessing_enabled_for_whisper,
    is_streaming_enabled_for_whisper,
)
from workflows.utils_report import get_performance_targets
from workflows.workflow_config import (
    WORKFLOW_REPORT_CONFIG,
)

# from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import DeviceTypes, ReportCheckTypes

logger = logging.getLogger(__name__)

# Media clients (audio, cnn) constants
FUNCTIONAL_TARGET = 10
COMPLETE_TARGET = 2


def generate_embedding_report_data(model_spec, eval_run_id):
    """Generate embedding-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for embedding evaluation results
    """
    # Embedding models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_audio_report_data(model_spec, eval_run_id):
    """Generate audio-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for audio evaluation results
    """
    # Audio models use *_results.json pattern (created by lmms-eval)
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/*_results.json"
    return file_name_pattern


def generate_cnn_report_data(model_spec, eval_run_id):
    """Generate CNN-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for CNN evaluation results
    """
    # CNN models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_video_report_data(model_spec, eval_run_id):
    """Generate video-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for CNN evaluation results
    """
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_image_report_data(model_spec, eval_run_id):
    """Generate image-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for image evaluation results
    """
    # Image models use results_*.json pattern
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def generate_tts_report_data(model_spec, eval_run_id):
    """Generate TTS-specific report data.

    Args:
        model_spec: Model specification
        eval_run_id: Evaluation run ID

    Returns:
        File pattern for TTS evaluation results
    """
    # TTS models use results_*.json pattern (same as image/cnn)
    file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
    return file_name_pattern


def get_embedding_benchmark_targets(model_spec, device_str, logger):
    """Get embedding-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for embedding models
    """
    from workflows.model_spec import model_performance_reference

    model_data = model_performance_reference.get(model_spec.model_name, {})
    device_json_list = model_data.get(device_str, [])

    if not device_json_list:
        logger.warning(
            f"No performance targets found for embedding model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def get_audio_benchmark_targets(model_spec, device_str, logger):
    """Get audio-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for audio models
    """
    from workflows.model_spec import model_performance_reference

    model_data = model_performance_reference.get(model_spec.model_name, {})
    device_json_list = model_data.get(device_str, [])

    if not device_json_list:
        logger.warning(
            f"No performance targets found for audio model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def get_cnn_benchmark_targets(model_spec, device_str, logger):
    """Get CNN-specific benchmark targets.

    Args:
        model_spec: Model specification
        device_str: Device string
        logger: Logger instance

    Returns:
        Benchmark target data for CNN models
    """
    from workflows.model_spec import model_performance_reference

    model_data = model_performance_reference.get(model_spec.model_name, {})
    device_json_list = model_data.get(device_str, [])

    if not device_json_list:
        logger.warning(
            f"No performance targets found for CNN model {model_spec.model_name} on {device_str}"
        )

    return device_json_list


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM reports")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run on",
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for report output",
        required=True,
    )
    ret_args = parser.parse_args()
    return ret_args


def flatten_target_checks(rows):
    flat_rows = []
    for row in rows:
        # Start with all the top-level keys except "target_checks"
        flat = {k: v for k, v in row.items() if k != "target_checks"}
        # For each target (e.g. "reference", "other"), and each metric inside it,
        # create a new key "<target>_<metric>"
        for target_name, checks in row.get("target_checks", {}).items():
            for metric, value in checks.items():
                flat[f"{target_name}_{metric}"] = value
        flat_rows.append(flat)
    return flat_rows


def benchmark_release_markdown(release_raw, target_checks=None):
    # Define display columns mapping
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    check_cols = []
    if target_checks:
        # NOTE: set column order via tuple
        check_cols = [
            (
                f"{k}_{metric}",
                " ".join(
                    w.upper() if w.lower() == "ttft" else w.capitalize()
                    for w in f"{k}_{metric}".split("_")
                )
                + (
                    ""  # no unit for any "_check" column
                    if metric.endswith("_check") or metric.endswith("_ratio")
                    else " (ms)"  # TTFT always in milliseconds
                    if metric.startswith("ttft")
                    else " (TPS)"  # any Tput* in transactions/second
                    if metric.startswith("tput")
                    else ""
                ),
            )
            for k in target_checks.keys()
            # NOTE: comment out columns to hide them from display
            for metric in (
                "ttft_check",
                "tput_user_check",
                # "tput_check",
                "ttft",
                # "ttft_ratio",
                "tput_user",
                # "tput_user_ratio",
                # "tput",
                # "tput_ratio",
            )
        ]
        check_cols.sort(key=lambda col: not col[0].endswith("_check"))

    display_cols += check_cols
    NOT_MEASURED_STR = "N/A"
    cols_to_round = [_col[0] for _col in check_cols]
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if isinstance(value, ReportCheckTypes):
                row_dict[display_header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round and isinstance(value, float):
                row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def benchmark_image_release_markdown(release_raw, target_checks=None):
    # Define display columns mapping for image benchmarks
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    check_cols = []
    if target_checks:
        # NOTE: set column order via tuple
        check_cols = [
            (
                f"{k}_{metric}",
                " ".join(
                    w.upper() if w.lower() == "ttft" else w.capitalize()
                    for w in f"{k}_{metric}".split("_")
                )
                + (
                    ""  # no unit for any "_check" column
                    if metric.endswith("_check") or metric.endswith("_ratio")
                    else " (ms)"  # TTFT always in milliseconds
                    if metric.startswith("ttft")
                    else " (TPS)"  # any Tput* in transactions/second
                    if metric.startswith("tput")
                    else ""
                ),
            )
            for k in target_checks.keys()
            # NOTE: comment out columns to hide them from display
            for metric in (
                "ttft_check",
                "tput_user_check",
                # "tput_check",
                "ttft",
                # "ttft_ratio",
                "tput_user",
                # "tput_user_ratio",
                # "tput",
                # "tput_ratio",
            )
        ]
        check_cols.sort(key=lambda col: not col[0].endswith("_check"))

    display_cols += check_cols
    NOT_MEASURED_STR = "N/A"
    cols_to_round = [_col[0] for _col in check_cols]
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if isinstance(value, ReportCheckTypes):
                row_dict[display_header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round and isinstance(value, float):
                row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_release_markdown(release_raw, is_image_benchmark=False):
    """Generate markdown table for AIPerf benchmarks with detailed metrics.

    This follows NVIDIA's genai-perf style output with mean, median, and p99 percentiles
    for each key metric category.

    Args:
        release_raw: Raw benchmark data
        is_image_benchmark: If True, includes image dimension columns (height, width, images per prompt)
    """
    # Define display columns mapping - NVIDIA style with detailed percentiles
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
    ]

    # Add image-specific columns for image benchmarks
    if is_image_benchmark:
        display_cols.extend(
            [
                ("image_height", "Image Height"),
                ("image_width", "Image Width"),
                ("images_per_prompt", "Images per Prompt"),
            ]
        )

    display_cols.extend(
        [
            ("num_requests", "N"),
            # TTFT metrics
            ("mean_ttft_ms", "TTFT Avg (ms)"),
            ("median_ttft_ms", "TTFT P50 (ms)"),
            ("p99_ttft_ms", "TTFT P99 (ms)"),
            # TPOT metrics (Time Per Output Token)
            ("mean_tpot_ms", "TPOT Avg (ms)"),
            ("median_tpot_ms", "TPOT P50 (ms)"),
            ("p99_tpot_ms", "TPOT P99 (ms)"),
            # E2EL metrics (End-to-End Latency)
            ("mean_e2el_ms", "E2EL Avg (ms)"),
            ("median_e2el_ms", "E2EL P50 (ms)"),
            ("p99_e2el_ms", "E2EL P99 (ms)"),
            # Throughput
            ("output_token_throughput", "Output Tok/s"),
            ("total_token_throughput", "Total Tok/s"),
            ("request_throughput", "Req/s"),
        ]
    )

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name in ("request_throughput",):
                    row_dict[display_header] = f"{value:.4f}"
                elif col_name in ("output_token_throughput", "total_token_throughput"):
                    row_dict[display_header] = f"{value:.2f}"
                else:
                    row_dict[display_header] = f"{value:.1f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_throughput_markdown(release_raw):
    """Generate markdown table for benchmarks with derived throughput metrics.

    This follows the genai-perf comparison style with Tput User, Tput Decode, and Tput Prefill
    columns for easy comparison between vLLM, AIPerf, and genai-perf benchmarks.
    """
    # Define display columns - genai-perf comparison style with Source column
    display_cols = [
        ("source", "Source"),
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
        ("num_requests", "N"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput_decode", "Tput Decode (TPS)"),
        ("tput_prefill", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        # Calculate derived throughput metrics
        tpot = row.get("mean_tpot_ms", 0)
        ttft = row.get("mean_ttft_ms", 0)
        isl = row.get("isl", 0)
        concurrency = row.get("concurrency", 1)

        tput_user = 1000.0 / tpot if tpot > 0 else 0
        tput_decode = tput_user * concurrency
        tput_prefill = (isl * concurrency) / (ttft / 1000.0) if ttft > 0 else 0

        # Add derived metrics to row
        row_with_derived = dict(row)
        row_with_derived["tput_user"] = tput_user
        row_with_derived["tput_decode"] = tput_decode
        row_with_derived["tput_prefill"] = tput_prefill

        row_dict = {}
        for col_name, display_header in display_cols:
            value = row_with_derived.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name == "request_throughput":
                    row_dict[display_header] = f"{value:.3f}"
                elif col_name in ("tput_user", "tput_decode", "tput_prefill"):
                    row_dict[display_header] = f"{value:.1f}"
                elif col_name in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms"):
                    row_dict[display_header] = f"{value:.1f}"
                else:
                    row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_throughput_markdown_with_images(release_raw):
    """Generate markdown table for image benchmarks with image parameters.
    Similar to aiperf_throughput_markdown but includes image dimensions.
    """
    # Define display columns for image benchmarks
    display_cols = [
        ("source", "Source"),
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
        ("num_requests", "N"),
        ("images", "Images"),
        ("image_width", "Width"),
        ("image_height", "Height"),
        ("mean_ttft_ms", "TTFT (ms)"),
        ("mean_tpot_ms", "TPOT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput_decode", "Tput Decode (TPS)"),
        ("tput_prefill", "Tput Prefill (TPS)"),
        ("mean_e2el_ms", "E2EL (ms)"),
        ("request_throughput", "Req Tput (RPS)"),
    ]

    NOT_MEASURED_STR = "N/A"
    display_dicts = []
    for row in release_raw:
        # Calculate derived throughput metrics
        tpot = row.get("mean_tpot_ms", 0)
        ttft = row.get("mean_ttft_ms", 0)
        isl = row.get("isl", 0)
        concurrency = row.get("concurrency", 1)

        tput_user = 1000.0 / tpot if tpot > 0 else 0
        tput_decode = tput_user * concurrency
        tput_prefill = (isl * concurrency) / (ttft / 1000.0) if ttft > 0 else 0

        # Add derived metrics to row
        row_with_derived = dict(row)
        row_with_derived["tput_user"] = tput_user
        row_with_derived["tput_decode"] = tput_decode
        row_with_derived["tput_prefill"] = tput_prefill

        row_dict = {}
        for col_name, display_header in display_cols:
            value = row_with_derived.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                # Format floats with appropriate precision
                if col_name == "request_throughput":
                    row_dict[display_header] = f"{value:.3f}"
                elif col_name in ("tput_user", "tput_decode", "tput_prefill"):
                    row_dict[display_header] = f"{value:.1f}"
                elif col_name in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2el_ms"):
                    row_dict[display_header] = f"{value:.1f}"
                else:
                    row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def aiperf_benchmark_generate_report(
    args, server_mode, model_spec, report_id, metadata={}
):
    """Generate benchmark report specifically for AIPerf results.

    AIPerf provides more detailed metrics than vLLM's benchmark_serving.py,
    including mean, median, and p99 percentiles for TTFT, TPOT, and E2EL.
    This function creates a separate report in NVIDIA's genai-perf style.
    Table 2 (Comparison) combines both vLLM and AIPerf results for easy comparison.
    """
    # All benchmark tools now use the same output directory
    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    # Look for aiperf benchmark files
    aiperf_pattern = f"aiperf_benchmark_{model_spec.model_id}_*.json"
    aiperf_files = glob(f"{benchmarks_output_dir}/{aiperf_pattern}")

    # Also look for vLLM benchmark files for comparison table
    vllm_pattern = f"benchmark_{model_spec.model_id}_*.json"
    vllm_files = glob(f"{benchmarks_output_dir}/{vllm_pattern}")

    # Look for GenAI-Perf benchmark files
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")

    output_dir = Path(args.output_path) / "benchmarks_aiperf"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("AIPerf Benchmark Summary")
    logger.info(f"Found {len(aiperf_files)} AIPerf benchmark files")
    logger.info(f"Found {len(vllm_files)} vLLM benchmark files for comparison")
    logger.info(f"Found {len(genai_files)} GenAI-Perf benchmark files for comparison")

    if not aiperf_files and not vllm_files and not genai_files:
        logger.info("No benchmark files found. Skipping AIPerf report.")
        return "", [], None, None

    # Helper function to keep only the latest file for each (isl, osl, concurrency, task_type) config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration.

        Files are sorted by name (which includes timestamp) in reverse order,
        so we keep the first occurrence of each config.

        Config key includes:
        - isl, osl, concurrency, num_requests (base params)
        - images, height, width (for image benchmarks - treated as separate configs)
        """
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, con, n = map(int, match.groups())

                # Check if this is an image benchmark (has images-X in filename)
                img_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if img_match:
                    images, height, width = map(int, img_match.groups())
                    config_key = (isl, osl, con, n, images, height, width)
                else:
                    # Text-only benchmark
                    config_key = (isl, osl, con, n, 0, 0, 0)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    # Deduplicate files to keep only latest run for each config
    vllm_files = deduplicate_by_config(vllm_files)
    aiperf_files = deduplicate_by_config(aiperf_files)
    genai_files = deduplicate_by_config(genai_files)

    logger.info(
        f"After deduplication: {len(vllm_files)} vLLM, {len(aiperf_files)} AIPerf, {len(genai_files)} GenAI-Perf files"
    )

    # Separate text-only and image benchmarks
    vllm_text_only_files = [f for f in vllm_files if "images" not in Path(f).name]
    vllm_image_files = [f for f in vllm_files if "images" in Path(f).name]
    aiperf_text_only_files = [f for f in aiperf_files if "images" not in Path(f).name]
    aiperf_image_files = [f for f in aiperf_files if "images" in Path(f).name]
    genai_text_only_files = [f for f in genai_files if "images" not in Path(f).name]
    genai_image_files = [f for f in genai_files if "images" in Path(f).name]

    logger.info(
        f"Text benchmarks: {len(vllm_text_only_files)} vLLM, {len(aiperf_text_only_files)} AIPerf, {len(genai_text_only_files)} GenAI-Perf"
    )
    logger.info(
        f"Image benchmarks: {len(vllm_image_files)} vLLM, {len(aiperf_image_files)} AIPerf, {len(genai_image_files)} GenAI-Perf"
    )

    # Process text-only vLLM benchmarks
    vllm_text_results = []
    for filepath in sorted(vllm_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}*.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "vLLM",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "vllm",
            }
            vllm_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing vLLM file {filepath}: {e}")
            continue

    # Process text-only AIPerf files
    aiperf_text_results = []
    for filepath in sorted(aiperf_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: aiperf_benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "aiperf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "aiperf",
            }
            aiperf_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing AIPerf file {filepath}: {e}")
            continue

    # Process image vLLM benchmarks
    vllm_image_results = []
    for filepath in sorted(vllm_image_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "vLLM",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "vllm",
            }
            vllm_image_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing vLLM image file {filepath}: {e}")
            continue

    # Process image AIPerf files
    aiperf_image_results = []
    for filepath in sorted(aiperf_image_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "aiperf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images_per_prompt": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "aiperf",
            }
            aiperf_image_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing AIPerf image file {filepath}: {e}")
            continue

    # Process GenAI-Perf text files
    genai_text_results = []
    for filepath in sorted(genai_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            # Pattern: genai_benchmark_*_isl-{isl}_osl-{osl}_maxcon-{con}_n-{n}.json
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf file {filepath}: {e}")
            continue

    # Process GenAI-Perf image files
    genai_image_results = []
    for filepath in sorted(genai_image_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_image_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf image file {filepath}: {e}")
            continue

    if (
        not aiperf_text_results
        and not vllm_text_results
        and not genai_text_results
        and not aiperf_image_results
        and not vllm_image_results
        and not genai_image_results
    ):
        return "", [], None, None

    # Sort text benchmarks by ISL, OSL, concurrency
    vllm_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
    aiperf_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))
    genai_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))

    # Sort image benchmarks by ISL, OSL, concurrency, image size
    vllm_image_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )
    aiperf_image_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )
    genai_image_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["image_height"],
            x["image_width"],
        )
    )

    # Build the complete report
    release_str = ""

    # Only include section if there are results to display
    if aiperf_text_results or aiperf_image_results:
        release_str = f"### Benchmark Performance Results for {model_spec.model_name} on {args.device}\n\n"

        # TEXT BENCHMARKS SECTION
        if aiperf_text_results:
            release_str += "#### AIPerf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"

            # Only show AIPerf-specific detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(aiperf_text_results)
            release_str += nvidia_markdown_str
            release_str += "\n\n"

        # IMAGE BENCHMARKS SECTION
        if aiperf_image_results:
            release_str += "#### AIPerf Image Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [AIPerf](https://github.com/ai-dynamo/aiperf)\n\n"

            # Only show AIPerf-specific detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(
                aiperf_image_results, is_image_benchmark=True
            )
            release_str += nvidia_markdown_str
            release_str += "\n\n"

        # Metric definitions
        release_str += "**Metric Definitions:**\n"
        release_str += "> - **ISL**: Input Sequence Length (tokens)\n"
        release_str += "> - **OSL**: Output Sequence Length (tokens)\n"
        release_str += "> - **Concur**: Concurrent requests (batch size)\n"
        release_str += "> - **N**: Total number of requests\n"
        release_str += "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
        release_str += "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **Output Tok/s**: Output token throughput\n"
        release_str += (
            "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
        )
        release_str += "> - **Req/s**: Request throughput\n"

    # Save markdown report
    disp_md_path = output_dir / f"aiperf_benchmark_display_{report_id}.md"
    with open(disp_md_path, "w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"AIPerf report saved to: {disp_md_path}")

    # Save CSV data for text benchmarks
    text_data_file_path = (
        output_dir / "data" / f"aiperf_benchmark_text_stats_{report_id}.csv"
    )
    text_data_file_path.parent.mkdir(parents=True, exist_ok=True)

    if aiperf_text_results:
        headers = list(aiperf_text_results[0].keys())
        with open(text_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in aiperf_text_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"AIPerf text benchmark data saved to: {text_data_file_path}")

    # Save CSV data for image benchmarks
    image_data_file_path = (
        output_dir / "data" / f"aiperf_benchmark_image_stats_{report_id}.csv"
    )
    if aiperf_image_results:
        headers = list(aiperf_image_results[0].keys())
        with open(image_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in aiperf_image_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"AIPerf image benchmark data saved to: {image_data_file_path}")

    # Return combined results for both text and image
    all_aiperf_results = aiperf_text_results + aiperf_image_results
    return release_str, all_aiperf_results, disp_md_path, text_data_file_path


def genai_perf_benchmark_generate_report(
    args, server_mode, model_spec, report_id, metadata={}
):
    """Generate benchmark report specifically for GenAI-Perf results.

    GenAI-Perf provides detailed metrics similar to AIPerf,
    including mean, median, and p99 percentiles for TTFT, TPOT, and E2EL.
    This function creates a separate detailed report following the same format as AIPerf.
    """
    # All benchmark tools now use the same output directory
    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    # Look for genai-perf benchmark files
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")

    output_dir = Path(args.output_path) / "benchmarks_genai_perf"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("GenAI-Perf Benchmark Summary")
    logger.info(f"Found {len(genai_files)} GenAI-Perf benchmark files")

    if not genai_files:
        logger.info("No GenAI-Perf benchmark files found. Skipping GenAI-Perf report.")
        return "", [], None, None

    # Helper function to keep only the latest file for each config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration."""
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, maxcon, n = match.groups()
                # For image benchmarks, also include image dimensions
                image_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if image_match:
                    images, height, width = image_match.groups()
                    config_key = (isl, osl, maxcon, n, images, height, width)
                else:
                    config_key = (isl, osl, maxcon, n)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    genai_files = deduplicate_by_config(genai_files)
    logger.info(f"After deduplication: {len(genai_files)} GenAI-Perf benchmark files")

    # Separate text-only and image benchmarks
    genai_text_only_files = [f for f in genai_files if "images" not in Path(f).name]
    genai_image_files = [f for f in genai_files if "images" in Path(f).name]

    logger.info(
        f"GenAI-Perf Text benchmarks: {len(genai_text_only_files)}, Image benchmarks: {len(genai_image_files)}"
    )

    # Process text-only GenAI-Perf benchmarks
    genai_text_results = []
    for filepath in sorted(genai_text_only_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, concurrency, num_requests = map(int, match.groups())
            else:
                # Fallback to data fields
                isl = data.get("total_input_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                osl = data.get("total_output_tokens", 0) // max(
                    data.get("num_prompts", 1), 1
                )
                concurrency = data.get("max_concurrency", 1)
                num_requests = data.get("num_prompts", 0)

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_text_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf text file {filepath}: {e}")
            continue

    # Process image GenAI-Perf files
    genai_image_results = []
    for filepath in sorted(genai_image_files):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            # Extract parameters from filename
            filename = Path(filepath).name
            match = re.search(
                r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
                filename,
            )
            if match:
                isl, osl, concurrency, num_requests, images, height, width = map(
                    int, match.groups()
                )
            else:
                logger.warning(f"Could not parse image parameters from {filename}")
                continue

            result = {
                "source": "genai-perf",
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "num_requests": num_requests,
                "images_per_prompt": images,
                "image_height": height,
                "image_width": width,
                # TTFT metrics
                "mean_ttft_ms": data.get("mean_ttft_ms", 0),
                "median_ttft_ms": data.get("median_ttft_ms", 0),
                "p99_ttft_ms": data.get("p99_ttft_ms", 0),
                "std_ttft_ms": data.get("std_ttft_ms", 0),
                # TPOT metrics
                "mean_tpot_ms": data.get("mean_tpot_ms", 0),
                "median_tpot_ms": data.get("median_tpot_ms", 0),
                "p99_tpot_ms": data.get("p99_tpot_ms", 0),
                "std_tpot_ms": data.get("std_tpot_ms", 0),
                # E2EL metrics
                "mean_e2el_ms": data.get("mean_e2el_ms", 0),
                "median_e2el_ms": data.get("median_e2el_ms", 0),
                "p99_e2el_ms": data.get("p99_e2el_ms", 0),
                "std_e2el_ms": data.get("std_e2el_ms", 0),
                # Throughput
                "output_token_throughput": data.get("output_token_throughput", 0),
                "total_token_throughput": data.get("total_token_throughput", 0),
                "request_throughput": data.get("request_throughput", 0),
                # Tokens
                "completed": data.get("completed", 0),
                "total_input_tokens": data.get("total_input_tokens", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                # Metadata
                "model_id": data.get("model_id", ""),
                "backend": "genai-perf",
            }
            genai_image_results.append(result)
        except Exception as e:
            logger.warning(f"Error processing GenAI-Perf image file {filepath}: {e}")
            continue

    if not genai_text_results and not genai_image_results:
        logger.info("No GenAI-Perf results to process.")
        return "", [], None, None

    # Sort text benchmarks by ISL, OSL, concurrency
    genai_text_results.sort(key=lambda x: (x["isl"], x["osl"], x["concurrency"]))

    # Sort image benchmarks by ISL, OSL, concurrency, image dimensions
    genai_image_results.sort(
        key=lambda x: (
            x["isl"],
            x["osl"],
            x["concurrency"],
            x["images_per_prompt"],
            x["image_height"],
            x["image_width"],
        )
    )

    # Build the complete report
    release_str = ""

    # Only include section if there are results to display
    if genai_text_results or genai_image_results:
        release_str = f"### GenAI-Perf Benchmark Performance Results for {model_spec.model_name} on {args.device}\n\n"

        # TEXT BENCHMARKS SECTION
        if genai_text_results:
            release_str += "#### GenAI-Perf Text Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"

            # Show GenAI-Perf detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(genai_text_results)
            release_str += nvidia_markdown_str
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        # IMAGE BENCHMARKS SECTION
        if genai_image_results:
            release_str += "#### GenAI-Perf Image Benchmarks - Detailed Percentiles\n\n"
            release_str += "**Benchmarking Tool:** [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer)\n\n"

            # Show GenAI-Perf detailed percentiles (mean, median, P99)
            nvidia_markdown_str = aiperf_release_markdown(
                genai_image_results, is_image_benchmark=True
            )
            release_str += nvidia_markdown_str
            release_str += "\n*Note: GenAI-Perf does not natively support total token throughput metrics.*\n\n"

        # Metric definitions
        release_str += "**Metric Definitions:**\n"
        release_str += "> - **ISL**: Input Sequence Length (tokens)\n"
        release_str += "> - **OSL**: Output Sequence Length (tokens)\n"
        release_str += "> - **Concur**: Concurrent requests (batch size)\n"
        release_str += "> - **N**: Total number of requests\n"
        release_str += "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
        release_str += "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
        release_str += "> - **Output Tok/s**: Output token throughput\n"
        release_str += (
            "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
        )
        release_str += "> - **Req/s**: Request throughput\n"

    # Save markdown report
    disp_md_path = output_dir / f"genai_perf_benchmark_display_{report_id}.md"
    with open(disp_md_path, "w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"GenAI-Perf report saved to: {disp_md_path}")

    # Save CSV data for text benchmarks
    text_data_file_path = (
        output_dir / "data" / f"genai_perf_benchmark_text_stats_{report_id}.csv"
    )
    text_data_file_path.parent.mkdir(parents=True, exist_ok=True)

    if genai_text_results:
        headers = list(genai_text_results[0].keys())
        with open(text_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in genai_text_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"GenAI-Perf text benchmark data saved to: {text_data_file_path}")

    # Save CSV data for image benchmarks
    image_data_file_path = (
        output_dir / "data" / f"genai_perf_benchmark_image_stats_{report_id}.csv"
    )
    if genai_image_results:
        headers = list(genai_image_results[0].keys())
        with open(image_data_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for result in genai_image_results:
                writer.writerow([str(result.get(h, "")) for h in headers])
        logger.info(f"GenAI-Perf image benchmark data saved to: {image_data_file_path}")

    # Return combined results for both text and image
    all_genai_results = genai_text_results + genai_image_results
    return release_str, all_genai_results, disp_md_path, text_data_file_path


def benchmark_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    # Look for vLLM, genai-perf, and AIPerf benchmark files (all stack together)
    # All benchmark tools now use the same unified output directory
    vllm_pattern = f"benchmark_{model_spec.model_id}_*.json"
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"
    aiperf_pattern = f"aiperf_benchmark_{model_spec.model_id}_*.json"

    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"

    vllm_files = glob(f"{benchmarks_output_dir}/{vllm_pattern}")
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")
    aiperf_files = glob(f"{benchmarks_output_dir}/{aiperf_pattern}")

    logger.info(
        f"Found {len(vllm_files)} vLLM, {len(genai_files)} genai-perf, and {len(aiperf_files)} AIPerf benchmark files before deduplication"
    )

    # Deduplicate files - keep only latest run for each config
    def deduplicate_by_config(files):
        """Keep only the latest file for each unique benchmark configuration."""
        config_to_file = {}
        # Sort in reverse order so latest files come first
        for filepath in sorted(files, reverse=True):
            filename = Path(filepath).name
            # Extract base config from filename
            match = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
            if match:
                isl, osl, con, n = map(int, match.groups())

                # Check if this is an image benchmark (has images-X in filename)
                img_match = re.search(
                    r"images-(\d+)_height-(\d+)_width-(\d+)", filename
                )
                if img_match:
                    images, height, width = map(int, img_match.groups())
                    config_key = (isl, osl, con, n, images, height, width)
                else:
                    # Text-only benchmark
                    config_key = (isl, osl, con, n, 0, 0, 0)

                # Only keep the first (latest) file for each config
                if config_key not in config_to_file:
                    config_to_file[config_key] = filepath
            else:
                # If no match, include the file anyway
                config_to_file[filepath] = filepath
        return list(config_to_file.values())

    vllm_files = deduplicate_by_config(vllm_files)
    genai_files = deduplicate_by_config(genai_files)
    aiperf_files = deduplicate_by_config(aiperf_files)

    logger.info(
        f"After deduplication: {len(vllm_files)} vLLM, {len(genai_files)} genai-perf, {len(aiperf_files)} AIPerf benchmark files"
    )
    output_dir = Path(args.output_path) / "benchmarks"

    if not vllm_files and not genai_files and not aiperf_files:
        logger.info("No benchmark files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )

    # Process each tool separately to generate individual tables
    # Order: vLLM -> AIPerf -> GenAI-Perf (for both text and image)
    all_tool_results = []

    # Import display functions once
    from benchmarking.summary_report import (
        create_audio_display_dict,
        create_display_dict,
        create_embedding_display_dict,
        create_image_display_dict,
        create_image_generation_display_dict,
        create_video_display_dict,
        get_markdown_table,
        save_markdown_table,
        save_to_csv,
    )

    # Process all tools and collect results by type (text/image/audio/tts/embedding/cnn)
    text_sections = []
    image_sections = []
    audio_sections = []
    tts_sections = []
    embedding_sections = []
    cnn_sections = []
    video_sections = []

    # Process vLLM benchmarks
    if vllm_files:
        _, vllm_release_raw, _, _ = benchmark_generate_report_helper(
            vllm_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(vllm_release_raw)

        # Separate text, vlm, image, audio, tts, embedding and cnn for vLLM
        vllm_text = [r for r in vllm_release_raw if r.get("task_type") == "text"]
        vllm_vlm = [r for r in vllm_release_raw if r.get("task_type") == "vlm"]
        vllm_image = [r for r in vllm_release_raw if r.get("task_type") == "image"]
        vllm_audio = [r for r in vllm_release_raw if r.get("task_type") == "audio"]
        vllm_tts = [r for r in vllm_release_raw if r.get("task_type") == "tts"]
        vllm_embedding = [
            r for r in vllm_release_raw if r.get("task_type") == "embedding"
        ]
        vllm_cnn = [r for r in vllm_release_raw if r.get("task_type") == "cnn"]
        vllm_video = [r for r in vllm_release_raw if r.get("task_type") == "video"]

        if vllm_text:
            vllm_text_display = [create_display_dict(r) for r in vllm_text]
            vllm_text_md = get_markdown_table(vllm_text_display)
            text_sections.append(
                f"#### vLLM Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_text_md}"
            )

        if vllm_vlm:
            vllm_vlm_display = [create_image_display_dict(r) for r in vllm_vlm]
            vllm_vlm_md = get_markdown_table(vllm_vlm_display)
            image_sections.append(
                f"#### vLLM VLM Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_vlm_md}"
            )

        if vllm_image:
            vllm_image_display = [create_image_display_dict(r) for r in vllm_image]
            vllm_image_md = get_markdown_table(vllm_image_display)
            image_sections.append(
                f"#### vLLM Image Generation Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_image_md}"
            )

        if vllm_audio:
            vllm_audio_display = [
                create_audio_display_dict(r, model_spec) for r in vllm_audio
            ]
            vllm_audio_md = get_markdown_table(vllm_audio_display)
            audio_sections.append(
                f"#### vLLM Audio Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_audio_md}"
            )

        if vllm_tts:
            from benchmarking.summary_report import create_tts_display_dict

            vllm_tts_display = [create_tts_display_dict(r) for r in vllm_tts]
            vllm_tts_md = get_markdown_table(vllm_tts_display)
            tts_sections.append(
                f"#### vLLM Text-to-Speech Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_tts_md}"
            )

        if vllm_embedding:
            vllm_embedding_display = [
                create_embedding_display_dict(r) for r in vllm_embedding
            ]
            vllm_embedding_md = get_markdown_table(vllm_embedding_display)
            embedding_sections.append(
                f"#### vLLM Embedding Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_embedding_md}"
            )

        if vllm_cnn:
            vllm_cnn_display = [
                create_image_generation_display_dict(r) for r in vllm_cnn
            ]
            vllm_cnn_md = get_markdown_table(vllm_cnn_display)
            cnn_sections.append(
                f"#### CNN Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_cnn_md}"
            )

        if vllm_video:
            vllm_video_display = [create_video_display_dict(r) for r in vllm_video]
            vllm_video_md = get_markdown_table(vllm_video_display)
            video_sections.append(
                f"#### vLLM Video Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{vllm_video_md}"
            )

    # Process AIPerf benchmarks
    if aiperf_files:
        _, aiperf_release_raw, _, _ = benchmark_generate_report_helper(
            aiperf_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(aiperf_release_raw)

        # Separate text, vlm, image, audio, embedding and cnn for AIPerf
        aiperf_text = [r for r in aiperf_release_raw if r.get("task_type") == "text"]
        aiperf_vlm = [r for r in aiperf_release_raw if r.get("task_type") == "vlm"]
        aiperf_image = [r for r in aiperf_release_raw if r.get("task_type") == "image"]
        aiperf_audio = [r for r in aiperf_release_raw if r.get("task_type") == "audio"]
        aiperf_embedding = [
            r for r in aiperf_release_raw if r.get("task_type") == "embedding"
        ]
        aiperf_cnn = [r for r in aiperf_release_raw if r.get("task_type") == "cnn"]
        aiperf_video = [r for r in aiperf_release_raw if r.get("task_type") == "video"]

        if aiperf_text:
            aiperf_text_display = [create_display_dict(r) for r in aiperf_text]
            aiperf_text_md = get_markdown_table(aiperf_text_display)
            text_sections.append(
                f"#### AIPerf Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_text_md}"
            )

        if aiperf_vlm:
            aiperf_vlm_display = [create_image_display_dict(r) for r in aiperf_vlm]
            aiperf_vlm_md = get_markdown_table(aiperf_vlm_display)
            image_sections.append(
                f"#### AIPerf VLM Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_vlm_md}"
            )

        if aiperf_image:
            aiperf_image_display = [create_image_display_dict(r) for r in aiperf_image]
            aiperf_image_md = get_markdown_table(aiperf_image_display)
            image_sections.append(
                f"#### AIPerf Image Generation Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_image_md}"
            )

        # Note: AIPerf does not currently support audio models
        # This section is here for future compatibility if support is added
        if aiperf_audio and aiperf_audio[0].get("backend") == "aiperf":
            aiperf_audio_display = [
                create_audio_display_dict(r, model_spec) for r in aiperf_audio
            ]
            aiperf_audio_md = get_markdown_table(aiperf_audio_display)
            audio_sections.append(
                f"#### AIPerf Audio Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_audio_md}"
            )

        # Note: AIPerf does not currently support embedding models
        # This section is here for future compatibility if support is added
        if aiperf_embedding and aiperf_embedding[0].get("backend") == "aiperf":
            aiperf_embedding_display = [
                create_embedding_display_dict(r) for r in aiperf_embedding
            ]
            aiperf_embedding_md = get_markdown_table(aiperf_embedding_display)
            embedding_sections.append(
                f"#### AIPerf Embedding Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_embedding_md}"
            )

        # Note: AIPerf does not currently support CNN models
        # This section is here for future compatibility if support is added
        if aiperf_cnn and aiperf_cnn[0].get("backend") == "aiperf":
            aiperf_cnn_display = [
                create_image_generation_display_dict(r) for r in aiperf_cnn
            ]
            aiperf_cnn_md = get_markdown_table(aiperf_cnn_display)
            cnn_sections.append(
                f"#### AIPerf CNN Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_cnn_md}"
            )

        if aiperf_video and aiperf_video[0].get("backend") == "aiperf":
            aiperf_video_display = [create_video_display_dict(r) for r in aiperf_video]
            aiperf_video_md = get_markdown_table(aiperf_video_display)
            video_sections.append(
                f"#### AIPerf Video Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{aiperf_video_md}"
            )

    # Process GenAI-Perf benchmarks
    if genai_files:
        _, genai_release_raw, _, _ = benchmark_generate_report_helper(
            genai_files, output_dir, report_id, metadata, model_spec=model_spec
        )
        all_tool_results.extend(genai_release_raw)

        # Separate text, vlm, image, audio, embedding and cnn for GenAI-Perf
        genai_text = [r for r in genai_release_raw if r.get("task_type") == "text"]
        genai_vlm = [r for r in genai_release_raw if r.get("task_type") == "vlm"]
        genai_image = [r for r in genai_release_raw if r.get("task_type") == "image"]
        genai_audio = [r for r in genai_release_raw if r.get("task_type") == "audio"]
        genai_embedding = [
            r for r in genai_release_raw if r.get("task_type") == "embedding"
        ]
        genai_cnn = [r for r in genai_release_raw if r.get("task_type") == "cnn"]
        genai_video = [r for r in genai_release_raw if r.get("task_type") == "video"]

        if genai_text:
            genai_text_display = [create_display_dict(r) for r in genai_text]
            genai_text_md = get_markdown_table(genai_text_display)
            text_sections.append(
                f"#### GenAI-Perf Text-to-Text Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_text_md}"
            )

        if genai_vlm:
            genai_vlm_display = [create_image_display_dict(r) for r in genai_vlm]
            genai_vlm_md = get_markdown_table(genai_vlm_display)
            image_sections.append(
                f"#### GenAI-Perf VLM Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_vlm_md}"
            )

        if genai_image:
            genai_image_display = [create_image_display_dict(r) for r in genai_image]
            genai_image_md = get_markdown_table(genai_image_display)
            image_sections.append(
                f"#### GenAI-Perf Image Generation Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_image_md}"
            )

        # Note: GenAI-Perf does not currently support audio models
        # This section is here for future compatibility if support is added
        if genai_audio and genai_audio[0].get("backend") == "genai-perf":
            genai_audio_display = [
                create_audio_display_dict(r, model_spec) for r in genai_audio
            ]
            genai_audio_md = get_markdown_table(genai_audio_display)
            audio_sections.append(
                f"#### GenAI-Perf Audio Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_audio_md}"
            )

        # Note: GenAI-Perf does not currently support embedding models
        # This section is here for future compatibility if support is added
        if genai_embedding and genai_embedding[0].get("backend") == "genai-perf":
            genai_embedding_display = [
                create_embedding_display_dict(r) for r in genai_embedding
            ]
            genai_embedding_md = get_markdown_table(genai_embedding_display)
            embedding_sections.append(
                f"#### GenAI-Perf Embedding Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_embedding_md}"
            )

        # Note: GenAI-Perf does not currently support CNN models
        # This section is here for future compatibility if support is added
        if genai_cnn and genai_cnn[0].get("backend") == "genai-perf":
            genai_cnn_display = [
                create_image_generation_display_dict(r) for r in genai_cnn
            ]
            genai_cnn_md = get_markdown_table(genai_cnn_display)
            cnn_sections.append(
                f"#### GenAI-Perf CNN Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_cnn_md}"
            )

        if genai_video and genai_video[0].get("backend") == "genai-perf":
            genai_video_display = [create_video_display_dict(r) for r in genai_video]
            genai_video_md = get_markdown_table(genai_video_display)
            video_sections.append(
                f"#### GenAI-Perf Video Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n{genai_video_md}"
            )

    # Combine sections: text, image, audio, embedding, then cnn (matching original order)
    markdown_sections = (
        text_sections
        + image_sections
        + audio_sections
        + tts_sections
        + embedding_sections
        + cnn_sections
        + video_sections
    )

    # Combine all sections
    release_str = ""
    if markdown_sections:
        release_str = (
            f"### Performance Benchmark Sweeps for {model_spec.model_name} on {args.device}\n\n"
            + "\n\n".join(markdown_sections)
        )

    # Save combined CSV for all tools
    stats_file_path = output_dir / "data" / f"benchmark_stats_{report_id}.csv"
    stats_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_csv(all_tool_results, stats_file_path)

    # Save display markdown
    disp_md_path = output_dir / f"benchmark_display_{report_id}.md"
    save_markdown_table(release_str, disp_md_path)

    release_raw = all_tool_results
    # release report for benchmarks
    device_type = DeviceTypes.from_string(args.device)

    # Apply capping to performance references (including vision tokens for VLM models)
    # to match what benchmarks actually use
    _model_max_concurrency = model_spec.device_model_spec.max_concurrency
    _max_context = model_spec.device_model_spec.max_context
    raw_perf_refs = (
        model_spec.device_model_spec.perf_reference
        if model_spec.device_model_spec.perf_reference
        else []
    )
    perf_refs = [
        cap_benchmark_params(
            params, _max_context, _model_max_concurrency, model_spec.model_name
        )
        for params in raw_perf_refs
    ]

    # For performance targets, use only vLLM results (as per user requirement)
    vllm_release_raw = [
        r for r in release_raw if r.get("backend") in ("vllm", "openai-chat")
    ]

    # Separate text and vlm benchmarks from vLLM results (for targets)
    text_release_raw = [
        r for r in vllm_release_raw if r.get("task_type", "text") == "text"
    ]
    vlm_release_raw = [
        r for r in vllm_release_raw if r.get("task_type", "text") == "vlm"
    ]

    # Separate text and vlm performance references
    text_perf_refs = [
        p_ref for p_ref in perf_refs if getattr(p_ref, "task_type", "text") == "text"
    ]
    vlm_perf_refs = [
        p_ref for p_ref in perf_refs if getattr(p_ref, "task_type", "text") == "vlm"
    ]

    release_sections = []

    # Process text benchmarks if they exist
    if text_perf_refs and text_release_raw:
        # make lookup dict so references can find the correct result row
        # key: (isl, osl, max_concurrency)
        text_res_dict = {
            (r["input_sequence_length"], r["output_sequence_length"], r["max_con"]): r
            for r in text_release_raw
        }
        text_perf_results = {}
        for p_ref in text_perf_refs:
            p_ref_key = (p_ref.isl, p_ref.osl, p_ref.max_concurrency)
            res = text_res_dict.get(p_ref_key)
            # add reference values to the result
            text_perf_results[p_ref_key] = {
                "isl": p_ref.isl,
                "osl": p_ref.osl,
                "max_concurrency": res["max_con"] if res else p_ref.max_concurrency,
                "model": model_spec.model_name,
                "device": args.device,
            }
            # add measurements to result and checks if defined
            if res:
                text_perf_results[p_ref_key].update(
                    {
                        "ttft": res["mean_ttft_ms"],
                        "tput_user": res["mean_tps"],
                        "tput": res["tps_decode_throughput"],
                    }
                )

                # Prepare a dictionary to hold checks for all targets.
                text_perf_results[p_ref_key]["target_checks"] = {}
                # Iterate over each target defined in p_ref.targets.
                for target_name, perf_target in p_ref.targets.items():
                    target_check = {}

                    # Check for ttft metric if defined.
                    if perf_target.ttft_ms is not None:
                        assert perf_target.ttft_ms > 0, (
                            f"ttft_ms for target '{target_name}' is not > 0: {perf_target.ttft_ms}"
                        )
                        ttft_ratio = res["mean_ttft_ms"] / perf_target.ttft_ms
                        check = ReportCheckTypes.from_result(
                            ttft_ratio < (1 + perf_target.tolerance)
                        )
                        target_check["ttft"] = perf_target.ttft_ms
                        target_check["ttft_ratio"] = ttft_ratio
                        target_check["ttft_check"] = check
                    else:
                        target_check["ttft_check"] = ReportCheckTypes.NA

                    # Check for tput_user metric if defined.
                    if perf_target.tput_user is not None:
                        assert perf_target.tput_user > 0, (
                            f"tput_user for target '{target_name}' is not > 0: {perf_target.tput_user}"
                        )
                        tput_user_ratio = res["mean_tps"] / perf_target.tput_user
                        check = ReportCheckTypes.from_result(
                            tput_user_ratio > (1 - perf_target.tolerance)
                        )
                        target_check["tput_user"] = perf_target.tput_user
                        target_check["tput_user_ratio"] = tput_user_ratio
                        target_check["tput_user_check"] = check
                    else:
                        target_check["tput_user_check"] = ReportCheckTypes.NA

                    # Check for tput metric if defined.
                    if perf_target.tput is not None:
                        assert perf_target.tput > 0, (
                            f"tput for target '{target_name}' is not > 0: {perf_target.tput}"
                        )
                        tput_ratio = res["tps_decode_throughput"] / perf_target.tput
                        check = ReportCheckTypes.from_result(
                            tput_ratio > (1 - perf_target.tolerance)
                        )
                        target_check["tput"] = perf_target.tput
                        target_check["tput_ratio"] = tput_ratio
                        target_check["tput_check"] = check
                    else:
                        target_check["tput_check"] = ReportCheckTypes.NA

                    # Save the computed checks under the target's name.
                    text_perf_results[p_ref_key]["target_checks"][target_name] = (
                        target_check
                    )

            else:
                # No result available from benchmark measurements.
                NA_STRING = "N/A"
                # In this case, add N/A for performance measures and an empty check dict per target.
                text_perf_results[p_ref_key].update(
                    {
                        "ttft": NA_STRING,
                        "tput_user": NA_STRING,
                        "tput": NA_STRING,
                        "target_checks": {
                            target_name: {
                                "ttft_check": ReportCheckTypes.NA,
                                "tput_user_check": ReportCheckTypes.NA,
                                "tput_check": ReportCheckTypes.NA,
                            }
                            for target_name in p_ref.targets.keys()
                        },
                    }
                )

        # build release performance benchmarking report for text
        sorted_text_perf_results = {
            k: text_perf_results[k] for k in sorted(text_perf_results)
        }

        text_release_raw_targets = [v for k, v in sorted_text_perf_results.items()]

        flat_text_release_raw = flatten_target_checks(text_release_raw_targets)
        text_section = f"#### Text-to-Text Performance Benchmark Targets {model_spec.model_name} on {args.device}\n\n"
        if text_release_raw_targets and text_release_raw_targets[0].get(
            "target_checks"
        ):
            text_section += benchmark_release_markdown(
                flat_text_release_raw,
                target_checks=text_release_raw_targets[0]["target_checks"],
            )
        else:
            text_section += benchmark_release_markdown(
                flat_text_release_raw, target_checks=None
            )
        release_sections.append(text_section)
    elif text_release_raw:
        # Show text benchmarks even without performance targets
        text_section = f"#### Text-to-Text Performance Benchmark Results {model_spec.model_name} on {args.device}\n\n"
        text_section += "No performance targets defined for text benchmarks.\n\n"
        release_sections.append(text_section)

    # Process VLM benchmarks if they exist
    print(f"vlm_release_raw: {vlm_release_raw}")
    if vlm_perf_refs and vlm_release_raw:
        # VLM models (ModelType.VLM) now distinguished from image generation models (ModelType.IMAGE)
        # make lookup dict so references can find the correct result row
        # key: (isl, osl, image_height, image_width, images_per_prompt, max_concurrency)
        vlm_res_dict = {
            (
                r["input_sequence_length"],
                r["output_sequence_length"],
                r["image_height"],
                r["image_width"],
                r["images_per_prompt"],
                r["max_con"],
            ): r
            for r in vlm_release_raw
        }
        vlm_perf_results = {}
        for p_ref in vlm_perf_refs:
            p_ref_key = (
                p_ref.isl,
                p_ref.osl,
                p_ref.image_height,
                p_ref.image_width,
                p_ref.images_per_prompt,
                p_ref.max_concurrency,
            )
            res = vlm_res_dict.get(p_ref_key)
            # add reference values to the result
            vlm_perf_results[p_ref_key] = {
                "isl": p_ref.isl,
                "osl": p_ref.osl,
                "max_concurrency": res["max_con"] if res else p_ref.max_concurrency,
                "image_height": p_ref.image_height,
                "image_width": p_ref.image_width,
                "images_per_prompt": p_ref.images_per_prompt,
                "num_requests": res["num_requests"] if res else "N/A",
                "model": model_spec.model_name,
                "device": args.device,
            }
            # add measurements to result and checks if defined
            if res:
                vlm_perf_results[p_ref_key].update(
                    {
                        "ttft": res["mean_ttft_ms"],
                        "tput_user": res["mean_tps"],
                        "tput": res["tps_decode_throughput"],
                    }
                )

                # Prepare a dictionary to hold checks for all targets.
                vlm_perf_results[p_ref_key]["target_checks"] = {}
                # Iterate over each target defined in p_ref.targets.
                for target_name, perf_target in p_ref.targets.items():
                    target_check = {}

                    # Check for ttft metric if defined.
                    if perf_target.ttft_ms is not None:
                        assert perf_target.ttft_ms > 0, (
                            f"ttft_ms for target '{target_name}' is not > 0: {perf_target.ttft_ms}"
                        )
                        ttft_ratio = res["mean_ttft_ms"] / perf_target.ttft_ms
                        check = ReportCheckTypes.from_result(
                            ttft_ratio < (1 + perf_target.tolerance)
                        )
                        target_check["ttft"] = perf_target.ttft_ms
                        target_check["ttft_ratio"] = ttft_ratio
                        target_check["ttft_check"] = check
                    else:
                        target_check["ttft_check"] = ReportCheckTypes.NA

                    # Check for tput_user metric if defined.
                    if perf_target.tput_user is not None:
                        assert perf_target.tput_user > 0, (
                            f"tput_user for target '{target_name}' is not > 0: {perf_target.tput_user}"
                        )
                        tput_user_ratio = res["mean_tps"] / perf_target.tput_user
                        check = ReportCheckTypes.from_result(
                            tput_user_ratio > (1 - perf_target.tolerance)
                        )
                        target_check["tput_user"] = perf_target.tput_user
                        target_check["tput_user_ratio"] = tput_user_ratio
                        target_check["tput_user_check"] = check
                    else:
                        target_check["tput_user_check"] = ReportCheckTypes.NA

                    # Check for tput metric if defined.
                    if perf_target.tput is not None:
                        assert perf_target.tput > 0, (
                            f"tput for target '{target_name}' is not > 0: {perf_target.tput}"
                        )
                        tput_ratio = res["tps_decode_throughput"] / perf_target.tput
                        check = ReportCheckTypes.from_result(
                            tput_ratio > (1 - perf_target.tolerance)
                        )
                        target_check["tput"] = perf_target.tput
                        target_check["tput_ratio"] = tput_ratio
                        target_check["tput_check"] = check
                    else:
                        target_check["tput_check"] = ReportCheckTypes.NA

                    # Save the computed checks under the target's name.
                    vlm_perf_results[p_ref_key]["target_checks"][target_name] = (
                        target_check
                    )

            else:
                # No result available from benchmark measurements.
                NA_STRING = "N/A"
                # In this case, add N/A for performance measures and an empty check dict per target.
                vlm_perf_results[p_ref_key].update(
                    {
                        "ttft": NA_STRING,
                        "tput_user": NA_STRING,
                        "tput": NA_STRING,
                        "target_checks": {
                            target_name: {
                                "ttft_check": ReportCheckTypes.NA,
                                "tput_user_check": ReportCheckTypes.NA,
                                "tput_check": ReportCheckTypes.NA,
                            }
                            for target_name in p_ref.targets.keys()
                        },
                    }
                )

        # build release performance benchmarking report for VLMs
        sorted_vlm_perf_results = {
            k: vlm_perf_results[k] for k in sorted(vlm_perf_results)
        }
        vlm_release_raw_targets = [v for k, v in sorted_vlm_perf_results.items()]

        flat_vlm_release_raw = flatten_target_checks(vlm_release_raw_targets)
        vlm_section = (
            f"#### VLM Benchmark Targets {model_spec.model_name} on {args.device}\n\n"
        )
        if vlm_release_raw_targets and vlm_release_raw_targets[0].get("target_checks"):
            vlm_section += benchmark_image_release_markdown(
                flat_vlm_release_raw,
                target_checks=vlm_release_raw_targets[0]["target_checks"],
            )
        else:
            vlm_section += benchmark_image_release_markdown(
                flat_vlm_release_raw, target_checks=None
            )
        release_sections.append(vlm_section)
    elif vlm_release_raw:
        # Show VLM benchmarks even without performance targets
        vlm_section = (
            f"#### VLM Benchmark Results {model_spec.model_name} on {args.device}\n\n"
        )
        vlm_section += "No performance targets defined for VLM benchmarks.\n\n"
        release_sections.append(vlm_section)

    # Combine sections or fallback to original behavior
    if release_sections:
        release_str = (
            f"### Performance Benchmark Targets {model_spec.model_name} on {args.device}\n\n"
            + "\n\n".join(release_sections)
        )
        # For backward compatibility, return the first section's data as release_raw
        if text_perf_refs:
            release_raw = (
                text_release_raw_targets
                if "text_release_raw_targets" in locals()
                else release_raw
            )
        elif vlm_perf_refs:
            release_raw = (
                vlm_release_raw_targets
                if "vlm_release_raw_targets" in locals()
                else release_raw
            )
    else:
        # Fallback to original behavior if no performance references exist
        release_str = f"### Performance Benchmark Targets {model_spec.model_name} on {args.device}\n\n"
        release_str += (
            "No performance targets defined for this model and device combination.\n"
        )

    return release_str, release_raw, disp_md_path, stats_file_path


def extract_eval_json_data(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = []

    results = data.get("results", {})
    configs = data.get("configs", {})

    first_key = list(results.keys())[0]

    for result_key, result_metrics in results.items():
        extracted_metrics = {
            k: v
            for k, v in result_metrics.items()
            if "alias" not in k and "_stderr" not in k
        }

        extracted.append({result_key: extracted_metrics})
    config = configs.get(first_key, {})
    task_name = config.get("task")
    if task_name is None:
        group_subtasks = data.get("group_subtasks")
        if group_subtasks:
            task_name = list(group_subtasks.keys())[0]
            config = configs.get(group_subtasks[task_name][0], {})

    if task_name != first_key:
        if first_key == "mmmu_val":
            task_name = "mmmu_val"

    dataset_path = config.get("dataset_path", "N/A")
    assert task_name == first_key, f"Task name mismatch: {task_name} != {first_key}"

    meta_data = {"task_name": task_name, "dataset_path": dataset_path}

    return extracted, meta_data


def extract_eval_results(files):
    results = {}
    meta_data = {}
    for json_file in files:
        # logger.info(f"Processing: {json_file}")
        res, meta = extract_eval_json_data(Path(json_file))
        _ = meta.pop("task_name", None)
        for task_dict in res:
            for specific_task_name, metrics in task_dict.items():
                results[specific_task_name] = metrics
                meta_data[specific_task_name] = meta

    return results, meta_data


def evals_release_report_data(args, results, meta_data, model_spec):
    eval_config = EVAL_CONFIGS[model_spec.model_name]

    report_rows = []

    for task in eval_config.tasks:
        if not task.score:
            logger.info(
                f"Skipping report for task:= {task.task_name}, no eval score is defined."
            )
            continue

        target_keys = []
        # Check for exact match (e.g. "meta_gpqa")
        if task.task_name in results:
            target_keys.append(task.task_name)
        else:
            # Check for subtasks (e.g. config says "longbench", results have "longbench_2wikimqa")
            prefix = f"{task.task_name}_"
            subtasks = [k for k in results if k.startswith(prefix)]
            target_keys.extend(sorted(subtasks))
        if target_keys:
            for t_key in target_keys:
                logger.info(f"eval processing task_name: {t_key}")

                # do NOT extract results[t_key] here.
                # The score_func expects the ROOT results dict so it can do results[task_name].

                kwargs = task.score.score_func_kwargs
                # Update task_name so the score function looks up the specific subtask (e.g. longbench_2wikimqa)
                kwargs["task_name"] = t_key
                configured_keys = kwargs.get("result_keys", [])
                actual_data = results.get(t_key, {})

                key_found = any(k in actual_data for k in configured_keys)

                if not key_found:
                    valid_candidates = [
                        k
                        for k, v in actual_data.items()
                        if isinstance(v, (int, float))
                        and "stderr" not in k
                        and "alias" not in k
                    ]

                    if valid_candidates:
                        logger.info(
                            f"  Metric mismatch for {t_key}. Auto-detected replacement: {valid_candidates[0]}"
                        )
                        kwargs["result_keys"] = [valid_candidates[0]]
                try:
                    score = task.score.score_func(
                        results, task_name=t_key, kwargs=kwargs
                    )
                except Exception as e:
                    logger.warning(f"  Could not calculate score for {t_key}: {e}")
                    score = 0.0
                if kwargs.get("unit") == "WER":
                    score = 100 - score

                if task.score.published_score:
                    assert task.score.published_score > 0, "Published score is not > 0"
                    ratio_to_published = score / task.score.published_score
                else:
                    ratio_to_published = "N/A"

                if task.score.gpu_reference_score:
                    assert task.score.gpu_reference_score > 0, (
                        "Reference score is not > 0"
                    )
                    ratio_to_reference = score / task.score.gpu_reference_score
                    accuracy_check = ReportCheckTypes.from_result(
                        ratio_to_reference >= (1.0 - task.score.tolerance)
                    )
                else:
                    ratio_to_reference = "N/A"
                    if task.score.published_score:
                        accuracy_check = ReportCheckTypes.from_result(
                            ratio_to_published >= (1.0 - task.score.tolerance)
                        )
                    else:
                        accuracy_check = ReportCheckTypes.NA

                report_rows.append(
                    {
                        "model": model_spec.model_name,
                        "device": args.device,
                        "task_name": t_key,
                        "accuracy_check": accuracy_check,
                        "score": score,
                        "ratio_to_reference": ratio_to_reference,
                        "gpu_reference_score": task.score.gpu_reference_score,
                        "gpu_reference_score_ref": task.score.gpu_reference_score_ref,
                        "ratio_to_published": ratio_to_published,
                        "published_score": task.score.published_score,
                        "published_score_ref": task.score.published_score_ref,
                        "metadata": meta_data.get(t_key),
                    }
                )
        else:
            score = "N/A"
            ratio_to_published = "N/A"
            ratio_to_reference = "N/A"
            accuracy_check = ReportCheckTypes.NA

            report_rows.append(
                {
                    "model": model_spec.model_name,
                    "device": args.device,
                    "task_name": task.task_name,
                    "accuracy_check": accuracy_check,
                    "score": score,
                    "ratio_to_reference": ratio_to_reference,
                    "gpu_reference_score": task.score.gpu_reference_score,
                    "gpu_reference_score_ref": task.score.gpu_reference_score_ref,
                    "ratio_to_published": ratio_to_published,
                    "published_score": task.score.published_score,
                    "published_score_ref": task.score.published_score_ref,
                    "metadata": meta_data.get(task.task_name),
                }
            )

    return report_rows


def generate_evals_release_markdown(report_rows):
    # Step 1: Convert all values to strings with proper formatting
    def format_value(key, value, row):
        if key == "published_score":
            # Format published_score as a hyperlink to published_score_ref
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref_val = row.get("published_score_ref", "")
            return f"[{score_val}]({ref_val})" if ref_val else score_val
        elif key == "gpu_reference_score":
            # Format gpu_reference_score as a hyperlink to gpu_reference_score_ref
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref_val = row.get("gpu_reference_score_ref", "")
            return f"[{score_val}]({ref_val})" if ref_val else score_val
        elif key == "accuracy_check":
            return ReportCheckTypes.to_display_string(value)
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    formatted_rows = [
        {k: format_value(k, v, row) for k, v in row.items()} for row in report_rows
    ]

    # Remove published_score_ref column from display
    remove_keys = ["published_score_ref", "metadata", "gpu_reference_score_ref"]
    headers = [h for h in formatted_rows[0].keys() if h not in remove_keys]

    # Step 2: Compute max width per column
    column_widths = {
        header: max(len(header), max(len(row[header]) for row in formatted_rows))
        for header in headers
    }

    # Step 3: Build table rows
    def format_row(row):
        return (
            "| " + " | ".join(f"{row[h]:<{column_widths[h]}}" for h in headers) + " |"
        )

    # Step 4: Build header and divider rows
    header_row = "| " + " | ".join(f"{h:<{column_widths[h]}}" for h in headers) + " |"
    divider_row = "|-" + "-|-".join("-" * column_widths[h] for h in headers) + "-|"

    row_strs = [format_row(row) for row in formatted_rows]

    explain_str = "\n\nNote: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score."

    markdown_str = (
        header_row + "\n" + divider_row + "\n" + "\n".join(row_strs) + explain_str
    )
    return markdown_str


def separate_files_by_format(files):
    """Separate eval files into dict-format and list-format.

    Detects JSON structure to differentiate between:
    - Dict format: {"results": {...}, "configs": {...}} (lmms-eval)
    - List format: [{...}] (image_client)

    Args:
        files: List of file paths to eval JSON files

    Returns:
        Tuple of (dict_format_files, list_format_files)
    """
    dict_format_files = []
    list_format_files = []

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                list_format_files.append(filepath)
            elif isinstance(data, dict):
                dict_format_files.append(filepath)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read or parse file {filepath}: {e}")

    return dict_format_files, list_format_files


def process_list_format_eval_files(list_files):
    """Process list-format JSON files from image_client.

    Extracts metrics from CNN image generation eval results.
    List format is: [{metric1: value1, metric2: value2, ...}]

    Args:
        list_files: List of file paths with list-format JSON

    Returns:
        Tuple of (results_dict, meta_data_dict) in the same format as extract_eval_results()
    """
    results = {}
    meta_data = {}

    for filepath in list_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # data is a list of dicts, typically with one element from image_client
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"List format file {filepath} is empty or invalid")
                continue

            # Extract the first dict from the list (image_client typically writes one)
            eval_data = data[0]

            # Extract task name if available
            task_name = eval_data.get("task_name", "image_generation")

            # Store metrics under task name
            if task_name not in results:
                results[task_name] = {}

            # Add all metrics from this eval data
            results[task_name].update(eval_data)

            # Store metadata
            if task_name not in meta_data:
                meta_data[task_name] = {
                    "task_name": task_name,
                    "dataset_path": eval_data.get("dataset_path", "N/A"),
                }
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not process list format file {filepath}: {e}")

    return results, meta_data


def evals_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    eval_run_id = f"{model_spec.model_id}"
    output_dir = Path(args.output_path) / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get file pattern based on model type
    if model_spec.model_type == ModelType.AUDIO:
        file_name_pattern = generate_audio_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.CNN:
        file_name_pattern = generate_cnn_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.IMAGE:
        file_name_pattern = generate_image_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.EMBEDDING:
        file_name_pattern = generate_embedding_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        file_name_pattern = generate_tts_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.VIDEO:
        file_name_pattern = generate_video_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    elif model_spec.model_type == ModelType.TEXT_TO_SPEECH:
        file_name_pattern = generate_tts_report_data(model_spec, eval_run_id)
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)
    else:
        # LLM models use results_*.json pattern
        file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/results_*.json"
        file_path_pattern = (
            f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
        )
        files = glob(file_path_pattern)

    if "image" in model_spec.supported_modalities:
        image_file_name_pattern = f"eval_{eval_run_id}/*_results.json"
        image_file_path_pattern = f"{get_default_workflow_root_log_dir()}/evals_output/{image_file_name_pattern}"
        image_files = glob(image_file_path_pattern)
        files.extend(image_files)
        image_file_name_pattern = f"eval_{eval_run_id}/{model_spec.hf_model_repo.replace('/', '__')}/*results.json"
        image_file_path_pattern = f"{get_default_workflow_root_log_dir()}/evals_output/{image_file_name_pattern}"
        logger.info(f"Image File Pattern: {image_file_path_pattern}")
        image_files = glob(image_file_path_pattern)
        logger.info(f"Image Files: {image_files}")
        files.extend(image_files)
    logger.info("Evaluations Summary")
    logger.info(f"Processing: {len(files)} files")
    if (
        model_spec.model_type.name == ModelType.CNN.name
        or model_spec.model_type.name == ModelType.IMAGE.name
        or model_spec.model_type.name == ModelType.EMBEDDING.name
        or model_spec.model_type.name == ModelType.VIDEO.name
        or model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name
    ):
        # TODO rewrite this
        data_fpath = data_dir / f"eval_data_{report_id}.json"

        # Combine files into one JSON
        combined_data = {}
        for i, file_path in enumerate(files):
            with open(file_path, "r") as f:
                file_data = json.load(f)
            combined_data = file_data

        # Write combined data to data_fpath
        with open(data_fpath, "w") as f:
            json.dump(combined_data, f, indent=4)

        release_str = (
            f"### Accuracy Evaluations for {model_spec.model_name} on {args.device}"
        )
        summary_fpath = output_dir / f"summary_{report_id}.md"
        with summary_fpath.open("w", encoding="utf-8") as f:
            f.write("MD summary to do")

        return release_str, combined_data, summary_fpath, data_fpath

    dict_format_files, list_format_files = separate_files_by_format(files)

    results = {}
    meta_data = {}

    if dict_format_files:
        dict_results, dict_meta_data = extract_eval_results(dict_format_files)
        results.update(dict_results)
        meta_data.update(dict_meta_data)
    if list_format_files:
        list_results, list_meta_data = process_list_format_eval_files(list_format_files)
        results.update(list_results)
        meta_data.update(list_meta_data)

    if not results:
        logger.warning("No evaluation files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )
    # generate release report
    report_rows = evals_release_report_data(args, results, meta_data, model_spec)

    # store results
    markdown_str = generate_evals_release_markdown(report_rows)

    release_str = f"### Accuracy Evaluations for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # generate summary report
    summary_fpath = output_dir / f"summary_{report_id}.md"
    summary_markdown_str = generate_evals_markdown_table(results, meta_data)
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(summary_markdown_str)

    # store raw data
    release_raw = report_rows
    data_fpath = data_dir / f"eval_data_{report_id}.json"

    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4)

    disp_md_path = summary_fpath
    data_file_path = data_fpath
    return release_str, release_raw, disp_md_path, data_file_path


def generate_tests_report(args, server_mode, model_spec, report_id, metadata={}):
    # glob on all test reports - each test category might produce its own report
    file_name_pattern = f"test_{model_spec.model_id}_*/*"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/tests_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"summary_{report_id}.md"

    logger.info("Tests Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
        logger.info("No tests report files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            None,
            None,
        )
    # TODO: Support handling of multiple test reports
    assert len(files) == 1, "Handling of multiple tests reports is unimplemented."
    files = files[0]

    # generate vLLM parameter coverage report
    markdown_str = generate_vllm_parameter_report(
        files, output_path, report_id, metadata, model_spec=model_spec
    )

    # Look for parameter_report.json in tests_output directory
    release_raw = None
    test_dir_pattern = f"test_{model_spec.model_id}_*"
    test_dir_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/tests_output/{test_dir_pattern}"
    )
    test_dirs = glob(test_dir_path_pattern)

    for test_dir in test_dirs:
        parameter_report_path = Path(test_dir) / "parameter_report.json"
        if parameter_report_path.exists():
            try:
                with open(parameter_report_path, "r", encoding="utf-8") as f:
                    release_raw = json.load(f)
                logger.info(f"Loaded parameter report from: {parameter_report_path}")
                break
            except Exception as e:
                logger.warning(
                    f"Could not read parameter report {parameter_report_path}: {e}"
                )

    if release_raw is None:
        logger.info("No parameter_report.json found in tests_output directory.")
        release_raw = [
            {
                "model": getattr(args, "model", "unknown_model"),
                "device": getattr(args, "device", "unknown_device"),
            }
        ]

    release_str = f"### Test Results for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # Write markdown report to file
    with output_path.open("w", encoding="utf-8") as f:
        f.write(release_str)
    logger.info(f"Tests report saved to: {output_path}")

    # Save raw data to data directory
    data_fpath = data_dir / f"tests_data_{report_id}.json"
    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4)
    logger.info(f"Tests data saved to: {data_fpath}")

    return release_str, release_raw, output_path, data_fpath


def generate_evals_markdown_table(results, meta_data) -> str:
    rows = []
    for task_name, metrics in results.items():
        for metric_name, metric_value in metrics.items():
            if metric_name and metric_name != " ":
                if not isinstance(
                    metric_value, float
                ):  # some metrics in image evals are not floats
                    continue
                rows.append((task_name, metric_name, f"{metric_value:.4f}"))

    if not rows:
        return "No evaluation results to display."
    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|"
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    return markdown


def generate_stress_tests_markdown_table(release_raw, model_config):
    """Generate markdown table for test results with mean values only (original format)."""

    # Define display columns: ISL, OSL, Concurrency, Num Prompts
    # Then mean values for TTFT, TPOT, ITL, E2EL
    # Then throughput metrics
    display_cols = [
        # Configuration columns
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("num_prompts", "Num Prompts"),
        # Mean metrics only (original format)
        ("ttft", "TTFT (ms)"),
        ("tpot", "TPOT (ms)"),
        ("itl", "ITL (ms)"),
        ("e2el", "E2EL (ms)"),
        # Throughput metrics at the end
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]

    NOT_MEASURED_STR = "N/A"

    # Define decimal formatting standards
    decimal_places_map = {
        "ISL": 0,
        "OSL": 0,
        "Concurrency": 0,
        "Num Prompts": 0,
        "TTFT (ms)": 1,
        "TPOT (ms)": 1,
        "ITL (ms)": 1,
        "E2EL (ms)": 1,
        "Tput User (TPS)": 2,
        "Tput Decode (TPS)": 1,
    }

    display_dicts = []

    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            if col_name == "isl":
                value = row.get("input_sequence_length", NOT_MEASURED_STR)
            elif col_name == "osl":
                value = row.get("output_sequence_length", NOT_MEASURED_STR)
            elif col_name == "max_concurrency":
                value = row.get("max_con", NOT_MEASURED_STR)
            elif col_name == "num_prompts":
                value = row.get("num_prompts", NOT_MEASURED_STR)
            elif col_name == "ttft":
                value = row.get("mean_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "tpot":
                value = row.get("mean_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "itl":
                value = row.get("mean_itl_ms", NOT_MEASURED_STR)
            elif col_name == "e2el":
                value = row.get("mean_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "tput_user":
                value = row.get("mean_tps", NOT_MEASURED_STR)
            elif col_name == "tput":
                value = row.get("tps_decode_throughput", NOT_MEASURED_STR)
            else:
                value = row.get(col_name, NOT_MEASURED_STR)

            # Format numeric values with consistent decimal places for proper alignment
            if value == NOT_MEASURED_STR or value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, (int, float)) and not (
                isinstance(value, float) and (value != value)
            ):  # Check for NaN
                decimal_places = decimal_places_map.get(display_header, 2)
                if decimal_places == 0:
                    # Format as integer
                    row_dict[display_header] = str(int(value))
                else:
                    # Format as float with specified decimal places
                    row_dict[display_header] = f"{float(value):.{decimal_places}f}"
            else:
                # Handle string numbers or other formats
                try:
                    numeric_value = float(value)
                    decimal_places = decimal_places_map.get(display_header, 2)
                    if decimal_places == 0:
                        row_dict[display_header] = str(int(numeric_value))
                    else:
                        row_dict[display_header] = f"{numeric_value:.{decimal_places}f}"
                except (ValueError, TypeError):
                    row_dict[display_header] = str(value)

        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def generate_stress_tests_markdown_table_detailed(release_raw, model_config):
    """Generate detailed markdown table with percentile statistics for test results."""

    # Define display columns in requested order:
    # ISL, OSL, Concurrency, Num Prompts
    # Then for each metric (ttft, tpot, itl, e2el): mean, p05, p25, p50, p95, p99
    # Then throughput metrics
    display_cols = [
        # Configuration columns
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("num_prompts", "Num Prompts"),
        # TTFT metrics: mean, p05, p25, p50, p95, p99
        ("ttft", "TTFT (ms)"),
        ("p5_ttft", "P5 TTFT (ms)"),
        ("p25_ttft", "P25 TTFT (ms)"),
        ("p50_ttft", "P50 TTFT (ms)"),
        ("p95_ttft", "P95 TTFT (ms)"),
        ("p99_ttft", "P99 TTFT (ms)"),
        # TPOT metrics: mean, p05, p25, p50, p95, p99
        ("tpot", "TPOT (ms)"),
        ("p5_tpot", "P5 TPOT (ms)"),
        ("p25_tpot", "P25 TPOT (ms)"),
        ("p50_tpot", "P50 TPOT (ms)"),
        ("p95_tpot", "P95 TPOT (ms)"),
        ("p99_tpot", "P99 TPOT (ms)"),
        # ITL metrics: mean, p05, p25, p50, p95, p99
        ("itl", "ITL (ms)"),
        ("p5_itl", "P5 ITL (ms)"),
        ("p25_itl", "P25 ITL (ms)"),
        ("p50_itl", "P50 ITL (ms)"),
        ("p95_itl", "P95 ITL (ms)"),
        ("p99_itl", "P99 ITL (ms)"),
        # E2EL metrics: mean, p05, p25, p50, p95, p99
        ("e2el", "E2EL (ms)"),
        ("p5_e2el", "P5 E2EL (ms)"),
        ("p25_e2el", "P25 E2EL (ms)"),
        ("p50_e2el", "P50 E2EL (ms)"),
        ("p95_e2el", "P95 E2EL (ms)"),
        ("p99_e2el", "P99 E2EL (ms)"),
        # Throughput metrics at the end
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]

    NOT_MEASURED_STR = "N/A"

    # Define decimal formatting standards based on benchmarking standards
    decimal_places_map = {
        "ISL": 0,
        "OSL": 0,
        "Concurrency": 0,
        "Num Prompts": 0,
        # TTFT
        "TTFT (ms)": 1,
        "P5 TTFT (ms)": 1,
        "P25 TTFT (ms)": 1,
        "P50 TTFT (ms)": 1,
        "P95 TTFT (ms)": 1,
        "P99 TTFT (ms)": 1,
        # TPOT
        "TPOT (ms)": 1,
        "P5 TPOT (ms)": 1,
        "P25 TPOT (ms)": 1,
        "P50 TPOT (ms)": 1,
        "P95 TPOT (ms)": 1,
        "P99 TPOT (ms)": 1,
        # ITL
        "ITL (ms)": 1,
        "P5 ITL (ms)": 1,
        "P25 ITL (ms)": 1,
        "P50 ITL (ms)": 1,
        "P95 ITL (ms)": 1,
        "P99 ITL (ms)": 1,
        # E2EL
        "E2EL (ms)": 1,
        "P5 E2EL (ms)": 1,
        "P25 E2EL (ms)": 1,
        "P50 E2EL (ms)": 1,
        "P95 E2EL (ms)": 1,
        "P99 E2EL (ms)": 1,
        # Throughput
        "Tput User (TPS)": 2,
        "Tput Decode (TPS)": 1,
    }

    display_dicts = []

    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            if col_name == "isl":
                value = row.get("input_sequence_length", NOT_MEASURED_STR)
            elif col_name == "osl":
                value = row.get("output_sequence_length", NOT_MEASURED_STR)
            elif col_name == "max_concurrency":
                value = row.get("max_con", NOT_MEASURED_STR)
            elif col_name == "num_prompts":
                value = row.get("num_prompts", NOT_MEASURED_STR)

            # TTFT metrics
            elif col_name == "ttft":
                value = row.get("mean_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p5_ttft":
                value = row.get("p5_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p25_ttft":
                value = row.get("p25_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p50_ttft":
                value = row.get("p50_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p95_ttft":
                value = row.get("p95_ttft_ms", NOT_MEASURED_STR)
            elif col_name == "p99_ttft":
                value = row.get("p99_ttft_ms", NOT_MEASURED_STR)

            # TPOT metrics
            elif col_name == "tpot":
                value = row.get("mean_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p5_tpot":
                value = row.get("p5_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p25_tpot":
                value = row.get("p25_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p50_tpot":
                value = row.get("p50_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p95_tpot":
                value = row.get("p95_tpot_ms", NOT_MEASURED_STR)
            elif col_name == "p99_tpot":
                value = row.get("p99_tpot_ms", NOT_MEASURED_STR)

            # ITL metrics
            elif col_name == "itl":
                value = row.get("mean_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p5_itl":
                value = row.get("p5_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p25_itl":
                value = row.get("p25_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p50_itl":
                value = row.get("p50_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p95_itl":
                value = row.get("p95_itl_ms", NOT_MEASURED_STR)
            elif col_name == "p99_itl":
                value = row.get("p99_itl_ms", NOT_MEASURED_STR)

            # E2EL metrics
            elif col_name == "e2el":
                value = row.get("mean_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p5_e2el":
                value = row.get("p5_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p25_e2el":
                value = row.get("p25_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p50_e2el":
                value = row.get("p50_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p95_e2el":
                value = row.get("p95_e2el_ms", NOT_MEASURED_STR)
            elif col_name == "p99_e2el":
                value = row.get("p99_e2el_ms", NOT_MEASURED_STR)

            # Throughput metrics
            elif col_name == "tput_user":
                value = row.get("mean_tps", NOT_MEASURED_STR)
            elif col_name == "tput":
                value = row.get("tps_decode_throughput", NOT_MEASURED_STR)

            else:
                value = row.get(col_name, NOT_MEASURED_STR)

            # Format numeric values with consistent decimal places for proper alignment
            if value == NOT_MEASURED_STR or value is None or value == "":
                row_dict[display_header] = NOT_MEASURED_STR
            elif isinstance(value, (int, float)) and not (
                isinstance(value, float) and (value != value)
            ):  # Check for NaN
                decimal_places = decimal_places_map.get(display_header, 2)
                if decimal_places == 0:
                    # Format as integer
                    row_dict[display_header] = str(int(value))
                else:
                    # Format as float with specified decimal places
                    row_dict[display_header] = f"{float(value):.{decimal_places}f}"
            else:
                # Handle string numbers or other formats
                try:
                    numeric_value = float(value)
                    decimal_places = decimal_places_map.get(display_header, 2)
                    if decimal_places == 0:
                        row_dict[display_header] = str(int(numeric_value))
                    else:
                        row_dict[display_header] = f"{numeric_value:.{decimal_places}f}"
                except (ValueError, TypeError):
                    row_dict[display_header] = str(value)

        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def stress_test_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    """Generate stress test report using stress_tests-specific summary report module."""
    file_name_pattern = f"stress_test_{model_spec.model_id}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/stress_tests_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "stress_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Stress Tests Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
        logger.info("No stress test files found. Skipping.")
        return "", None, None, None

    # Use the stress_tests-specific generate_report function
    release_str, release_raw, disp_md_path, stats_file_path = (
        stress_test_generate_report_helper(files, output_dir, report_id, metadata)
    )

    # Generate stress test-specific release report
    # Build stress test performance report
    stress_test_release_str = (
        f"### Stress Test Results for {model_spec.model_name} on {args.device}\n\n"
    )

    if release_raw:
        # Check if percentile report is requested
        percentile_report = getattr(args, "percentile_report", False)

        # Create stress test-specific markdown table (detailed or simple format)
        if percentile_report:
            logger.info("Generating detailed percentile report for stress tests")
            stress_test_markdown = generate_stress_tests_markdown_table_detailed(
                release_raw, model_spec
            )
        else:
            logger.info(
                "Generating simplified report for stress tests (use --percentile-report for detailed statistics)"
            )
            stress_test_markdown = generate_stress_tests_markdown_table(
                release_raw, model_spec
            )

        stress_test_release_str += stress_test_markdown
    else:
        stress_test_release_str += (
            "No stress test results found for this model and device combination.\n"
        )

    # Save stress test-specific summary
    summary_fpath = output_dir / f"stress_test_summary_{report_id}.md"
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(stress_test_release_str)

    # Save raw data
    data_fpath = data_dir / f"stress_test_data_{report_id}.json"
    with data_fpath.open("w", encoding="utf-8") as f:
        json.dump(release_raw, f, indent=4, default=str)

    return stress_test_release_str, release_raw, summary_fpath, data_fpath


def benchmarks_release_data_format(model_spec, device_str, benchmark_summary_data):
    """Convert the benchmark release data to the desired format"""
    reformated_benchmarks_release_data = []

    benchmark_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model": model_spec.model_name,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "backend": model_spec.model_type.name.lower(),
        "device": device_str,
        "num_requests": benchmark_summary_data.get("num_requests", 1),
        "num_inference_steps": benchmark_summary_data.get("num_inference_steps", 0),
        "ttft": benchmark_summary_data.get("mean_ttft_ms", 0) / 1000,
        "inference_steps_per_second": benchmark_summary_data.get(
            "inference_steps_per_second", 0
        ),
        "filename": benchmark_summary_data.get("filename", ""),
        "task_type": model_spec.model_type.name.lower(),
    }

    if (
        model_spec.model_type.name == ModelType.CNN.name
        or model_spec.model_type.name == ModelType.IMAGE.name
        or model_spec.model_type.name == ModelType.VIDEO.name
    ):
        benchmark_summary["tput_user"] = benchmark_summary_data.get("tput_user", 0)

    if model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
        benchmark_summary["ttft_p90"] = (
            benchmark_summary_data.get("p90_ttft_ms", 0) / 1000
        )
        benchmark_summary["ttft_p95"] = (
            benchmark_summary_data.get("p95_ttft_ms", 0) / 1000
        )
        benchmark_summary["rtr"] = benchmark_summary_data.get("rtr", 0)

    # Add Whisper-specific fields only for Whisper models
    if "whisper" in model_spec.hf_model_repo.lower():
        # Create a simple object that mimics what the utility functions expect
        class ModelSpecWrapper:
            def __init__(self, model_spec):
                self.model_spec = model_spec

        wrapper = ModelSpecWrapper(model_spec)
        streaming_enabled = is_streaming_enabled_for_whisper(wrapper)
        preprocessing_enabled = is_preprocessing_enabled_for_whisper(wrapper)

        benchmark_summary["streaming_enabled"] = streaming_enabled
        benchmark_summary["preprocessing_enabled"] = preprocessing_enabled

    reformated_benchmarks_release_data.append(benchmark_summary)
    return reformated_benchmarks_release_data


def benchmarks_release_data_format_embedding(
    model_spec, device_str, benchmark_summary_data
):
    """Convert the benchmark release data to the desired format for EMBEDDING models"""

    return [
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "model": model_spec.model_name,
            "model_name": model_spec.model_name,
            "model_id": model_spec.model_id,
            "backend": model_spec.model_type.name.lower(),
            "device": device_str,
            "num_requests": benchmark_summary_data.get("num_requests", 1),
            "ISL": benchmark_summary_data.get("input_sequence_length", 0),
            "concurrency": benchmark_summary_data.get("max_con", 0),
            "tput_user": benchmark_summary_data.get("mean_tps", 0),
            "tput_prefill": benchmark_summary_data.get("tps_prefill_throughput", 0),
            "e2el_ms": benchmark_summary_data.get("mean_e2el_ms", 0),
            "filename": benchmark_summary_data.get("filename", ""),
            "task_type": model_spec.model_type.name.lower(),
        }
    ]


def add_target_checks_cnn_image_video(
    targets, evals_release_data, benchmark_summary_data, metrics
):
    """Add target checks for CNN, IMAGE and VIDEO models based on evals and benchmark data."""
    logger.info("Adding target_checks to CNN, IMAGE and VIDEO benchmark release data")
    tput_user = evals_release_data[0].get("tput_user", 0) if evals_release_data else 0
    benchmark_summary_data["tput_user"] = tput_user

    # extract targets for functional, complete, target and calculate them
    target_tput_user = targets.tput_user
    complete_tput_user = target_tput_user / 2  # Complete target is 2x slower
    functional_tput_user = target_tput_user / 10  # Functional target is 10x slower

    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "ttft": metrics["functional_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["functional_ttft_ratio"],
            "ttft_check": metrics["functional_ttft_check"],
            "tput_check": 2 if tput_user > functional_tput_user else 3,
        },
        "complete": {
            "ttft": metrics["complete_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["complete_ttft_ratio"],
            "ttft_check": metrics["complete_ttft_check"],
            "tput_check": 2 if tput_user > complete_tput_user else 3,
        },
        "target": {
            "ttft": metrics["target_ttft"] / 1000,  # Convert ms to seconds
            "ttft_ratio": metrics["target_ttft_ratio"],
            "ttft_check": metrics["target_ttft_check"],
            "tput_check": 2 if tput_user > target_tput_user else 3,
        },
    }

    return target_checks


def add_target_checks_embedding(metrics):
    """Add target checks for EMBEDDING models based on evals and benchmark data."""
    logger.info("Adding target_checks to EMBEDDING benchmark release data")

    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "tput_user": metrics["functional_tput_user"],
            "tput_user_ratio": metrics["functional_tput_user_ratio"],
            "tput_user_check": metrics["functional_tput_user_check"],
            "tput_prefill": metrics["functional_tput_prefill"],
            "tput_prefill_ratio": metrics["functional_tput_prefill_ratio"],
            "tput_prefill_check": metrics["functional_tput_prefill_check"],
            "e2el_ms": metrics["functional_e2el_ms"],
            "e2el_ms_ratio": metrics["functional_e2el_ms_ratio"],
            "e2el_ms_check": metrics["functional_e2el_ms_check"],
        },
        "complete": {
            "tput_user": metrics["complete_tput_user"],
            "tput_user_ratio": metrics["complete_tput_user_ratio"],
            "tput_user_check": metrics["complete_tput_user_check"],
            "tput_prefill": metrics["complete_tput_prefill"],
            "tput_prefill_ratio": metrics["complete_tput_prefill_ratio"],
            "tput_prefill_check": metrics["complete_tput_prefill_check"],
            "e2el_ms": metrics["complete_e2el_ms"],
            "e2el_ms_ratio": metrics["complete_e2el_ms_ratio"],
            "e2el_ms_check": metrics["complete_e2el_ms_check"],
        },
        "target": {
            "tput_user": metrics["target_tput_user"],
            "tput_user_ratio": metrics["target_tput_user_ratio"],
            "tput_user_check": metrics["target_tput_user_check"],
            "tput_prefill": metrics["target_tput_prefill"],
            "tput_prefill_ratio": metrics["target_tput_prefill_ratio"],
            "tput_prefill_check": metrics["target_tput_prefill_check"],
            "e2el_ms": metrics["target_e2el_ms"],
            "e2el_ms_ratio": metrics["target_e2el_ms_ratio"],
            "e2el_ms_check": metrics["target_e2el_ms_check"],
        },
    }

    return target_checks


def add_target_checks_video(metrics):
    """Add target checks for VIDEO models based on evals and benchmark data."""
    logger.info("Adding target_checks to VIDEO benchmark release data")
    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "concurrency": metrics["functional_concurrency"],
            "concurrency_ratio": metrics["functional_concurrency_ratio"],
            "concurrency_check": metrics["functional_concurrency_check"],
        },
        "complete": {
            "concurrency": metrics["complete_concurrency"],
            "concurrency_ratio": metrics["complete_concurrency_ratio"],
            "concurrency_check": metrics["complete_concurrency_check"],
        },
        "target": {
            "concurrency": metrics["target_concurrency"],
            "concurrency_ratio": metrics["target_concurrency_ratio"],
            "concurrency_check": metrics["target_concurrency_check"],
        },
    }

    return target_checks


def calculate_target_metrics(metrics_config):
    """Calculate metrics for functional, complete, and target thresholds.

    Args:
        metrics_config: List of metric configurations. Each config is a dict with:
            - avg_metric: Average metric from benchmark results
            - target_metric: Target metric from performance reference
            - field_name: Name of the metric field
            - is_ascending_metric: If True, higher values are preffered (e.g., throughput).
            If False, lower values are preffered (e.g., latency, TTFT).

    Returns:
        Dict containing metrics for all target levels (functional, complete, target)
    """

    def get_metric_ratio_and_check(avg_metric, ref_metric, is_ascending_metric):
        if not ref_metric:
            return "Undefined", "Undefined"
        ratio = avg_metric / ref_metric
        if is_ascending_metric:
            check = 2 if ratio > 1.0 else 3
        else:
            check = 2 if ratio < 1.0 else 3
        return ratio, check

    # Define target level multipliers
    target_multipliers = {
        "functional": FUNCTIONAL_TARGET,  # 10x slower than target
        "complete": COMPLETE_TARGET,  # 2x slower than target
        "target": 1,  # actual target
    }

    metrics = {}
    for config in metrics_config:
        avg_metric = config["avg_metric"]
        target_metric = config["target_metric"]
        field_name = config["field_name"]
        is_ascending_metric = config.get("is_ascending_metric", False)

        # Skip if target_metric is None (e.g., for TTS when target_rtr is not set)
        if target_metric is None:
            logger.warning(
                f"Skipping metric calculation for {field_name}: target_metric is None"
            )
            continue

        for level, multiplier in target_multipliers.items():
            if is_ascending_metric:
                level_metric = target_metric / multiplier
            else:
                level_metric = target_metric * multiplier

            ratio, check = get_metric_ratio_and_check(
                avg_metric, level_metric, is_ascending_metric
            )

            metrics[f"{level}_{field_name}"] = level_metric
            metrics[f"{level}_{field_name}_ratio"] = ratio
            metrics[f"{level}_{field_name}_check"] = check

    return metrics


def add_target_checks_audio(metrics):
    logger.info("Adding target_checks to Audio benchmark release data")
    # tput_check is always 1 for now (no tput target)
    tput_check = 1
    target_checks = {
        "functional": {
            "ttft": metrics["functional_ttft"],
            "ttft_ratio": metrics["functional_ttft_ratio"],
            "ttft_check": metrics["functional_ttft_check"],
            "tput_check": tput_check,
        },
        "complete": {
            "ttft": metrics["complete_ttft"],
            "ttft_ratio": metrics["complete_ttft_ratio"],
            "ttft_check": metrics["complete_ttft_check"],
            "tput_check": tput_check,
        },
        "target": {
            "ttft": metrics["target_ttft"],
            "ttft_ratio": metrics["target_ttft_ratio"],
            "ttft_check": metrics["target_ttft_check"],
            "tput_check": tput_check,
        },
    }

    return target_checks


def add_target_checks_tts(metrics):
    logger.info("Adding target_checks to TTS benchmark release data")
    # tput_check is always 1 for now (no tput target)
    tput_check = 1
    target_checks = {
        "functional": {
            "ttft": metrics.get("functional_ttft"),
            "ttft_ratio": metrics.get("functional_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("functional_ttft_check", "Undefined"),
            "rtr_check": metrics.get("functional_rtr_check", 1),
            "tput_check": tput_check,
        },
        "complete": {
            "ttft": metrics.get("complete_ttft"),
            "ttft_ratio": metrics.get("complete_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("complete_ttft_check", "Undefined"),
            "rtr_check": metrics.get("complete_rtr_check", 1),
            "tput_check": tput_check,
        },
        "target": {
            "ttft": metrics.get("target_ttft"),
            "ttft_ratio": metrics.get("target_ttft_ratio", "Undefined"),
            "ttft_check": metrics.get("target_ttft_check", "Undefined"),
            "rtr_check": metrics.get("target_rtr_check", 1),
            "tput_check": tput_check,
        },
    }

    return target_checks


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    model = cli_args.get("model")
    device_str = cli_args.get("device")
    docker_server = cli_args.get("docker_server", False)

    workflow_config = WORKFLOW_REPORT_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    device = DeviceTypes.from_string(device_str)
    assert device == model_spec.device_type

    server_mode = "API"
    command_flag = ""
    if docker_server:
        server_mode = "docker"
        command_flag = "--docker-server"

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_id = f"{model_spec.model_id}_{run_timestamp}"

    # only show the impl run command if non-default impl is used
    if model_spec.device_model_spec.default_impl:
        run_cmd = f"python run.py --model {model} --device {device_str} --workflow release {command_flag}"
    else:
        run_cmd = f"python run.py --model {model} --device {device_str} --impl {model_spec.impl.impl_name} --workflow release {command_flag}"

    metadata = {
        "report_id": report_id,
        "model_name": model_spec.model_name,
        "model_id": model_spec.model_id,
        "model_spec_json": args.model_spec_json,
        "model_repo": model_spec.hf_model_repo,
        "model_impl": model_spec.impl.impl_name,
        "inference_engine": model_spec.inference_engine,
        "device": device_str,
        "server_mode": server_mode,
        "tt_metal_commit": model_spec.tt_metal_commit,
        "vllm_commit": model_spec.vllm_commit,
        "run_command": run_cmd,
    }

    json_str = json.dumps(metadata, indent=4)
    metadata_str = f"### Metadata: {model_spec.model_name} on {device_str}\n```json\n{json_str}\n```"

    # Create a simple args object for the report generation functions
    class SimpleArgs:
        def __init__(
            self, output_path, model, device, model_spec_json, percentile_report=False
        ):
            self.output_path = output_path
            self.model = model
            self.device = device
            self.model_spec_json = model_spec_json
            self.percentile_report = percentile_report

    # Extract percentile_report flag from cli_args
    percentile_report = cli_args.get("percentile_report", False)

    simple_args = SimpleArgs(
        args.output_path,
        model,
        device_str,
        args.model_spec_json,
        percentile_report=percentile_report,
    )

    # generate vLLM benchmarks report
    (
        benchmarks_release_str,
        benchmarks_release_data,
        benchmarks_disp_md_path,
        benchmarks_data_file_path,
    ) = benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate AIPerf benchmarks report (separate detailed report)
    (
        aiperf_release_str,
        aiperf_release_data,
        aiperf_disp_md_path,
        aiperf_data_file_path,
    ) = aiperf_benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate GenAI-Perf benchmarks report (separate detailed report)
    (
        genai_perf_release_str,
        genai_perf_release_data,
        genai_perf_disp_md_path,
        genai_perf_data_file_path,
    ) = genai_perf_benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate evals report
    evals_release_str, evals_release_data, evals_disp_md_path, evals_data_file_path = (
        evals_generate_report(
            simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
        )
    )

    # generate tests report
    (
        tests_release_str,
        tests_release_data,
        tests_disp_md_path,
        tests_data_file_path,
    ) = generate_tests_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )
    # generate stress test report
    (
        stress_tests_release_str,
        stress_tests_release_data,
        stress_tests_disp_md_path,
        stress_tests_data_file_path,
    ) = stress_test_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate server tests report
    server_tests_release_str, server_tests_release_data = server_tests_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # Collect benchmark display content
    benchmarks_disp_md_str = ""
    try:
        if benchmarks_disp_md_path:
            with open(benchmarks_disp_md_path, "r", encoding="utf-8") as f:
                benchmarks_disp_md_str = f.read()
    except (TypeError, FileNotFoundError):
        pass

    logging.info("Release Summary\n\n")

    release_header = (
        f"## Tenstorrent Model Release Summary: {model_spec.model_name} on {device_str}"
    )

    # Combine all benchmark sections
    all_benchmarks_str = ""
    if benchmarks_disp_md_str:
        all_benchmarks_str += benchmarks_disp_md_str + "\n\n"
    if benchmarks_release_str:
        all_benchmarks_str += benchmarks_release_str + "\n\n"
    if aiperf_release_str:
        all_benchmarks_str += aiperf_release_str + "\n\n"
    if genai_perf_release_str:
        all_benchmarks_str += genai_perf_release_str + "\n\n"

    release_str = f"{release_header}\n\n{metadata_str}\n\n{all_benchmarks_str}{evals_release_str}\n\n{tests_release_str}\n\n{stress_tests_release_str}\n\n{server_tests_release_str}"
    print(release_str)
    # save to file
    release_output_dir = Path(args.output_path) / "release"
    release_output_dir.mkdir(parents=True, exist_ok=True)
    release_data_dir = release_output_dir / "data"
    release_data_dir.mkdir(parents=True, exist_ok=True)
    release_file = release_output_dir / f"report_{report_id}.md"
    raw_file = release_data_dir / f"report_data_{report_id}.json"
    with release_file.open("w", encoding="utf-8") as f:
        f.write(release_str)

    with raw_file.open("w", encoding="utf-8") as f:
        # Read detailed benchmark statistics from CSV if available
        benchmarks_detailed_data = None
        if benchmarks_data_file_path:
            try:
                with open(benchmarks_data_file_path, "r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    benchmarks_detailed_data = list(csv_reader)
            except Exception as e:
                logger.warning(f"Could not read benchmark CSV data: {e}")

        # Check for server tests JSON files
        server_tests_data = []

        # Use tests_release_data for parameter_support_tests
        parameter_support_tests_data = tests_release_data if tests_release_data else []

        # Add target_checks for specific model if applicable
        if (
            model_spec.model_type.name == ModelType.CNN.name
            or model_spec.model_type.name == ModelType.IMAGE.name
            or model_spec.model_type.name == ModelType.AUDIO.name
            or model_spec.model_type.name == ModelType.VIDEO.name
            or model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name
        ):
            # Get performance targets using the shared utility
            # Extract the device we are running on
            device_str = cli_args.get("device").lower()
            targets = get_performance_targets(
                model_spec.model_name,
                device_str,
                model_type=model_spec.model_type.name,
            )
            logger.info(f"Performance targets: {targets}")

            # extract targets for functional, complete, target and calculate them
            target_ttft = targets.ttft_ms
            target_rtr = targets.rtr if hasattr(targets, "rtr") else None

            # Initialize the benchmark summary data
            benchmark_summary_data = {}

            # Aggregate mean_ttft_ms and inference_steps_per_second across all benchmarks
            total_ttft = 0.0
            total_tput = 0.0
            total_rtr = 0.0
            for benchmark in benchmarks_release_data:
                total_ttft += benchmark.get("mean_ttft_ms", 0)
                total_tput += benchmark.get("inference_steps_per_second", 0)
                # Aggregate RTR for TTS models
                if model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
                    total_rtr += benchmark.get("rtr", 0)
                benchmark_summary_data["num_requests"] = benchmark.get(
                    "num_requests", 0
                )
                benchmark_summary_data["num_inference_steps"] = benchmark.get(
                    "num_inference_steps", 0
                )
                benchmark_summary_data["inference_steps_per_second"] = benchmark.get(
                    "inference_steps_per_second", 0
                )
                benchmark_summary_data["filename"] = benchmark.get("filename", "")
                benchmark_summary_data["mean_ttft_ms"] = benchmark.get(
                    "mean_ttft_ms", 0
                )

            avg_ttft = (
                total_ttft / len(benchmarks_release_data)
                if len(benchmarks_release_data) > 0
                else 0
            )

            # For TTS, also calculate average RTR
            avg_rtr = None
            if model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
                avg_rtr = (
                    total_rtr / len(benchmarks_release_data)
                    if len(benchmarks_release_data) > 0
                    else 0
                )

            # Calculate all target metrics using centralized function
            # TTFT: lower is better, so is_ascending_metric=False
            metrics_config = [
                {
                    "avg_metric": avg_ttft,
                    "target_metric": target_ttft,
                    "field_name": "ttft",
                    "is_ascending_metric": False,
                },
            ]

            # For TTS, also calculate RTR metrics if target is available
            if (
                model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name
                and target_rtr is not None
                and avg_rtr is not None
            ):
                metrics_config.append(
                    {
                        "avg_metric": avg_rtr,
                        "target_metric": target_rtr,
                        "field_name": "rtr",
                        "is_ascending_metric": True,  # RTR: higher is better
                    }
                )

            metrics = calculate_target_metrics(metrics_config)

            target_checks = {}
            if (
                model_spec.model_type.name == ModelType.CNN.name
                or model_spec.model_type.name == ModelType.IMAGE.name
                or model_spec.model_type.name == ModelType.VIDEO.name
            ):
                logger.info(
                    "Adding target_checks for tput_user to CNN, IMAGE and VIDEO benchmark release data"
                )
                target_checks = add_target_checks_cnn_image_video(
                    targets,
                    evals_release_data,
                    benchmark_summary_data,
                    metrics,
                )
            elif model_spec.model_type.name == ModelType.AUDIO.name:
                logger.info("Adding target_checks for Audio benchmark release data")
                target_checks = add_target_checks_audio(metrics)
            elif model_spec.model_type.name == ModelType.TEXT_TO_SPEECH.name:
                logger.info("Adding target_checks for TTS benchmark release data")
                target_checks = add_target_checks_tts(metrics)
            else:
                logger.warning(f"Unknown model type: {model_spec.model_type.name}")
                target_checks = add_target_checks_audio(metrics)

            # Make sure benchmarks_release_data is of proper format for CNN and IMAGE
            benchmarks_release_data = benchmarks_release_data_format(
                model_spec, device_str, benchmark_summary_data
            )

            # Add target_checks to the existing benchmark object
            if benchmarks_release_data:
                benchmarks_release_data[0]["target_checks"] = target_checks

        elif model_spec.model_type.name == ModelType.EMBEDDING.name:
            # Get performance targets using the shared utility
            # Extract the device we are running on
            device_str = cli_args.get("device").lower()
            targets = get_performance_targets(
                model_spec.model_name,
                device_str,
                model_type=model_spec.model_type.name,
            )
            logger.info(f"Performance targets: {targets}")

            benchmark_summary_data = benchmarks_release_data[0]

            avg_tput_user = benchmark_summary_data.get("mean_tps", 0)
            avg_tput_prefill = benchmark_summary_data.get("tps_prefill_throughput", 0)
            avg_e2el_ms = benchmark_summary_data.get("mean_e2el_ms", 0)

            metrics = calculate_target_metrics(
                [
                    {
                        "avg_metric": avg_tput_user,
                        "target_metric": targets.tput_user,
                        "field_name": "tput_user",
                        "is_ascending_metric": True,
                    },
                    {
                        "avg_metric": avg_tput_prefill,
                        "target_metric": targets.tput_prefill,
                        "field_name": "tput_prefill",
                        "is_ascending_metric": True,
                    },
                    {
                        "avg_metric": avg_e2el_ms,
                        "target_metric": targets.e2el_ms,
                        "field_name": "e2el_ms",
                        "is_ascending_metric": False,
                    },
                ]
            )

            logger.info("Adding target_checks for Embedding benchmark release data")
            target_checks = add_target_checks_embedding(
                metrics,
            )

            benchmarks_release_data = benchmarks_release_data_format_embedding(
                model_spec, device_str, benchmark_summary_data
            )

            if benchmarks_release_data:
                benchmarks_release_data[0]["target_checks"] = target_checks

        # Read AIPerf benchmark data if available
        aiperf_detailed_data = None
        if aiperf_data_file_path:
            try:
                with open(aiperf_data_file_path, "r", encoding="utf-8") as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    aiperf_detailed_data = list(csv_reader)
            except Exception as e:
                logger.warning(f"Could not read AIPerf CSV data: {e}")

        # Read server tests data if available
        server_tests_data = []
        server_tests_path = Path(project_root) / "test_reports"
        if server_tests_path.exists():
            server_tests_json_files = list(server_tests_path.glob("*.json"))
            if server_tests_json_files:
                logger.info(
                    f"Found {len(server_tests_json_files)} server test report(s)"
                )
                for json_file in server_tests_json_files:
                    try:
                        with open(json_file, "r", encoding="utf-8") as test_file:
                            test_data = json.load(test_file)
                            server_tests_data.append(test_data)
                    except Exception as e:
                        logger.warning(
                            f"Could not read server test file {json_file}: {e}"
                        )

        # Build the final JSON output
        output_data = {
            "metadata": metadata,
            "benchmarks_summary": benchmarks_release_data,
            "aiperf_benchmarks": aiperf_release_data if aiperf_release_data else [],
            "evals": evals_release_data,
            "stress_tests": stress_tests_release_data,
            "benchmarks": benchmarks_detailed_data
            if benchmarks_detailed_data
            else [
                {
                    "model_id": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
            "aiperf_benchmarks_detailed": aiperf_detailed_data
            if aiperf_detailed_data
            else [],
        }

        # Add server_tests only if data exists
        if server_tests_data:
            output_data["server_tests"] = server_tests_data

        # Add parameter_support_tests only if data exists
        if parameter_support_tests_data:
            output_data["parameter_support_tests"] = parameter_support_tests_data

        json.dump(output_data, f, indent=4)

    main_return_code = 0
    return main_return_code


def server_tests_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    """Generate server tests report by reading all markdown files from test_reports directory.

    Args:
        args: Command line arguments
        server_mode: Server mode (API/docker)
        model_spec: Model specification
        report_id: Report identifier
        metadata: Additional metadata

    Returns:
        Tuple of (release_str, release_data) where:
            release_str: Markdown formatted string of all test reports
            release_data: List of test report data
    """
    output_dir = Path(args.output_path) / "server_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for markdown files in project_root/test_reports
    test_reports_path = Path(project_root) / "test_reports"

    logger.info("Server Tests Summary")

    if not test_reports_path.exists():
        logger.info(f"Test reports directory not found: {test_reports_path}")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
        )

    # Find all markdown files
    md_files = list(test_reports_path.glob("*.md"))

    logger.info(f"Processing: {len(md_files)} markdown file(s)")

    if not md_files:
        logger.info("No server test report markdown files found. Skipping.")
        return (
            "",
            [
                {
                    "model": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
        )

    # Read and combine all markdown files
    combined_markdown = []
    release_data = []

    for md_file in sorted(md_files):
        try:
            logger.info(f"Reading: {md_file.name}")
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
                combined_markdown.append(f"#### {md_file.stem}\n\n{content}")

                # Try to extract JSON data if corresponding JSON file exists
                json_file = md_file.with_suffix(".json")
                if json_file.exists():
                    with open(json_file, "r", encoding="utf-8") as jf:
                        json_data = json.load(jf)
                        release_data.append(json_data)
        except Exception as e:
            logger.warning(f"Could not read file {md_file}: {e}")

    # Join all markdown content
    markdown_str = "\n\n---\n\n".join(combined_markdown)

    release_str = f"### Server Test Results for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    # Save combined report
    summary_fpath = output_dir / f"summary_{report_id}.md"
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(markdown_str)

    logger.info(f"Server tests summary saved to: {summary_fpath}")

    return release_str, release_data


if __name__ == "__main__":
    sys.exit(main())
