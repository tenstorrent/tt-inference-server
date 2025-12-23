# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import csv
import json
import logging
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
from benchmarking.summary_report import generate_report, get_markdown_table
from evals.eval_config import EVAL_CONFIGS
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


def benchmark_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    # Look for both vLLM and genai-perf benchmark files
    vllm_pattern = f"benchmark_{model_spec.model_id}_*.json"
    genai_pattern = f"genai_benchmark_{model_spec.model_id}_*.json"

    benchmarks_output_dir = f"{get_default_workflow_root_log_dir()}/benchmarks_output"
    vllm_files = glob(f"{benchmarks_output_dir}/{vllm_pattern}")
    genai_files = glob(f"{benchmarks_output_dir}/{genai_pattern}")

    files = vllm_files + genai_files
    logger.info(
        f"Found {len(vllm_files)} vLLM benchmark files and {len(genai_files)} genai-perf benchmark files"
    )
    output_dir = Path(args.output_path) / "benchmarks"
    logger.info("Benchmark Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
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
    # extract summary data
    release_str, release_raw, disp_md_path, stats_file_path = generate_report(
        files, output_dir, report_id, metadata, model_spec=model_spec
    )
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

    # Separate text and image benchmarks from release_raw
    text_release_raw = [r for r in release_raw if r.get("task_type", "text") == "text"]
    image_release_raw = [
        r for r in release_raw if r.get("task_type", "text") == "image"
    ]

    # Separate text and image performance references
    text_perf_refs = [
        p_ref for p_ref in perf_refs if getattr(p_ref, "task_type", "text") == "text"
    ]
    image_perf_refs = [
        p_ref for p_ref in perf_refs if getattr(p_ref, "task_type", "text") == "image"
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

    # Process image benchmarks if they exist
    print(f"image_release_raw: {image_release_raw}")
    if image_perf_refs and image_release_raw and False:
        # hard coded for now - using fallback option for image benchmarks
        # TODO: implement proper image benchmark targets retrieval
        # make lookup dict so references can find the correct result row
        # key: (isl, osl, image_height, image_width, images_per_prompt, max_concurrency)
        image_res_dict = {
            (
                r["input_sequence_length"],
                r["output_sequence_length"],
                r["image_height"],
                r["image_width"],
                r["images_per_prompt"],
                r["max_con"],
            ): r
            for r in image_release_raw
        }
        image_perf_results = {}
        for p_ref in image_perf_refs:
            p_ref_key = (
                p_ref.isl,
                p_ref.osl,
                p_ref.image_height,
                p_ref.image_width,
                p_ref.images_per_prompt,
                p_ref.max_concurrency,
            )
            res = image_res_dict.get(p_ref_key)
            # add reference values to the result
            image_perf_results[p_ref_key] = {
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
                image_perf_results[p_ref_key].update(
                    {
                        "ttft": res["mean_ttft_ms"],
                        "tput_user": res["mean_tps"],
                        "tput": res["tps_decode_throughput"],
                    }
                )

                # Prepare a dictionary to hold checks for all targets.
                image_perf_results[p_ref_key]["target_checks"] = {}
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
                    image_perf_results[p_ref_key]["target_checks"][target_name] = (
                        target_check
                    )

            else:
                # No result available from benchmark measurements.
                NA_STRING = "N/A"
                # In this case, add N/A for performance measures and an empty check dict per target.
                image_perf_results[p_ref_key].update(
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

        # build release performance benchmarking report for images
        sorted_image_perf_results = {
            k: image_perf_results[k] for k in sorted(image_perf_results)
        }
        image_release_raw_targets = [v for k, v in sorted_image_perf_results.items()]

        flat_image_release_raw = flatten_target_checks(image_release_raw_targets)
        image_section = (
            f"#### Image Benchmark Targets {model_spec.model_name} on {args.device}\n\n"
        )
        if image_release_raw_targets and image_release_raw_targets[0].get(
            "target_checks"
        ):
            image_section += benchmark_image_release_markdown(
                flat_image_release_raw,
                target_checks=image_release_raw_targets[0]["target_checks"],
            )
        else:
            image_section += benchmark_image_release_markdown(
                flat_image_release_raw, target_checks=None
            )
        release_sections.append(image_section)
    elif image_release_raw:
        # Show image benchmarks even without performance targets
        image_section = (
            f"#### Image Benchmark Results {model_spec.model_name} on {args.device}\n\n"
        )
        image_section += "No performance targets defined for image benchmarks.\n\n"
        release_sections.append(image_section)

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
        elif image_perf_refs:
            release_raw = (
                image_release_raw_targets
                if "image_release_raw_targets" in locals()
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
        task_name = meta.pop("task_name")
        check_task_name = list(res[0].keys())[0]
        assert task_name == check_task_name, (
            f"Task name mismatch: {task_name} != {check_task_name}"
        )
        results[task_name] = {k: v for d in res for k, v in d.items()}
        meta_data[task_name] = meta

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
        if task.task_name in results:
            logger.info(f"eval processing task_name: {task.task_name}")
            res = results[task.task_name]
            kwargs = task.score.score_func_kwargs
            kwargs["task_name"] = task.task_name
            score = task.score.score_func(res, task_name=task.task_name, kwargs=kwargs)

            # For WER (Word Error Rate), convert to accuracy once before all calculations
            # WER is an error rate (lower is better), but published/reference scores are accuracy (higher is better)
            if kwargs.get("unit") == "WER":
                score = 100 - score

            if task.score.published_score:
                assert task.score.published_score > 0, "Published score is not > 0"
                ratio_to_published = score / task.score.published_score
            else:
                ratio_to_published = "N/A"
            if task.score.gpu_reference_score:
                assert task.score.gpu_reference_score > 0, "Reference score is not > 0"
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
        )
    # TODO: Support handling of multiple test reports
    assert len(files) == 1, "Handling of multiple tests reports is unimplemented."
    files = files[0]

    # generate vLLM parameter coverage report
    # TODO: Implement returning raw report, defaulting to None for now
    markdown_str, release_raw = (
        generate_vllm_parameter_report(
            files, output_path, report_id, metadata, model_spec=model_spec
        ),
        None,
    )

    release_str = f"### Test Results for {model_spec.model_name} on {args.device}\n\n{markdown_str}"

    return release_str, release_raw


def generate_evals_markdown_table(results, meta_data) -> str:
    rows = []
    for task_group, tasks in results.items():
        for task_name, metrics in tasks.items():
            for metric_name, metric_value in metrics.items():
                if metric_name and metric_name != " ":
                    if not isinstance(
                        metric_value, float
                    ):  # some metrics in image evals are not floats
                        continue
                    rows.append((task_name, metric_name, f"{metric_value:.4f}"))
    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|"
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    return markdown


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
    ):
        benchmark_summary["tput_user"] = benchmark_summary_data.get("tput_user", 0)

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

def benchmarks_release_data_format_embedding(model_spec, device_str, benchmark_summary_data):
    """Convert the benchmark release data to the desired format for EMBEDDING models"""

    return [{
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
    }]


def add_target_checks_cnn_and_image(
    targets, evals_release_data, benchmark_summary_data, metrics
):
    """Add target checks for CNN and IMAGE models based on evals and benchmark data."""
    logger.info("Adding target_checks to CNN and IMAGE benchmark release data")
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

def add_target_checks_embedding(
    targets, benchmark_summary_data
):
    """Add target checks for EMBEDDING models based on evals and benchmark data."""
    logger.info("Adding target_checks to EMBEDDING benchmark release data")

    # extract targets for functional, complete, target and calculate them
    tput_user = benchmark_summary_data.get("mean_tps", 0)
    functional_tput_user = targets.tput_user / 100
    complete_tput_user = targets.tput_user / 10
    target_tput_user = targets.tput_user

    tput_prefill = benchmark_summary_data.get("tps_prefill_throughput", 0)
    functional_tput_prefill = targets.tput_prefill / 100
    complete_tput_prefill = targets.tput_prefill / 10
    target_tput_prefill = targets.tput_prefill

    e2el_ms = benchmark_summary_data.get("mean_e2el_ms", 0)
    functional_e2el_ms = targets.e2el_ms * 100
    complete_e2el_ms = targets.e2el_ms * 10
    target_e2el_ms = targets.e2el_ms

    logger.info("Calculating target checks")
    target_checks = {
        "functional": {
            "tput_user": functional_tput_user,
            "tput_user_ratio": tput_user / functional_tput_user,
            "tput_user_check": 2 if tput_user > functional_tput_user else 3,
            "tput_prefill": functional_tput_prefill,
            "tput_prefill_ratio": tput_prefill / functional_tput_prefill,
            "tput_prefill_check": 2 if tput_prefill > functional_tput_prefill else 3,
            "e2el_ms": functional_e2el_ms,
            "e2el_ms_ratio": e2el_ms / functional_e2el_ms,
            "e2el_ms_check": 2 if e2el_ms < functional_e2el_ms else 3,
        },
        "complete": {
            "tput_user": complete_tput_user,
            "tput_user_ratio": tput_user / complete_tput_user,
            "tput_user_check": 2 if tput_user > complete_tput_user else 3,
            "tput_prefill": complete_tput_prefill,
            "tput_prefill_ratio": tput_prefill / complete_tput_prefill,
            "tput_prefill_check": 2 if tput_prefill > complete_tput_prefill else 3,
            "e2el_ms": complete_e2el_ms,
            "e2el_ms_ratio": e2el_ms / complete_e2el_ms,
            "e2el_ms_check": 2 if e2el_ms < complete_e2el_ms else 3,
        },
        "target": {
            "tput_user": target_tput_user,
            "tput_user_ratio": tput_user / target_tput_user,
            "tput_user_check": 2 if tput_user > target_tput_user else 3,
            "tput_prefill": target_tput_prefill,
            "tput_prefill_ratio": tput_prefill / target_tput_prefill,
            "tput_prefill_check": 2 if tput_prefill > target_tput_prefill else 3,
            "e2el_ms": target_e2el_ms,
            "e2el_ms_ratio": e2el_ms / target_e2el_ms,
            "e2el_ms_check": 2 if e2el_ms < target_e2el_ms else 3,
        },
    }

    return target_checks


def calculate_target_metrics(avg_ttft, target_ttft):
    """Calculate TTFT metrics for functional, complete, and target thresholds.

    Args:
        avg_ttft: Average TTFT from benchmark results
        target_ttft: Target TTFT from performance reference

    Returns:
        Dict containing metrics for all target levels (functional, complete, target)
    """

    def get_ttft_ratio_and_check(avg_ttft, ref_ttft):
        if not ref_ttft:
            return "Undefined", "Undefined"
        ratio = avg_ttft / ref_ttft
        if ratio < 1.0:
            check = 2
        elif ratio > 1.0:
            check = 3
        else:
            check = "Undefined"
        return ratio, check

    # Define target level multipliers
    target_multipliers = {
        "functional": FUNCTIONAL_TARGET,  # 10x slower than target
        "complete": COMPLETE_TARGET,  # 2x slower than target
        "target": 1,  # actual target
    }

    metrics = {}
    for level, multiplier in target_multipliers.items():
        level_ttft = target_ttft * multiplier
        ratio, check = get_ttft_ratio_and_check(avg_ttft, level_ttft)

        metrics[f"{level}_ttft"] = level_ttft
        metrics[f"{level}_ttft_ratio"] = ratio
        metrics[f"{level}_ttft_check"] = check

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
        def __init__(self, output_path, model, device, model_spec_json):
            self.output_path = output_path
            self.model = model
            self.device = device
            self.model_spec_json = model_spec_json

    simple_args = SimpleArgs(args.output_path, model, device_str, args.model_spec_json)

    # generate benchmarks report
    (
        benchmarks_release_str,
        benchmarks_release_data,
        benchmarks_disp_md_path,
        benchmarks_data_file_path,
    ) = benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate evals report
    evals_release_str, evals_release_data, evals_disp_md_path, evals_data_file_path = (
        evals_generate_report(
            simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
        )
    )

    # generate tests report
    tests_release_str, tests_release_data = generate_tests_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # generate server tests report
    server_tests_release_str, server_tests_release_data = server_tests_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )

    # if no benchmark data exists, do not
    try:
        with open(benchmarks_disp_md_path, "r", encoding="utf-8") as f:
            benchmarks_disp_md_str = f.read()
    except TypeError:
        benchmarks_disp_md_str = ""

    logging.info("Release Summary\n\n")

    release_header = (
        f"## Tenstorrent Model Release Summary: {model_spec.model_name} on {device_str}"
    )
    release_str = f"{release_header}\n\n{metadata_str}\n\n{benchmarks_disp_md_str}\n\n{benchmarks_release_str}\n\n{evals_release_str}\n\n{tests_release_str}\n\n{server_tests_release_str}"
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

        # Add target_checks for specific model if applicable
        if (
            model_spec.model_type.name == ModelType.CNN.name
            or model_spec.model_type.name == ModelType.IMAGE.name
            or model_spec.model_type.name == ModelType.AUDIO.name
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

            # Initialize the benchmark summary data
            benchmark_summary_data = {}

            # Aggregate mean_ttft_ms and inference_steps_per_second across all benchmarks
            total_ttft = 0.0
            total_tput = 0.0
            for benchmark in benchmarks_release_data:
                total_ttft += benchmark.get("mean_ttft_ms", 0)
                total_tput += benchmark.get("inference_steps_per_second", 0)
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

            # Calculate all target metrics using centralized function
            metrics = calculate_target_metrics(avg_ttft, target_ttft)

            target_checks = {}
            if (
                model_spec.model_type.name == ModelType.CNN.name
                or model_spec.model_type.name == ModelType.IMAGE.name
            ):
                logger.info(
                    "Adding target_checks for tput_user to CNN and IMAGE benchmark release data"
                )
                target_checks = add_target_checks_cnn_and_image(
                    targets,
                    evals_release_data,
                    benchmark_summary_data,
                    metrics,
                )
            else:
                logger.info("Adding target_checks for Audio benchmark release data")
                target_checks = add_target_checks_audio(metrics)

            # Make sure benchmarks_release_data is of proper format for CNN and IMAGE
            benchmarks_release_data = benchmarks_release_data_format(
                model_spec, device_str, benchmark_summary_data
            )

            # Add target_checks to the existing benchmark object
            if benchmarks_release_data:
                benchmarks_release_data[0]["target_checks"] = target_checks

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

            logger.info("Adding target_checks for Embedding benchmark release data")
            target_checks = add_target_checks_embedding(
                targets,
                benchmark_summary_data,
            )

            benchmarks_release_data = benchmarks_release_data_format_embedding(
                model_spec, device_str, benchmark_summary_data
            )

            if benchmarks_release_data:
                benchmarks_release_data[0]["target_checks"] = target_checks


        # Build the final JSON output
        output_data = {
            "metadata": metadata,
            "benchmarks_summary": benchmarks_release_data,
            "evals": evals_release_data,
            "benchmarks": benchmarks_detailed_data
            if benchmarks_detailed_data
            else [
                {
                    "model_id": getattr(args, "model", "unknown_model"),
                    "device": getattr(args, "device", "unknown_device"),
                }
            ],
        }

        # Add server_tests only if data exists
        if server_tests_data:
            output_data["server_tests"] = server_tests_data

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
