# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import logging
import json
import csv
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict
from dataclasses import field

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_spec import ModelSpec
from evals.eval_config import EVAL_CONFIGS
from workflows.workflow_config import (
    WORKFLOW_REPORT_CONFIG,
)
from workflows.utils import get_default_workflow_root_log_dir

# from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import DeviceTypes, ReportCheckTypes
from workflows.log_setup import setup_workflow_script_logger

from benchmarking.summary_report import generate_report, get_markdown_table


logger = logging.getLogger(__name__)


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
    file_name_pattern = f"benchmark_{model_spec.model_id}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/benchmarks_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
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
        files, output_dir, report_id, metadata
    )
    # release report for benchmarks
    device_type = DeviceTypes.from_string(args.device)

    perf_refs = (
        model_spec.device_model_spec.perf_reference
        if model_spec.device_model_spec.perf_reference
        else []
    )

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
                "max_concurrency": p_ref.max_concurrency,
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
                        assert (
                            perf_target.ttft_ms > 0
                        ), f"ttft_ms for target '{target_name}' is not > 0: {perf_target.ttft_ms}"
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
                        assert (
                            perf_target.tput_user > 0
                        ), f"tput_user for target '{target_name}' is not > 0: {perf_target.tput_user}"
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
                        assert (
                            perf_target.tput > 0
                        ), f"tput for target '{target_name}' is not > 0: {perf_target.tput}"
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
    if image_perf_refs and image_release_raw:
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
                "max_concurrency": p_ref.max_concurrency,
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
                        assert (
                            perf_target.ttft_ms > 0
                        ), f"ttft_ms for target '{target_name}' is not > 0: {perf_target.ttft_ms}"
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
                        assert (
                            perf_target.tput_user > 0
                        ), f"tput_user for target '{target_name}' is not > 0: {perf_target.tput_user}"
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
                        assert (
                            perf_target.tput > 0
                        ), f"tput for target '{target_name}' is not > 0: {perf_target.tput}"
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
        assert (
            task_name == check_task_name
        ), f"Task name mismatch: {task_name} != {check_task_name}"
        results[task_name] = {k: v for d in res for k, v in d.items()}
        meta_data[task_name] = meta

    return results, meta_data


def evals_release_report_data(args, results, meta_data, model_spec):
    eval_config = EVAL_CONFIGS[model_spec.model_name]
    
    # Apply audio dataset transformation if specified
    audio_eval_dataset = getattr(args, "audio_eval_dataset", None)
    if audio_eval_dataset and model_spec.model_type.name == "AUDIO":
        from evals.eval_config import apply_audio_dataset_transformation
        eval_config = apply_audio_dataset_transformation(eval_config, audio_eval_dataset)
        logger.info(f"Applied audio dataset transformation for report: {audio_eval_dataset}")
    
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


def evals_generate_report(args, server_mode, model_spec, report_id, metadata={}):
    eval_run_id = f"{model_spec.model_id}"
    output_dir = Path(args.output_path) / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
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
    logger.info("Evaluations Summary")
    logger.info(f"Processing: {len(files)} files")
    if (model_spec.model_type.name == "CNN") or (model_spec.model_type.name == "AUDIO"):
        # TODO rewrite this
        data_fpath = data_dir / f"eval_data_{report_id}.json"
        
        # Combine files into one JSON
        combined_data = {}
        for i, file_path in enumerate(files):
            with open(file_path, 'r') as f:
                file_data = json.load(f)
            combined_data = file_data
        
        # Write combined data to data_fpath
        with open(data_fpath, 'w') as f:
            json.dump(combined_data, f, indent=4)
        
        release_str = f"### Accuracy Evaluations for {model_spec.model_name} on {args.device}"
        summary_fpath = output_dir / f"summary_{report_id}.md"
        with summary_fpath.open("w", encoding="utf-8") as f:
            f.write("MD summary to do")
        
        return release_str, combined_data, summary_fpath, data_fpath

    results, meta_data = extract_eval_results(files)
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


def generate_evals_markdown_table(results, meta_data) -> str:
    rows = []
    for task_group, tasks in results.items():
        for task_name, metrics in tasks.items():
            for metric_name, metric_value in metrics.items():
                if metric_name and metric_name != " ":
                    if type(metric_value) != float: # some metrics in image evals are not floats
                        continue
                    rows.append((task_name, metric_name, f"{metric_value:.4f}"))
    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|"
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    return markdown

def benchmarks_release_data_cnn_format(model_spec, device_str, benchmark_summary_data):
    """ Convert the benchmark release data to the desired CNN format"""
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
        "mean_ttft_ms": benchmark_summary_data.get("mean_ttft_ms", 0),
        "inference_steps_per_second": benchmark_summary_data.get("inference_steps_per_second", 0),
        "filename": benchmark_summary_data.get("filename", ""),
        "task_type": model_spec.model_type.name.lower()
    }
    
    reformated_benchmarks_release_data.append(benchmark_summary)
    return reformated_benchmarks_release_data
    

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
    local_server = False  # Not passed via CLI args anymore
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

    (
        benchmarks_release_str,
        benchmarks_release_data,
        benchmarks_disp_md_path,
        benchmarks_data_file_path,
    ) = benchmark_generate_report(
        simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
    )
    evals_release_str, evals_release_data, evals_disp_md_path, evals_data_file_path = (
        evals_generate_report(
            simple_args, server_mode, model_spec, report_id=report_id, metadata=metadata
        )
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
    release_str = f"{release_header}\n\n{metadata_str}\n\n{benchmarks_disp_md_str}\n\n{benchmarks_release_str}\n\n{evals_release_str}"
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

        # Add target_checks for specific model if applicable
        if model_spec.model_type.name == "CNN" or model_spec.model_type.name == "AUDIO":
            # Import model_performance_reference from model_spec
            from workflows.model_spec import model_performance_reference

            # Extract the device we are running on
            device_str = cli_args.get("device").lower()

            # Get model performance targets from model_performance_reference.json and get data for the current model and device
            model_data = model_performance_reference.get(model_spec.model_name, {})
            if model_data == {} and "whisper" in model_spec.model_id.lower():
                # For whisper models, try looking up by model_name under whisper/ if lookup fails
                model_data = model_performance_reference.get("distil-whisper/" + model_spec.model_name, {})
            device_json_list = model_data.get(device_str, [])

            # extract targets for functional, complete, target and calculate them
            target_ttft = device_json_list[0]["targets"]["theoretical"]["ttft_ms"]
            functional_ttft = target_ttft * 10  # Functional target is 10x slower
            complete_ttft = target_ttft * 2     # Complete target is 2x slower

            # Initialize the benchmark summary data
            benchmark_summary_data = {}

            # Aggregate mean_ttft_ms and inference_steps_per_second across all benchmarks
            total_ttft = 0.0
            total_tput = 0.0
            for benchmark in benchmarks_release_data:
                total_ttft += benchmark.get("mean_ttft_ms", 0)
                total_tput += benchmark.get("inference_steps_per_second", 0)
                benchmark_summary_data["num_requests"] = benchmark.get("num_requests", 0)
                benchmark_summary_data["num_inference_steps"] = benchmark.get("num_inference_steps", 0)
                benchmark_summary_data["inference_steps_per_second"] = benchmark.get("inference_steps_per_second", 0)
                benchmark_summary_data["filename"] = benchmark.get("filename", "")
                benchmark_summary_data["mean_ttft_ms"] = benchmark.get("mean_ttft_ms", 0)

            avg_ttft = total_ttft / len(benchmarks_release_data) if len(benchmarks_release_data) > 0 else 0

            # Calculate ratios and checks for each target
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

            functional_ttft_ratio, functional_ttft_check = get_ttft_ratio_and_check(avg_ttft, functional_ttft)
            complete_ttft_ratio, complete_ttft_check = get_ttft_ratio_and_check(avg_ttft, complete_ttft)
            target_ttft_ratio, target_ttft_check = get_ttft_ratio_and_check(avg_ttft, target_ttft)

            # tput_check is always 1 for now (no tput target)
            tput_check = 1

            target_checks = {
                "functional": {
                    "ttft": functional_ttft,
                    "ttft_ratio": functional_ttft_ratio,
                    "ttft_check": functional_ttft_check,
                    "tput_check": tput_check
                },
                "complete": {
                    "ttft": complete_ttft,
                    "ttft_ratio": complete_ttft_ratio,
                    "ttft_check": complete_ttft_check,
                    "tput_check": tput_check
                },
                "target": {
                    "ttft": target_ttft,
                    "ttft_ratio": target_ttft_ratio,
                    "ttft_check": target_ttft_check,
                    "tput_check": tput_check
                }
            }

            # Make sure benchmarks_release_data is of proper format for CNN
            benchmarks_release_data = benchmarks_release_data_cnn_format(model_spec, device_str, benchmark_summary_data)
            
            # Add target_checks to the existing benchmark object
            if benchmarks_release_data:
                benchmarks_release_data[0]['target_checks'] = target_checks

        json.dump(
            {
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
            },
            f,
            indent=4,
        )

    main_return_code = 0
    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
