# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import logging
import json
from glob import glob
from pathlib import Path

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import MODEL_CONFIGS
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
    parser = argparse.ArgumentParser(description="Run vLLM evals")
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for evaluation output",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="DeviceTypes str used to simulate different hardware configurations",
    )
    # optional
    parser.add_argument(
        "--local-server", action="store_true", help="Run inference server on localhost"
    )
    parser.add_argument(
        "--docker-server",
        action="store_true",
        help="Run inference server in Docker container",
    )
    ret_args = parser.parse_args()
    return ret_args


def benchmark_release_markdown(release_raw):
    # Define display columns mapping
    display_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("ref_ttft", "Ref TTFT (ms)"),
        ("ttft_ratio", "TTFT Ratio"),
        ("ttft_check", "TTFT Check"),
        ("tput_user", "Tput User (TPS)"),
        ("ref_tput_user", "Ref Tput User (TPS)"),
        ("tput_user_ratio", "Tput User Ratio"),
        ("tput_user_check", "Tput User Check"),
        ("tput", "Tput Decode (TPS)"),
    ]
    NOT_MEASURED_STR = "N/A"
    cols_to_round = ["ttft_ratio", "tput_user_ratio", "tput_ratio"]
    display_dicts = []
    for row in release_raw:
        row_dict = {}
        for col_name, display_header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if isinstance(value, ReportCheckTypes):
                row_dict[display_header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round:
                row_dict[display_header] = f"{value:.2f}"
            else:
                row_dict[display_header] = str(value)
        display_dicts.append(row_dict)

    # Create the markdown table
    markdown_str = get_markdown_table(display_dicts)
    return markdown_str


def benchmark_generate_report(args, server_mode, model_config, metadata={}):
    file_name_pattern = f"benchmark_{model_config.model_name}_{args.device}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/benchmarks_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "benchmarks"

    logger.info("Benchmark Summary")
    logger.info(f"Processing: {len(files)} files")
    if not files:
        logger.info("No benchmark files found. Skipping.")
        return "", None, None, None
    # extract summary data
    release_str, release_raw, disp_md_path, stats_file_path = generate_report(
        files, output_dir, metadata
    )
    # release report for benchmarks
    device_type = DeviceTypes.from_string(args.device)
    perf_refs = model_config.perf_reference_map[device_type]
    # make lookup dict so references can find the correct result row
    # key: (isl, osl, mac_concurrency)
    res_dict = {
        (r["input_sequence_length"], r["output_sequence_length"], r["max_con"]): r
        for r in release_raw
    }
    perf_check_results = {}
    for p_ref in perf_refs:
        p_ref_key = (p_ref.isl, p_ref.osl, p_ref.max_concurrency)
        res = res_dict.get(p_ref_key)
        # add reference values to the result
        perf_check_results[p_ref_key] = {
            "isl": p_ref.isl,
            "osl": p_ref.osl,
            "max_concurrency": p_ref.max_concurrency,
            "ref_ttft": p_ref.ref_ttft_ms,
            "ref_tput_user": p_ref.ref_tput_user,
            "ref_tput": p_ref.ref_tput,
        }
        # add measurements to result and checks if defined
        if res:
            perf_check_results[p_ref_key].update(
                {
                    "ttft": res["mean_ttft_ms"],
                    "tput_user": res["mean_tps"],
                    "tput": res["tps_decode_throughput"],
                }
            )

            if p_ref.ref_ttft_ms:
                assert (
                    p_ref.ref_ttft_ms > 0
                ), f"ref_ttft_ms:={p_ref.ref_ttft_ms} is not > 0"
                ttft_ratio = res["mean_ttft_ms"] / p_ref.ref_ttft_ms
                check = ReportCheckTypes.from_result(ttft_ratio < (1 + p_ref.tolerance))
                perf_check_results[p_ref_key].update(
                    {"ttft_ratio": ttft_ratio, "ttft_check": check}
                )
            else:
                perf_check_results[p_ref_key].update(
                    {"ttft_check": ReportCheckTypes.NA}
                )

            if p_ref.ref_tput_user:
                assert (
                    p_ref.ref_tput_user > 0
                ), f"ref_tput_user:={p_ref.ref_tput_user} is not > 0"
                tput_user_ratio = res["mean_tps"] / p_ref.ref_tput_user
                check = ReportCheckTypes.from_result(
                    tput_user_ratio > (1 - p_ref.tolerance)
                )
                perf_check_results[p_ref_key].update(
                    {"tput_user_ratio": tput_user_ratio, "tput_user_check": check}
                )
            else:
                perf_check_results[p_ref_key].update(
                    {"tput_user_check": ReportCheckTypes.NA}
                )

            if p_ref.ref_tput:
                assert p_ref.ref_tput > 0, f"ref_tput:={p_ref.ref_tput} is not > 0"
                tput_ratio = res["tps_decode_throughput"] / p_ref.ref_tput
                check = ReportCheckTypes.from_result(
                    tput_user_ratio > (1 - p_ref.tolerance)
                )
                perf_check_results[p_ref_key].update(
                    {"tput_ratio": tput_ratio, "tput_check": check}
                )
            else:
                perf_check_results[p_ref_key].update(
                    {"tput_check": ReportCheckTypes.NA}
                )
        else:
            NA_STRING = "N/A"
            perf_check_results[p_ref_key] = {
                "ttft": NA_STRING,
                "tput_user": NA_STRING,
                "tput": NA_STRING,
                "ttft_check": ReportCheckTypes.NA,
                "tput_user_check": ReportCheckTypes.NA,
                "tput_check": ReportCheckTypes.NA,
            }

    # build release performance benchmarking report
    sorted_perf_results = {k: perf_check_results[k] for k in sorted(perf_check_results)}
    release_raw = [v for k, v in sorted_perf_results.items()]
    release_str = benchmark_release_markdown(release_raw)

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
        logger.info(f"Processing: {json_file}")
        res, meta = extract_eval_json_data(Path(json_file))
        task_name = meta.pop("task_name")
        check_task_name = list(res[0].keys())[0]
        assert (
            task_name == check_task_name
        ), f"Task name mismatch: {task_name} != {check_task_name}"
        results[task_name] = {k: v for d in res for k, v in d.items()}
        meta_data[task_name] = meta

    return results, meta_data


def evals_release_report_data(args, results, meta_data):
    eval_config = EVAL_CONFIGS[args.model]
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
                "model": args.model,
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


def evals_generate_report(args, server_mode, model_config, metadata={}):
    eval_run_id = f"{model_config.model_name}_{args.device}"
    output_dir = Path(args.output_path) / "evals"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_name_pattern = f"eval_{eval_run_id}/{model_config.hf_model_repo.replace('/', '__')}/results_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    logger.info("Evaluations Summary")
    logger.info(f"Processing: {len(files)} files")
    results, meta_data = extract_eval_results(files)
    if not results:
        logger.warning("No evaluation files found. Skipping.")
        return "", None, None, None
    # generate release report
    report_rows = evals_release_report_data(args, results, meta_data)

    # store results
    data_file_path = output_dir / f"report_{eval_run_id}.md"

    markdown_str = generate_evals_release_markdown(report_rows)

    release_str = f"### Accuracy evaluations for {model_config.model_name} on {args.device}\n\n{markdown_str}"

    # generate summary report
    summary_fpath = output_dir / f"summary_{eval_run_id}.md"
    summary_markdown_str = generate_evals_markdown_table(results, meta_data)
    with summary_fpath.open("w", encoding="utf-8") as f:
        f.write(summary_markdown_str)

    # store raw data
    release_raw = report_rows
    data_fpath = data_dir / f"eval_data_{eval_run_id}.json"

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
                    rows.append((task_name, metric_name, f"{metric_value:.4f}"))
    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = (
        f"|{'-'*(col_widths[0]+2)}|{'-'*(col_widths[1]+2)}|{'-'*(col_widths[2]+2)}|"
    )
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    return markdown


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    workflow_config = WORKFLOW_REPORT_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    assert DeviceTypes.from_string(args.device) in model_config.device_configurations

    assert not (
        args.local_server and args.docker_server
    ), "Cannot specify both --local-server and --docker-server"
    server_mode = "API"
    command_flag = ""
    if args.local_server:
        server_mode = "local"
        command_flag = "--local-server"
    elif args.docker_server:
        server_mode = "docker"
        command_flag = "--docker-server"

    release_run_id = f"{model_config.model_name}_{args.device}"

    metadata = {
        "model_name": model_config.model_name,
        "model_id": model_config.hf_model_repo,
        "device": args.device,
        "server_mode": server_mode,
        "tt_metal_commit": model_config.tt_metal_commit,
        "vllm_commit": model_config.vllm_commit,
        "run_command": f"python run.py --model {args.model} --device {args.device} --workflow release {command_flag}",
    }
    json_str = json.dumps(metadata, indent=4)
    metadata_str = f"### Metadata: {model_config.model_name} on {args.device}\n```json\n{json_str}\n```"

    (
        benchmarks_release_str,
        benchmarks_release_data,
        benchmarks_disp_md_path,
        benchmarks_data_file_path,
    ) = benchmark_generate_report(args, server_mode, model_config, metadata=metadata)
    evals_release_str, evals_release_data, evals_disp_md_path, evals_data_file_path = (
        evals_generate_report(args, server_mode, model_config, metadata=metadata)
    )

    logging.info("Release Summary\n\n")

    release_header = f"## Tenstorrent Model Release Summary: {model_config.model_name} on {args.device}"
    release_str = f"{release_header}\n\n{metadata_str}\n\n{benchmarks_release_str}\n\n{evals_release_str}"
    print(release_str)
    # save to file
    release_output_dir = Path(args.output_path) / "release"
    release_output_dir.mkdir(parents=True, exist_ok=True)
    release_data_dir = release_output_dir / "data"
    release_data_dir.mkdir(parents=True, exist_ok=True)
    release_file = release_output_dir / f"report_{release_run_id}.md"
    raw_file = release_data_dir / f"report_data_{release_run_id}.json"
    with release_file.open("w", encoding="utf-8") as f:
        f.write(release_str)

    with raw_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmarks": benchmarks_release_data,
                "evals": evals_release_data,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
