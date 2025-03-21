# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import logging
import json
from glob import glob
from pathlib import Path
from typing import List, Dict

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import (
    WORKFLOW_REPORT_CONFIG,
)
from workflows.utils import get_default_workflow_root_log_dir

# from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.log_setup import setup_workflow_script_logger

from benchmarking.summary_report import generate_report


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


def benchmark_generate_report(args, server_mode, model_config, metadata={}):
    file_name_pattern = f"benchmark_{model_config.model_name}_{args.device}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/benchmarks_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "benchmarks"

    logger.info("Benchmark Summary")
    logger.info(f"Processing: {len(files)} files")
    disp_md_path, stats_file_path = generate_report(files, output_dir, metadata)
    return disp_md_path, stats_file_path


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
    dataset_path = config.get("dataset_path", "N/A")
    assert task_name == first_key

    meta_data = {"task_name": task_name, "dataset_path": dataset_path}

    return extracted, meta_data


def extract_eval_results(files):
    results = []
    meta_data = []
    for json_file in files:
        logger.info(f"Processing: {json_file}")
        res, meta = extract_eval_json_data(Path(json_file))
        results.append(res)
        meta_data.append(meta)

    return results, meta_data


def evals_generate_report(args, server_mode, model_config, metadata={}):
    file_name_pattern = f"eval_{model_config.model_name}_{args.device}/{model_config.hf_model_repo.replace('/', '__')}/results_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/evals_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    results, meta_data = extract_eval_results(files)
    # for res, meta in zip(results, meta_data):

    generate_evals_markdown_table(results, meta_data)


def generate_evals_markdown_table(
    results: List[List[Dict]], meta_data: List[Dict]
) -> str:
    rows = []
    for task_group in results:
        for task in task_group:
            for task_name, metrics in task.items():
                for metric_name, metric_value in metrics.items():
                    rows.append((task_name, metric_name, f"{metric_value:.4f}"))

    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = (
        f"|{'-'*(col_widths[0]+2)}|{'-'*(col_widths[1]+2)}|{'-'*(col_widths[2]+2)}|"
    )
    markdown = header + "\n" + separator + "\n"

    for task_name, metric_name, metric_value in rows:
        markdown += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"

    print(markdown)

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
    if args.local_server:
        server_mode = "local"
    elif args.docker_server:
        server_mode = "docker"

    metadata = {
        "device": args.device,
        "server_mode": server_mode,
        "tt_metal_commit": model_config.tt_metal_commit,
        "vllm_commit": model_config.vllm_commit,
    }

    disp_md_path, stats_file_path = benchmark_generate_report(
        args, server_mode, model_config, metadata=metadata
    )

    evals_generate_report(args, server_mode, model_config, metadata=metadata)


if __name__ == "__main__":
    main()
