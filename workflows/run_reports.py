# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import argparse
import logging
from glob import glob
from pathlib import Path

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

from benchmarking.summary_report import (
    process_benchmark_files,
    get_markdown_table,
    save_markdown_table,
    save_to_csv,
    create_display_dict,
)

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


def extract_benchmark_results(args, server_mode, model_config):
    file_name_pattern = f"benchmark_{model_config.model_name}_{args.device}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/benchmarks_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    assert len(files) > 0, f"No benchmark files found for pattern: {file_path_pattern}"

    results = process_benchmark_files(files, pattern="benchmark_*.json")
    timestamp_str = results[0]["timestamp"]

    # Display basic statistics
    logger.info("Benchmark Summary:")
    logger.info(f"Total files processed: {len(results)}")

    # Save to CSV
    output_dir = Path(args.output_path) / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # save stats
    stats_file_path = output_dir / f"benchmark_stats_{timestamp_str}.csv"
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


def extract_eval_results():
    pass


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

    extract_benchmark_results(args, server_mode, model_config)


if __name__ == "__main__":
    main()
