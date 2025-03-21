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


def extract_benchmark_results(args, server_mode, model_config):
    file_name_pattern = f"benchmark_{model_config.model_name}_{args.device}_*.json"
    file_path_pattern = (
        f"{get_default_workflow_root_log_dir()}/benchmarks_output/{file_name_pattern}"
    )
    files = glob(file_path_pattern)
    output_dir = Path(args.output_path) / "benchmarks"
    metadata = {
        "device": args.device,
        "server_mode": server_mode,
        "tt_metal_commit": model_config.tt_metal_commit,
        "vllm_commit": model_config.vllm_commit,
    }

    logger.info("Benchmark Summary")
    logger.info(f"Processing: {len(files)} files")
    disp_md_path, stats_file_path = generate_report(files, output_dir, metadata)


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
