#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import argparse
import sys
import logging

# Add the script's directory to the Python path
# this for 0 setup python setup script
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from setup.configs import model_config
from setup.setup_host import setup_host

# Configure logging to include the time, log level, and message.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    valid_workflows = {"benchmarks", "evals", "server", "release", "report"}
    valid_devices = {"N150", "N300", "T3K"}
    valid_models = model_config.keys()
    # required
    parser = argparse.ArgumentParser(
        description="A CLI for running workflows with optional docker, device, and workflow-args.",
        epilog="\nAvailable models:\n  " + "\n  ".join(model_config.keys()),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=valid_models, help="Model to run"
    )
    parser.add_argument(
        "--workflow",
        required=True,
        choices=valid_workflows,
        help=f"Workflow to run (choices: {', '.join(valid_workflows)})",
    )
    # optional
    parser.add_argument("--docker", action="store_true", help="Enable docker mode")
    parser.add_argument(
        "--device",
        choices=valid_devices,
        help=f"Device option (choices: {', '.join(valid_devices)})",
    )
    parser.add_argument(
        "--workflow-args",
        help="Additional workflow arguments (e.g., 'param1=value1 param2=value2')",
    )

    return parser.parse_args()


def run_benchmarks(args):
    logging.info("Running benchmarks...")
    # Insert your benchmark commands here.


def run_evals(args):
    logging.info("Running evaluations...")
    # Insert your evaluation commands here.


def run_server(args):
    logging.info("Starting server...")
    # Insert your server commands here.


def run_workflow(args):
    # Mapping workflow names to functions.
    workflow_map = {
        "benchmarks": run_benchmarks,
        "evals": run_evals,
        "server": run_server,
    }

    # Execute the workflow function using dictionary lookup.
    workflow_func = workflow_map.get(args.workflow)
    if not workflow_func:
        logging.error(f"Error: Unknown workflow '{args.workflow}'")
        sys.exit(1)
    workflow_func(args)

    # Process additional workflow arguments if provided.
    if args.workflow_args:
        logging.info(f"Additional workflow arguments: {args.workflow_args}")
        # Process additional workflow arguments as needed.


def main():
    if __name__ != "__main__":
        logging.error(
            "⛔ Error: This script is being imported. Please execute it directly."
        )
        sys.exit(1)
    args = parse_arguments()
    logging.info(f"model:          {args.model}")
    logging.info(f"workflow:       {args.workflow}")
    logging.info(f"docker:         {args.docker}")
    logging.info(f"device:         {args.device}")
    logging.info(f"workflow_args:  {args.workflow_args}")
    logging.info("-------------------------------")

    if args.docker:
        logging.info("Docker mode enabled")
        setup_host(args.model)
    else:
        # run outside docks user existing dev env
        logging.info("Running on host without Docker ...")
        # validate_environment()

        run_workflow(args)


if __name__ == "__main__":
    main()
