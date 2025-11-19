# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import os
import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.log_setup import setup_workflow_script_logger


logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for benchmark output",
        required=True,
    )

    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    ret_args = parser.parse_args()
    return ret_args


if __name__ == "__main__":
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    breakpoint()
