#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Standalone CLI tool for parsing a single workflow_logs_* directory.

This script parses a workflow_logs_* directory produced by CI runs and outputs
the extracted model specifications, performance reports, and status information
to a JSON file.

Usage:
    python3 scripts/release/parse_workflow_logs.py \\
        release_logs/On_nightly_281/workflow_logs_release_Model_config \\
        --output results.json
"""

import argparse
import logging
import sys
from pathlib import Path

from workflow_logs_parser import parse_workflow_logs_dir, write_workflow_logs_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse a single workflow_logs_* directory and output results to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a workflow logs directory and save results
  python3 scripts/release/parse_workflow_logs.py \\
      release_logs/On_nightly_281/workflow_logs_release_Llama-3.3-70B-Instruct_llmbox \\
      --output llama_results.json
  
  # Parse with debug logging
  python3 scripts/release/parse_workflow_logs.py \\
      release_logs/On_nightly_281/workflow_logs_release_Qwen3-8B_n300 \\
      --output qwen_results.json \\
      --verbose
        """,
    )
    parser.add_argument(
        "workflow_logs_dir",
        type=str,
        help="Path to workflow_logs_* directory (e.g., release_logs/On_nightly_281/workflow_logs_release_Model_config)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSON file path"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Update logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Parse the directory
    workflow_logs_path = Path(args.workflow_logs_dir).resolve()
    logger.info(f"Parsing workflow logs from: {workflow_logs_path}")

    parsed_data = parse_workflow_logs_dir(workflow_logs_path)

    if not parsed_data:
        logger.error(f"Failed to parse workflow logs from: {workflow_logs_path}")
        logger.error(
            "Please check that the directory exists and contains valid run_specs/ and reports_output/ subdirectories"
        )
        return 1

    # Write output
    output_path = Path(args.output).resolve()
    write_workflow_logs_output(parsed_data, output_path)

    logger.info("✅ Successfully parsed workflow logs")
    logger.info(f"   Model ID: {parsed_data.get('model_id')}")
    logger.info(f"   Perf Status: {parsed_data.get('perf_status')}")
    logger.info(f"   Accuracy Status: {parsed_data.get('accuracy_status')}")
    logger.info(f"   Passing: {parsed_data.get('is_passing')}")
    logger.info(f"   Output: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
