#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import os
import json
import jwt
import sys

# Add the script's directory to the Python path
# this for 0 setup python setup script
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Add the project root to the Python path to ensure imports work
sys.path.insert(0, str(project_root))

from stress_tests import StressTests
from stress_tests.stress_tests_args import StressTestsArgs
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.workflow_types import DeviceTypes
from workflows.workflow_config import (
    WORKFLOW_STRESS_TESTS_CONFIG,
)
from workflows.log_setup import setup_workflow_script_logger
import logging

# Removed get_model_id - now using ModelSpec.from_json
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Stress Tests.")
    parser.add_argument(
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for stress test output",
        required=True,
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (kept for compatibility)",
        required=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device name (kept for compatibility)",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")
    args = parse_arguments()

    if args.jwt_secret:
        # If jwt-secret is provided, generate the JWT and set OPENAI_API_KEY.
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, args.jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )

    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)

    # runtime config loaded from JSON
    device_str = runtime_config.device
    disable_trace_capture = runtime_config.disable_trace_capture
    workflow_args = runtime_config.workflow_args

    # Parse workflow_args if provided (same logic as was in run_workflows.py)
    parsed_workflow_args = {}
    if workflow_args:
        workflow_args_pairs = workflow_args.split()
        for pair in workflow_args_pairs:
            if "=" in pair:
                key, value = pair.split("=", 1)
                # Convert key from kebab-case to snake_case for internal use
                key = key.replace("-", "_")
                # Try to convert numeric values to int
                try:
                    parsed_workflow_args[key] = int(value)
                except ValueError:
                    # Try to convert boolean strings to actual booleans
                    if value.lower() in ("true", "false"):
                        parsed_workflow_args[key] = value.lower() == "true"
                    else:
                        parsed_workflow_args[key] = value

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_STRESS_TESTS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    assert device == model_spec.device_type

    service_port = runtime_config.service_port
    logger.info(f"service_port=: {service_port}")
    logger.info(f"run_mode=: {parsed_workflow_args.get('run_mode', 'multiple')}")
    logger.info(
        f"max_context_length=: {parsed_workflow_args.get('max_context_length')}"
    )
    logger.info(f"endurance_mode=: {parsed_workflow_args.get('endurance_mode', False)}")
    logger.info(f"workflow_args=: {workflow_args}")
    logger.info(f"output_path=: {args.output_path}")
    logger.info("Wait for the vLLM server to be ready ...")

    # Create consolidated stress tests arguments from multiple sources
    stress_args = StressTestsArgs.from_sources(
        args, runtime_config, model_spec, parsed_workflow_args
    )
    run_stress_test = StressTests(stress_args, model_spec)

    run_stress_test.run()
    logger.info("✅ Completed stress tests")
