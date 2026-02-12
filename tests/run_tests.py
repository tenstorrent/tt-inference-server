# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from datetime import datetime
import os
import argparse
import json
import jwt
import logging
import sys
from pathlib import Path
from typing import List

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from tests.test_config import TEST_CONFIGS, TestTask
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.utils import run_command
from workflows.workflow_config import (
    WORKFLOW_TESTS_CONFIG,
)
from workflows.workflow_venvs import VENV_CONFIGS


logger = logging.getLogger(__name__)


def build_test_command(
    task: TestTask,
    model_spec,
    device,
    output_path,
    service_port,
) -> List[str]:
    """
    Build the command for tests by templating command-line arguments using properties
    from the given task and model configuration.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]

    test_exec = task_venv_config.venv_path / "bin" / "pytest"

    test_kwargs_list = [f"-{arg}" for arg in task.test_args]

    # set output_dir
    # results go to {output_dir_path}/{hf_repo}/results_{timestamp}
    output_dir_path = (
        Path(output_path)
        / f"test_{model_spec.model_id}__{run_timestamp}_{task.task_name}"
    )

    cmd = [
        str(test_exec),
        task.test_path,
        "--model-name",
        model_spec.model_name,
        "--model-impl",
        model_spec.impl.impl_name,
        "--output-path",
        output_dir_path,
    ]
    cmd.extend(test_kwargs_list)
    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run tests")
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


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    jwt_secret = args.jwt_secret
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    device_str = cli_args.get("device")
    service_port = cli_args.get("service_port", os.getenv("SERVICE_PORT", "8000"))

    workflow_config = WORKFLOW_TESTS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    logger.info(f"service_port=: {service_port}")
    logger.info(f"output_path=: {args.output_path}")

    # set environment vars
    if jwt_secret:
        # If jwt-secret is provided, generate the JWT and set OPENAI_API_KEY.
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using BENCHMARK_CONFIGS.
    if model_spec.model_name not in TEST_CONFIGS:
        raise ValueError(
            f"No benchmark tasks defined for model: {model_spec.model_name}"
        )
    test_config = TEST_CONFIGS[model_spec.model_name]

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = args.jwt_secret
    env_config.service_port = cli_args.get("service_port")
    env_config.vllm_model = model_spec.hf_model_repo

    # Execute pytest for each task.
    logger.info("Running test client ...")
    return_codes = []
    for task in test_config.tasks:
        logger.info(
            f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
        )

        logger.info(f"Running tests for:\n {task}")
        cmd = build_test_command(
            task,
            model_spec,
            device_str,
            args.output_path,
            cli_args.get("service_port"),
        )
        return_code = run_command(command=cmd, logger=logger, env=env_vars)
        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed tests")
        return 0
    else:
        logger.error(
            f"⛔ tests failed with return codes: {return_codes}. See logs above for details."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
