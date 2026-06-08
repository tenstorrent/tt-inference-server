# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
from datetime import datetime
import os
import argparse
import json
import jwt
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.remote_readiness import _wait_for_remote_openai_ready
from server_tests.test_config import TEST_CONFIGS, TestTask
from utils.prompt_client import PromptClient
from utils.prompt_configs import EnvironmentConfig
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import run_command
from workflows.workflow_config import (
    WORKFLOW_TESTS_CONFIG,
)
from workflows.workflow_venvs import VENV_CONFIGS


logger = logging.getLogger(__name__)


def _resolve_deploy_url(runtime_config: RuntimeConfig) -> str:
    """Resolve the inference server base URL for tests."""
    server_url = getattr(runtime_config, "server_url", None)
    if server_url:
        return server_url
    return os.environ.get("DEPLOY_URL", "http://127.0.0.1")


def _is_remote_server(runtime_config: RuntimeConfig) -> bool:
    return bool(getattr(runtime_config, "server_url", None))


def _resolve_api_base_url(deploy_url: str, service_port: int) -> str:
    """Build the OpenAI API base URL (without /v1 suffix)."""
    parsed = urlparse(deploy_url.rstrip("/"))
    if parsed.port is not None:
        return deploy_url.rstrip("/")
    return f"{deploy_url.rstrip('/')}:{service_port}"


def _setup_tests_auth(jwt_secret: str, remote_server: bool, logger) -> None:
    """Configure OPENAI_API_KEY for pytest subprocesses.

    Remote (--server-url): literal API_KEY / OPENAI_API_KEY only.
    Local: JWT_SECRET (standard workflow auth) or literal key fallback.
    """
    if remote_server:
        literal_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not literal_key:
            logger.warning(
                "No API_KEY or OPENAI_API_KEY set; remote endpoint requests "
                "will likely fail with 401."
            )
            return
        os.environ["OPENAI_API_KEY"] = literal_key
        logger.info(
            "OPENAI_API_KEY set from API_KEY / OPENAI_API_KEY for remote tests."
        )
        return

    if jwt_secret:
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    elif os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"):
        literal_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = literal_key
        logger.info(
            "OPENAI_API_KEY environment variable set from literal "
            "API_KEY / OPENAI_API_KEY."
        )


def build_test_command(
    task: TestTask,
    model_spec,
    device,
    output_dir_path,
    service_port,
    deploy_url: str = "http://127.0.0.1",
) -> list[str]:
    """
    Build the command for tests by templating command-line arguments using properties
    from the given task and model configuration.

    Returns cmd list.
    """
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]

    test_exec = task_venv_config.venv_path / "bin" / "pytest"

    test_kwargs_list = [f"-{arg}" for arg in task.test_args]

    base = _resolve_api_base_url(deploy_url, service_port)
    if task.task_name == "vllm_responses":
        test_kwargs_list.extend(["--endpoint-url", f"{base}/v1/responses"])
    elif task.task_name == "vllm_chat_completions":
        test_kwargs_list.extend(["--endpoint-url", f"{base}/v1/chat/completions"])
    cmd = [
        str(test_exec),
        task.test_path,
        "--model-name",
        model_spec.hf_model_repo,
        "--model-impl",
        model_spec.impl.impl_name,
        "--output-path",
        output_dir_path,
        "--task-name",
        task.task_name,
        "--max-context",
        str(model_spec.device_model_spec.max_context),
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
        "--runtime-model-spec-json",
        type=str,
        help="Use runtime model specification from JSON file",
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
    model_spec = ModelSpec.from_json(args.runtime_model_spec_json)
    runtime_config = RuntimeConfig.from_json(args.runtime_model_spec_json)
    remote_server = _is_remote_server(runtime_config)

    # runtime config loaded from JSON
    device_str = runtime_config.device
    service_port = runtime_config.service_port
    deploy_url = _resolve_deploy_url(runtime_config)
    # Propagate to subprocesses (pytest, etc.) that read DEPLOY_URL /
    # SERVICE_PORT (conftest --endpoint-url default, BaseTest). Without
    # exporting SERVICE_PORT, a non-default --service-port wouldn't reach
    # pytest and tests would target :8000. setdefault so an explicit
    # SERVICE_PORT already in the env still wins.
    os.environ["DEPLOY_URL"] = deploy_url
    os.environ.setdefault("SERVICE_PORT", str(service_port))

    workflow_config = WORKFLOW_TESTS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    logger.info(f"service_port=: {service_port}")
    logger.info(f"output_path=: {args.output_path}")

    _setup_tests_auth(jwt_secret, remote_server, logger)
    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using BENCHMARK_CONFIGS.
    if model_spec.model_name not in TEST_CONFIGS:
        message = f"No tests defined for model: {model_spec.model_name}"
        raise ValueError(message)
    test_config = TEST_CONFIGS[model_spec.model_name]

    if remote_server:
        logger.info("Wait for remote OpenAI-compatible endpoint to be ready ...")
    else:
        logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    if remote_server:
        env_config.vllm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    else:
        env_config.jwt_secret = args.jwt_secret
        env_config.vllm_api_key = os.getenv("VLLM_API_KEY")
    env_config.service_port = runtime_config.service_port
    env_config.vllm_model = model_spec.hf_model_repo
    env_config.deploy_url = deploy_url

    prompt_client = PromptClient(
        env_config,
        model_spec=model_spec,
        runtime_config=runtime_config,
    )
    if remote_server:
        if not _wait_for_remote_openai_ready(prompt_client):
            logger.error(
                "⛔️ Remote inference endpoint is not ready. Aborting tests."
            )
            return 1
    elif not prompt_client.wait_for_healthy():
        logger.error("⛔️ vLLM server is not healthy. Aborting tests.")
        return 1

    # Create a single shared output directory for all tasks in this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir_path = (
        Path(args.output_path) / f"test_{model_spec.model_id}__{run_timestamp}"
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Test output directory: {output_dir_path}")

    # Execute pytest for each task.
    logger.info(
        f"Running test client with {len(test_config.tasks)} task(s): {[t.task_name for t in test_config.tasks]}"
    )
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
            str(output_dir_path),
            runtime_config.service_port,
            deploy_url=deploy_url,
        )
        return_code = run_command(command=cmd, logger=logger, env=env_vars)
        return_codes.append(return_code)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed tests")
        return 0
    logger.error(
        f"⛔ tests failed with return codes: {return_codes}. See logs above for details."
    )
    # tests are scored against acceptance criteria
    return 0


if __name__ == "__main__":
    sys.exit(main())
