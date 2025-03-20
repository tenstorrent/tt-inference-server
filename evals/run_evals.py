# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import List

import jwt

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_config import (
    WORKFLOW_EVALS_CONFIG,
)
from workflows.utils import run_command
from evals.eval_config import EVAL_CONFIGS, EvalTask
from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import WorkflowVenvType
from workflows.log_setup import setup_workflow_script_logger

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
        "--mesh-device",
        type=str,
        help="MESH_DEVICE used to simulate different hardware configurations",
        default=os.getenv("MESH_DEVICE", "T3K"),
    )
    # optional
    parser.add_argument(
        "--run-server",
        action="store_true",
        help="Start the vLLM inference server (otherwise assume it is already running)",
    )
    parser.add_argument(
        "--service-port",
        type=str,
        help="inference server port",
        default=os.getenv("SERVICE_PORT", "8000"),
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already runnning and traces captured.",
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


def build_eval_command(
    task: EvalTask,
    model_config,
    output_path,
    service_port,
) -> List[str]:
    """
    Build the command for lm_eval by templating command-line arguments using properties
    from the given evaluation task and model configuration.
    """
    base_url = f"http://127.0.0.1:{service_port}/v1"
    eval_class = task.eval_class
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    if task.use_chat_api:
        # dont double apply the chat template
        assert not task.apply_chat_template, "chat api already applies chat template"
        # chat end point applies chat template by default, this is required for most instruct models
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/completions"

    lm_eval_exec = task_venv_config.venv_path / "bin" / "lm_eval"

    if task.max_concurrent:
        if task.eval_class == "local-mm-chat-completions":
            concurrent_users_str = f"num_concurrent={task.max_concurrent}"
        else:
            concurrent_users_str = f"max_concurrent={task.max_concurrent}"
    else:
        # concurrent_users_str = f"batch_size={task.batch_size}"
        concurrent_users_str = ""
    # newer lm-evals expect full completions api route
    _base_url = (
        base_url if task.workflow_venv_type == WorkflowVenvType.EVALS_META else api_url
    )
    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in task.gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)
    # fmt: off
    cmd = [
        str(lm_eval_exec),
        "--tasks", task.task,
        "--model", eval_class,
        "--model_args", (
            f"model={model_config.hf_model_repo},"
            f"base_url={_base_url},"
            f"tokenizer_backend={task.tokenizer_backend},"
            f"{concurrent_users_str}"
        ),
        "--gen_kwargs", gen_kwargs_str,
        "--output_path", output_path,
        "--seed", task.seed,
        "--num_fewshot", task.num_fewshot,
        "--batch_size", task.batch_size,
        "--log_samples",
        "--show_config",
    ]
    # fmt: on

    if task.include_path:
        cmd.append("--include_path")
        cmd.append(task_venv_config.venv_path / task.include_path)
        os.chdir(task_venv_config.venv_path)
    if task.apply_chat_template:
        cmd.append("--apply_chat_template")  # Flag argument (no value)

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    workflow_config = WORKFLOW_EVALS_CONFIG
    logger.info(f"workflow_config=: \n{workflow_config}\n")
    logger.info(f"model_config=: \n{model_config}\n")

    # set environment vars
    os.environ["MESH_DEVICE"] = args.mesh_device
    os.environ["HF_MODEL_REPO_ID"] = model_config.hf_model_repo
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
    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

    # Look up the evaluation configuration for the model using EVAL_CONFIGS.
    if model_config.hf_model_repo not in EVAL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.hf_model_repo}"
        )
    eval_config = EVAL_CONFIGS[model_config.model_name]

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = args.jwt_secret
    env_config.service_port = args.service_port
    env_config.vllm_model = model_config.hf_model_repo
    prompt_client = PromptClient(env_config)
    prompt_client.wait_for_healthy(timeout=7200.0)
    if not args.disable_trace_capture:
        prompt_client.capture_traces()

    # Execute lm_eval for each task.
    logger.info("Running vLLM evals client ...")
    for task in eval_config.tasks:
        logger.info(f"Starting workflow: {workflow_config.name} task: {task.task}")
        logger.info(f"Running lm_eval for:\n {task}")
        cmd = build_eval_command(
            task, model_config, args.output_path, args.service_port
        )
        run_command(command=cmd, logger=logger, env=env_vars)

    logger.info("✅ Completed evals")


if __name__ == "__main__":
    main()
