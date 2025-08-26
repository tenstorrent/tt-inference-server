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
from utils.image_client import ImageClient

from workflows.model_spec import ModelSpec, ModelType
from workflows.workflow_config import (
    WORKFLOW_EVALS_CONFIG,
)
from workflows.utils import run_command
from evals.eval_config import EVAL_CONFIGS, EvalTask
from workflows.workflow_venvs import VENV_CONFIGS
from workflows.workflow_types import WorkflowVenvType, DeviceTypes
from workflows.log_setup import setup_workflow_script_logger
from evals.eval_utils import get_coco_dataset
from evals.coco_utils import run_yolov4_coco_evaluation

logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM evals")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for evaluation output",
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


def build_eval_command(
    task: EvalTask,
    model_spec,
    device,
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


    optional_model_args = []
    if task.max_concurrent:
        if task.eval_class != "openai_compatible":
            optional_model_args.append(f"num_concurrent={task.max_concurrent}")

    # newer lm-evals expect full completions api route
    _base_url = (
        base_url if task.workflow_venv_type == WorkflowVenvType.EVALS_META else api_url
    )

    if task.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
        os.environ['OPENAI_API_BASE'] = base_url
    
    if task.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lmms-eval"
    else:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lm_eval"

    model_kwargs_list = [f"{k}={v}" for k, v in task.model_kwargs.items()]
    model_kwargs_list += optional_model_args
    model_kwargs_str = ",".join(model_kwargs_list)

    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in task.gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)

    # set output_dir
    # results go to {output_dir_path}/{hf_repo}/results_{timestamp}
    output_dir_path = Path(output_path) / f"eval_{model_spec.model_id}"

    # fmt: off
    if task.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model_version={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
            "--seed", task.seed,
            "--num_fewshot", task.num_fewshot,
            "--batch_size", task.batch_size,
            "--log_samples",
            "--show_config",
        ]
    else:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
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

    # Apply optional per-task sample limit for faster debugging cycles
    if task.limit_samples is not None:
        cmd.extend(["--limit", str(task.limit_samples)])

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


def wait_for_cnn_server_health(image_client: ImageClient, timeout_seconds: int = 300) -> tuple[bool, str | None]:
    """
    Wait for CNN server to be healthy with retry logic.
    
    Args:
        image_client: ImageClient instance to check health
        timeout_seconds: Maximum time to wait in seconds (default: 5 minutes)
        
    Returns:
        tuple: (health_status: bool, runner_in_use: str or None)
    """
    import time
    from urllib3.exceptions import NewConnectionError
    from requests.exceptions import ConnectionError
    
    start_time = time.time()
    retry_interval = 5  # seconds between retries
    
    health_status = False
    runner_in_use = None
    
    while time.time() - start_time < timeout_seconds:
        try:
            # ImageClient.get_health() includes retry logic and raises exception on failure
            health_status, runner_in_use = image_client.get_health()
            if health_status:
                logger.info(f"✅ CNN server is healthy. Runner in use: {runner_in_use}")
                break
            else:
                logger.warning("⚠️ CNN server returned unhealthy status, retrying...")
        except (ConnectionError, NewConnectionError) as e:
            elapsed = int(time.time() - start_time)
            remaining = int(timeout_seconds - elapsed)
            logger.info(f"⏳ CNN server connection failed (elapsed: {elapsed}s, remaining: {remaining}s). Retrying in {retry_interval}s...")
            if remaining <= 0:
                break
        except Exception as e:
            elapsed = int(time.time() - start_time)
            remaining = int(timeout_seconds - elapsed)
            logger.warning(f"⚠️ CNN server health check error: {e} (elapsed: {elapsed}s, remaining: {remaining}s). Retrying in {retry_interval}s...")
            if remaining <= 0:
                break
        
        time.sleep(retry_interval)
    
    return health_status, runner_in_use


def run_coco_evaluation_task(task: EvalTask, model_spec, cli_args, output_path):
    """Run COCO object detection evaluation."""
    logger.info(f"Running COCO evaluation task: {task.task_name}")
    
    # Get COCO dataset, downloading if necessary
    coco_dataset_path, coco_annotations_path = get_coco_dataset()
    logger.info(f"Using COCO dataset from: {coco_dataset_path}")

    # Extract COCO-specific parameters
    max_images = task.model_kwargs.get("max_images")
    
    # Run COCO evaluation
    metrics = run_yolov4_coco_evaluation(
        service_port=cli_args.get("service_port"),
        output_path=output_path,
        coco_dataset_path=coco_dataset_path,
        coco_annotations_path=coco_annotations_path,
        max_images=max_images,
        jwt_secret=os.getenv("JWT_SECRET")
    )
    
    return metrics


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    device_str = cli_args.get("device")
    disable_trace_capture = cli_args.get("disable_trace_capture", False)

    device = DeviceTypes.from_string(device_str)
    workflow_config = WORKFLOW_EVALS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_spec=: {model_spec}")
    logger.info(f"device=: {device_str}")
    assert device == model_spec.device_type

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
    if model_spec.model_name not in EVAL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_spec.model_name}"
        )
    eval_config = EVAL_CONFIGS[model_spec.model_name]

    # handle by model type
    if model_spec.model_type == ModelType.LLM:
        # Standard LLM evaluation path
        logger.info("Wait for the vLLM server to be ready ...")
        env_config = EnvironmentConfig()
        env_config.jwt_secret = args.jwt_secret
        env_config.service_port = cli_args.get("service_port")
        env_config.vllm_model = model_spec.hf_model_repo

        prompt_client = PromptClient(env_config)
        if not prompt_client.wait_for_healthy(timeout=30 * 60.0):
            logger.error("⛔️ vLLM server is not healthy. Aborting evaluations. ")
            return 1

        if not disable_trace_capture:
            prompt_client.capture_traces()

        # Execute lm_eval for each task.
        logger.info("Running vLLM evals client ...")
        return_codes = []
        for task in eval_config.tasks:
            health_check = prompt_client.get_health()
            if health_check.status_code != 200:
                logger.error("⛔️ vLLM server is not healthy. Aborting evaluations.")
                return 1

            logger.info(
                f"Starting workflow: {workflow_config.name} task_name: {task.task_name}"
            )

            logger.info(f"Running lm_eval for:\n {task}")
            cmd = build_eval_command(
                task,
                model_spec,
                device_str,
                args.output_path,
                cli_args.get("service_port"),
            )
            return_code = run_command(command=cmd, logger=logger, env=env_vars)
            return_codes.append(return_code)
    elif model_spec.model_type == ModelType.CNN:
        logger.info("Running CNN (YOLOv4) COCO object detection evaluation...")
        
        # Wait for server to be ready using ImageClient for CNN models
        service_port = cli_args.get("service_port")
        image_client = ImageClient(
            all_params=None,  # Not used for health checks
            model_spec=model_spec,
            device=device,
            output_path=args.output_path,
            service_port=service_port
        )
        
        # Wait for CNN server to be healthy (5-minute timeout)
        health_status, runner_in_use = wait_for_cnn_server_health(image_client, timeout_seconds=300)
        
        if not health_status:
            logger.error(f"⛔️ CNN server health check failed after 5 minutes. Aborting evaluation.")
            return 1
        
        # Note: CNN models don't need trace capture like vLLM models
        # The trace capture is handled within the CNN inference pipeline
        
        # Run CNN evaluation tasks
        return_codes = []
        for task in eval_config.tasks:           
            logger.info(f"Starting CNN evaluation: {task.task_name}")
            if task.task_name == "coco_detection_val2017":
                try:
                    metrics = run_coco_evaluation_task(task, model_spec, cli_args, args.output_path)
                    
                    # Calculate score using the task's scoring function
                    if task.score:
                        score = task.score.score_func(
                            {task.task_name: metrics}, 
                            task.task_name, 
                            task.score.score_func_kwargs
                        )
                        logger.info(f"✅ COCO evaluation score: {score:.4f}")
                        
                        if task.score.published_score:
                            ratio = score / task.score.published_score
                            logger.info(f"Published score ratio: {ratio:.4f}")
                        
                        if task.score.gpu_reference_score:
                            ratio = score / task.score.gpu_reference_score
                            logger.info(f"Reference score ratio: {ratio:.4f}")
                    
                    return_codes.append(0)
                    
                except Exception as e:
                    logger.error(f"⛔ CNN evaluation ({task.task_name}) failed: {e}")
                    return_codes.append(1)

    if all(return_code == 0 for return_code in return_codes):
        logger.info("✅ Completed evals")
        main_return_code = 0
    else:
        logger.error(
            f"⛔ evals failed with return codes: {return_codes}. See logs above for details."
        )
        main_return_code = 1

    return main_return_code


if __name__ == "__main__":
    sys.exit(main())
