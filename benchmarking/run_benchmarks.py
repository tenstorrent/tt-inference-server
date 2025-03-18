# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path

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
    WORKFLOW_BENCHMARKS_CONFIG,
)
from workflows.utils import run_command
from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from workflows.workflow_venvs import VENV_CONFIGS
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
        "--trace-capture",
        action="store_true",
        help="Run tracing prompts at different input sequence lengths",
    )
    parser.add_argument(
        "--service-port",
        type=str,
        help="inference server port",
        default=os.getenv("SERVICE_PORT", "8000"),
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


def build_benchmark_command(
    task, benchmark_script, params, args, benchmark_config, model_config
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    isl = params["input_len"]
    osl = params["output_len"]
    max_concurrency = params["max_concurrency"]
    num_prompts = params["num_prompts"]
    result_filename = (
        Path(args.output_path)
        / f"benchmark_{run_timestamp}_{model_config.model_name}_{args.mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrency}_n-{num_prompts}.json"
    )
    vllm_dir = os.environ.get("vllm_dir")
    assert vllm_dir is not None, "vllm_dir must be set."

    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    # fmt: off
    cmd = [
        str(task_venv_config.venv_python), str(benchmark_script),
        "--backend", "vllm",
        "--model", model_config.hf_model_repo,
        "--port", str(args.service_port),
        "--dataset-name", "random",
        "--max-concurrency", str(params["max_concurrency"]),
        "--num-prompts", str(params["num_prompts"]),
        "--random-input-len", str(params["input_len"]),
        "--random-output-len", str(params["output_len"]),
        "--ignore-eos",  # Ignore EOS tokens to force max output length as set
        "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
        "--save-result",
        "--result-filename", str(result_filename),
    ]
    # fmt: on
    return cmd


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)

    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    workflow_config = WORKFLOW_BENCHMARKS_CONFIG
    logger.info(f"workflow_config=: \n{workflow_config}\n")
    logger.info(f"model_config=: \n{model_config}\n")

    vllm_dir = os.getenv("vllm_dir")
    if vllm_dir is None:
        raise ValueError("vllm_dir must be set.")
    benchmark_script = Path(vllm_dir) / "benchmarks" / "benchmark_serving.py"

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
    if model_config.hf_model_repo not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.hf_model_repo}"
        )
    benchmark_config = BENCHMARK_CONFIGS[model_config.hf_model_repo]

    # Get all benchmark combinations using the original function
    # fmt: off
    combinations = [
        # sweeps for batch-1 (max_concurrency=1)
        {"input_len": 128, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 4},
        {"input_len": 2048, "output_len": 128, "max_concurrency": 1, "num_prompts": 32},
        {"input_len": 3000, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 8},
        {"input_len": 4096, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 4},
        {"input_len": 8192, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 2},
        {"input_len": 16384, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 2},
        # sweeps for batch-32 (max_concurrency=32)
        {"input_len": 128, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 16},
        {"input_len": 128, "output_len": 1024, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 2048, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 2048, "output_len": 2048, "max_concurrency": 32, "num_prompts": 32 * 4},
        {"input_len": 3000, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 3900, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 4500, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
    ]
    # fmt: on

    context_lens = [(it["input_len"], it["output_len"]) for it in combinations]
    # de-dupe
    context_lens = list(set(context_lens))

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = args.jwt_secret
    env_config.service_port = args.service_port
    env_config.vllm_model = model_config.hf_model_repo
    prompt_client = PromptClient(env_config)
    prompt_client.wait_for_healthy(timeout=7200.0)
    if not args.trace_capture:
        prompt_client.capture_traces(context_lens=context_lens, timeout=1200.0)

    # Run benchmarks
    for task in benchmark_config.tasks:
        for i, params in enumerate(combinations, 1):
            logger.info(f"\nRunning benchmark {i}/{len(combinations)}")
            # Add a small delay between runs to ensure system stability
            time.sleep(2)
            cmd = build_benchmark_command(
                task,
                benchmark_script,
                params=params,
                args=args,
                benchmark_config=benchmark_config,
                model_config=model_config,
            )
            run_command(command=cmd, logger=logger, env=env_vars)

    logger.info("Benchmark suite completed")


if __name__ == "__main__":
    main()
