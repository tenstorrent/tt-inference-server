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
from workflows.workflow_types import DeviceTypes


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


def build_benchmark_command(
    task, benchmark_script, params, args, benchmark_config, model_config
):
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    isl = params.isl
    osl = params.osl
    max_concurrency = params.max_concurrency
    num_prompts = params.num_prompts
    result_filename = (
        Path(args.output_path)
        / f"benchmark_{model_config.model_name}_{args.device}_{run_timestamp}_isl-{isl}_osl-{osl}_maxcon-{max_concurrency}_n-{num_prompts}.json"
    )

    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    # fmt: off
    cmd = [
        str(task_venv_config.venv_python), str(benchmark_script),
        "--backend", "vllm",
        "--model", model_config.hf_model_repo,
        "--port", str(args.service_port),
        "--dataset-name", "random",
        "--max-concurrency", str(max_concurrency),
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
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
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_config = MODEL_CONFIGS[args.model]
    device = DeviceTypes.from_string(args.device)
    workflow_config = WORKFLOW_BENCHMARKS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    logger.info(f"service_port=: {args.service_port}")
    logger.info(f"output_path=: {args.output_path}")

    # set environment vars
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
    if model_config.model_name not in BENCHMARK_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.model_name}"
        )
    benchmark_config = BENCHMARK_CONFIGS[model_config.model_name]

    logger.info("Wait for the vLLM server to be ready ...")
    env_config = EnvironmentConfig()
    env_config.jwt_secret = args.jwt_secret
    env_config.service_port = args.service_port
    env_config.vllm_model = model_config.hf_model_repo
    prompt_client = PromptClient(env_config)
    prompt_client.wait_for_healthy(timeout=7200.0)

    # keep track of captured traces to avoid re-running requests
    captured_traces = set()

    # Run benchmarks
    for task in benchmark_config.tasks:
        venv_config = VENV_CONFIGS[task.workflow_venv_type]
        benchmark_script = venv_config.venv_path / "scripts" / "benchmark_serving.py"
        params_list = task.param_map[device]
        context_lens = [(params.isl, params.osl) for params in params_list]
        # de-dupe
        context_lens_set = set(context_lens)
        context_lens_set.difference_update(captured_traces)
        if not args.disable_trace_capture:
            prompt_client.capture_traces(
                context_lens=list(context_lens_set), timeout=1200.0
            )
            captured_traces.update(context_lens_set)
        for i, params in enumerate(params_list, 1):
            logger.info(
                f"Running benchmark {model_config.model_name}: {i}/{len(params_list)}"
            )
            # Add a small delay between runs to ensure system stability
            time.sleep(2)
            cmd = build_benchmark_command(
                task,
                benchmark_script,
                args=args,
                params=params,
                benchmark_config=benchmark_config,
                model_config=model_config,
            )
            run_command(command=cmd, logger=logger, env=env_vars)

    logger.info("✅ Completed benchmarks")


if __name__ == "__main__":
    main()
