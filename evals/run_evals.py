# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

import jwt

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient

model_evals = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": [
        ("leaderboard_ifeval", {}),
        ("gpqa_diamond_cot_zeroshot", {}),
        ("mmlu_pro", {}),
    ],
    "Qwen/Qwen2.5-72B-Instruct": [
        ("leaderboard_ifeval", {}),
        ("gpqa_diamond_cot_zeroshot", {}),
        ("mmlu_pro", {}),
    ],
}


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run vLLM evals")
    parser.add_argument(
        "--run_server",
        action="store_true",
        help="Start the vLLM inference server (otherwise assume it is already running)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to evaluate (overrides HF_MODEL_REPO_ID environment variable)",
        default=None,
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set OPENAI_API_KEY",
        default=None,
    )
    # TODO: wire these up
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for evaluation output",
        default=".",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="Path for evaluation output",
        default="/home/container_app_user/cache_root/eval_output",
    )
    return parser.parse_args()


def run_server(env_vars, log_timestamp):
    """
    Start the vLLM inference server.

    This function creates a timestamped log file, starts the server process with
    line buffering to reduce disk IO overhead, and returns both the process and the log file.
    """
    vllm_log_file_path = (
        Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"run_vllm_{log_timestamp}.log"
    )
    vllm_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_log = open(vllm_log_file_path, "w", buffering=1)
    logging.info("Running vLLM server ...")
    vllm_process = subprocess.Popen(
        ["python", "/home/container_app_user/app/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )
    return vllm_process, vllm_log


def run_command(cmd, log_file, env):
    """
    Run a command using subprocess.Popen and wait for its completion.
    Exits the script if the command fails.
    """
    logging.info("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=log_file, text=True, env=env
    )
    process.wait()
    if process.returncode != 0:
        logging.error(
            "Command %s failed with return code %d", " ".join(cmd), process.returncode
        )


def dict_to_args(arg_dict):
    """
    Convert a dictionary of arguments into a list of command-line arguments.
    If a value is None, only the flag (e.g., --log_samples) is added.
    Otherwise, both the flag and its value are added.
    """
    args = []
    for key, value in arg_dict.items():
        args.append(f"--{key}")
        if value is not None:
            args.append(str(value))
    return args


def main():
    # Setup logging configuration.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()

    # If jwt-secret is provided, generate the JWT and set OPENAI_API_KEY.
    if args.jwt_secret:
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, args.jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logging.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )

    # Set evaluation environment variables.
    # TODO: programmitcally determine this
    os.environ["MESH_DEVICE"] = "T3K"
    env_vars = os.environ.copy()

    # Determine the model name either from CLI argument or environment variable.
    hf_model_repo_id = args.model or env_vars.get("HF_MODEL_REPO_ID")
    if hf_model_repo_id is None:
        raise ValueError(
            "Model name must be provided via --model or HF_MODEL_REPO_ID environment variable"
        )

    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Optionally start the vLLM server.
    if args.run_server:
        vllm_process, vllm_log = run_server(env_vars, log_timestamp)
    else:
        logging.info(
            "Skipping vLLM server startup. Assuming server is already running."
        )
        vllm_process = None
        vllm_log = None

    # Prepare the evaluation log file.
    eval_log_file_path = (
        Path(os.getenv("CACHE_ROOT", "."))
        / "logs"
        / f"run_eval_client_{log_timestamp}.log"
    )
    eval_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log = open(eval_log_file_path, "w", buffering=1)

    logging.info("Running vLLM evals client ...")
    # Define common arguments for lm_eval.
    common_args_dict = {
        "model": "local-completions",
        "model_args": f"model={hf_model_repo_id},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface",
        "gen_kwargs": "stream=False",
        "output_path": "/home/container_app_user/cache_root/eval_output",
        "seed": "42",
        "apply_chat_template": None,  # Flag argument (no value)
        "log_samples": None,
    }
    common_args_list = dict_to_args(common_args_dict)

    # Define the evaluation tasks.
    if hf_model_repo_id not in model_evals:
        raise ValueError(f"No evaluation tasks defined for model: {hf_model_repo_id}")
    tasks = model_evals[hf_model_repo_id]

    # Wait for the vLLM server to be ready.
    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    prompt_client.capture_traces(timeout=1200.0)

    # Execute lm_eval for each task.
    for task_name, task_arg in tasks:
        logging.info("Running lm_eval for %s (args: %s) ...", task_name, task_arg)
        cmd = ["lm_eval", "--tasks", task_name] + common_args_list
        run_command(cmd=cmd, log_file=eval_log, env=env_vars)

    logging.info("All commands executed successfully.")
    logging.info("✅ vllm evals completed!")

    # If we started the server, shut it down gracefully.
    if vllm_process is not None:
        vllm_process.terminate()
        vllm_process.wait()
        logging.info("✅ vLLM shutdown.")
    eval_log.close()
    if vllm_log is not None:
        vllm_log.close()


if __name__ == "__main__":
    main()
