# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
import os
import subprocess
import argparse
import logging
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
from workflows.workflow_config import EVALS_CONFIG, SERVER_CONFIG

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
        "--log-path",
        type=str,
        help="Path for log output",
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
        "--server-port",
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


def run_server(env_vars, log_timestamp, run_script_path):
    """
    Start the vLLM inference server.

    This function creates a timestamped log file, starts the server process with
    line buffering to reduce disk IO overhead, and returns both the process and the log file.
    """
    vllm_log_file_path = (
        SERVER_CONFIG.workflow_log_dir / f"run_vllm_{log_timestamp}.log"
    )
    vllm_log_file_path.parent.mkdir(parents=False, exist_ok=True)
    vllm_log = open(vllm_log_file_path, "w", buffering=1)
    logging.info("Running vLLM server ...")
    vllm_process = subprocess.Popen(
        ["python", run_script_path],
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
    model_config = MODEL_CONFIGS[args.model]

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
        logging.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    # copy env vars to pass to subprocesses
    env_vars = os.environ.copy()

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
    eval_log_file_path = Path(args.log_path) / f"run_eval_client_{log_timestamp}.log"
    eval_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log = open(eval_log_file_path, "w", buffering=1)

    logging.info("Running vLLM evals client ...")
    # Define common arguments for lm_eval.
    # TODO add server port
    common_args_dict = {
        "model": "local-chat-completions",
        "model_args": f"model={model_config.hf_model_repo},base_url=http://127.0.0.1:7000/v1/chat/completions,tokenizer_backend=huggingface",
        "gen_kwargs": "stream=False",
        "output_path": args.output_path,
        "seed": "42",
        "apply_chat_template": None,  # Flag argument (no value)
        "log_samples": None,
    }
    common_args_list = dict_to_args(common_args_dict)

    # Define the evaluation tasks.
    if model_config.hf_model_repo not in model_evals:
        raise ValueError(
            f"No evaluation tasks defined for model: {model_config.hf_model_repo}"
        )
    tasks = model_evals[model_config.hf_model_repo]

    # Wait for the vLLM server to be ready.
    # TODO: make this optional via runtime arg
    if False:
        env_config = EnvironmentConfig()
        env_config.vllm_model = model_config.hf_model_repo
        prompt_client = PromptClient(env_config)
        prompt_client.capture_traces(timeout=1200.0)

    # Execute lm_eval for each task.
    lm_eval_exec = EVALS_CONFIG.venv_path / "bin" / "lm_eval"
    for task_name, task_arg in tasks:
        logging.info("Running lm_eval for %s (args: %s) ...", task_name, task_arg)
        cmd = [str(lm_eval_exec), "--tasks", task_name] + common_args_list
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
