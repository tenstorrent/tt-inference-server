# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
from datetime import datetime
from pathlib import Path

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient


model_evals = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": [
        ("leaderboard_ifeval", {}),
        ("gpqa_diamond_cot_zeroshot", {}),
        ("mmlu_pro", {}),
    ]
}


def run_command(cmd, log_file, env):
    """
    Run a command using subprocess.Popen and wait for its completion.
    Exits the script if the command fails.
    """
    print(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=log_file, stderr=log_file, text=True, env=env
    )
    process.wait()
    if process.returncode != 0:
        print(f"Command {' '.join(cmd)} failed with return code {process.returncode}")


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
    # set eval env vars
    # TODO: figure out the correct MESH_DEVICE, use same logic as run vllm script
    os.environ["MESH_DEVICE"] = "T3K"
    env_vars = os.environ.copy()
    hf_model_repo_id = env_vars["HF_MODEL_REPO_ID"]

    # start vLLM inference server
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vllm_log_file_path = (
        Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"run_vllm_{log_timestamp}.log"
    )
    vllm_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # note: line buffering of log file to reduce disk IO overhead on vLLM run process
    # run_vllm_api_server.py uses runpy so vLLM process inherits sys.stdour and std.stderr
    vllm_log = open(vllm_log_file_path, "w", buffering=1)
    print("running vllm server ...")
    vllm_process = subprocess.Popen(
        ["python", "/home/container_app_user/app/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )

    # note: eval script uses capture_traces, which will wait for
    # vLLM health endpoint to return a 200 OK
    # it will wait up to 300s by default.
    eval_log_file_path = (
        Path(os.getenv("CACHE_ROOT", "."))
        / "logs"
        / f"run_eval_client_{log_timestamp}.log"
    )
    eval_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    eval_log = open(eval_log_file_path, "w", buffering=1)
    print("running vllm evals client ...")
    # Define the common arguments as a dictionary.
    common_args_dict = {
        "model": "local-completions",
        "model_args": f"model={hf_model_repo_id},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface",
        "gen_kwargs": "stream=False",
        "output_path": "/home/container_app_user/cache_root/eval_output",
        "seed": "42",
        "apply_chat_template": None,  # Flag argument (no value)
        "log_samples": None,
    }

    # Convert the dictionary to a list of command-line arguments.
    common_args_list = dict_to_args(common_args_dict)

    # Define the evaluation tasks.
    tasks = model_evals[hf_model_repo_id]

    # wait for vllm server is ready
    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    prompt_client.capture_traces(timeout=1200.0)
    # Execute lm_eval for each task.
    for task_name, task_arg in tasks:
        print(f"Running lm_eval for {task_name} (args: {task_arg}) ...")
        cmd = ["lm_eval", "--tasks", task_name] + common_args_list
        run_command(cmd=cmd, log_file=eval_log, env=env_vars)

    # Return to the original directory.
    print("All commands executed successfully.")
    print("✅ vllm evals completed!")
    # terminate and wait for graceful shutdown of vLLM server
    vllm_process.terminate()
    vllm_process.wait()
    print("✅ vllm shutdown.")
    eval_log.close()
    vllm_log.close()
    # TODO: extract eval output


if __name__ == "__main__":
    main()
