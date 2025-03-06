# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import argparse

import subprocess
from datetime import datetime
from pathlib import Path
import itertools

# Path to Python environment
PYTHON_ENV = "/home/stisi/tt-metal/python_env/bin/python"

# Load environment variables
ENV_FILE = "model_envs/env_benchmarking.env"

def generate_combinations():
    """Generates argument combinations for benchmark runs."""
    max_seq_values = ["8192", "1212"]
    continuous_batch_values = ["8192", "1212"]
    input_size_values = ["1024", "256"]
    output_size_values = ["512", "128"]
    batch_size_values = ["1", "8"]
    users_values = ["1", "16"]

    benchmark_combinations = []

    # Max_seq Mode (Mutually exclusive with batch_size & users)
    for max_seq in max_seq_values:
        for input_size in input_size_values:
            benchmark_combinations.append(["--max_seq", max_seq, "--input_size", input_size])
        for output_size in output_size_values:
            benchmark_combinations.append(["--max_seq", max_seq, "--output_size", output_size])

    # Continuous Batch Mode (Explores batch_size and users separately)
    for continuous_batch in continuous_batch_values:
        for input_size in input_size_values + output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append(
                    ["--continuous_batch", continuous_batch, "--input_size", input_size, "--batch_size", batch_size, "--users", users]
                )
        for output_size in output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append(
                    ["--continuous_batch", continuous_batch, "--output_size", output_size, "--batch_size", batch_size, "--users", users]
                )

    return benchmark_combinations

# Load environment variables from model_envs/env_benchmarking.env
def load_env_variables():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

def run_benchmark(args, env_vars):
    """Runs a single benchmark process with given arguments."""
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_log_file_path = (
            Path(os.getenv("CACHE_ROOT", "."))
            / "logs"
            / f"4_run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print(f"Running benchmark with args: {args}")
    benchmark_command = [
                            PYTHON_ENV, "/home/stisi/tt-inference-server/benchmarking/edge_case_vllm_execute_benchmark.py"
                        ] + args  # Append arguments dynamically

    benchmark_process = subprocess.Popen(
        benchmark_command,
        stdout=benchmark_log,
        stderr=benchmark_log,
        text=True,
        env=env_vars,
    )
    return benchmark_log, benchmark_process


def process_max_seq(hyperparam):
    # Your logic for the max_seq process
    result = {
        "process": "--max_seq",
        "max_seq": hyperparam.max_seq,
    }
    return result

def process_continuous_batch(hyperparam):
    # Your logic for the continuous_batch process
    result = {
        "process": "--continuous_batch",
        "max_seq": hyperparam.continuous_batch,
    }
    return result



def start_server(env_vars, log_timestamp):
    vllm_log_file_path = (
            Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"1_start_vllm_{log_timestamp}.log"
    )
    vllm_log = open(vllm_log_file_path, "w")
    print("running vllm server ...")
    vllm_process = subprocess.Popen(
        ["python", "-u", "/home/stisi/tt-inference-server/vllm-tt-metal-llama3/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        text=True,
        env=env_vars,
    )
    return vllm_log, vllm_process


def execute_edge_case(args, env_vars, log_timestamp):
    benchmark_log_file_path = (
            Path(os.getenv("CACHE_ROOT", "."))
            / "logs"
            / f"2_run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print("running vllm benchmarks client ...")
    result = initialize_from_args(args)
    benchmark_process = subprocess.Popen(
        [
            "python",
            "-u",
            "/home/stisi/tt-inference-server/benchmarking/edge_case_vllm_execute_benchmark.py",
            result["process"],
            str(result["max_seq"]),
            result["token_picked"],
            str(result["token_size"]),
            "--users",
            str(args.users),
            "--batch_size",
            str(args.batch_size),
        ],
        stdout=benchmark_log,
        stderr=benchmark_log,
        text=True,
        env=env_vars,
    )
    return benchmark_log, benchmark_process


def initialize_from_args(args):
    if args.max_seq is not None:
        result = process_max_seq(args)
    elif args.continuous_batch is not None:
        result = process_continuous_batch(args)
    if args.input_size is not None:
        result["token_size"] = args.input_size
        result["token_picked"] = "--input_size"
    elif args.output_size is not None:
        result["token_size"] = args.output_size
        result["token_picked"] = "--output_size"
    else:
        result["token_size"] = 16
        result["token_picked"] = "--output_size"
    return result


def read_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_seq", type=int, help="Run the max_seq process (hyperparameter value)")
    group.add_argument("--continuous_batch", type=int, help="Run the continuous_batch process (hyperparameter value)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_size", type=int, help="Input token length")
    group.add_argument("--output_size", type=int, help="Output token length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Optional Batch Size AKA max_concurrent (default: 8).")
    parser.add_argument("--users", type=int, default=1, help="Optional number of Users AKA Num Requests (default: 8).")
    args = parser.parse_args()
    print(f"Processing with arguments: {args}")
    return args

def main():
    load_env_variables()
    args = read_args()
    env_vars = os.environ.copy()

    # start vLLM inference server
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    vllm_log, vllm_process = start_server(env_vars, log_timestamp)

    # Run all benchmark combinations
    benchmark_combinations = generate_combinations()
    for benchmark_args in benchmark_combinations:
        benchmark_log, benchmark_process = run_benchmark(benchmark_args, env_vars)
        benchmark_process.wait()
        benchmark_log.close()

    # note: benchmarking script uses capture_traces, which will wait for
    # vLLM health endpoint to return a 200 OK
    # it will wait up to 300s by default.

    benchmark_log, benchmark_process = execute_edge_case(args, env_vars, log_timestamp)

    # wait for benchmark script to finish
    benchmark_process.wait()
    print("✅ vllm benchmarks completed!")
    # terminate and wait for graceful shutdown of vLLM server
    vllm_process.terminate()
    vllm_process.wait()
    print("✅ vllm shutdown.")
    benchmark_log.close()
    vllm_log.close()

if __name__ == "__main__":
    main()
