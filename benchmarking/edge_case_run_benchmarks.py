# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import argparse

import subprocess
from datetime import datetime
from pathlib import Path
from utils.prompt_configs import EnvironmentConfig

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

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_seq", type=int, help="Run the max_seq process (hyperparameter value)")
    group.add_argument("--continuous_batch", type=int, help="Run the continuous_batch process (hyperparameter value)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_size", type=int, help="Input token length")
    group.add_argument("--output_size", type=int, help="Output token length")

    parser.add_argument("--batch_size", type=int, default=1, help="Optional Batch Size AKA max_concurrent (default: 8).")
    parser.add_argument("--users", type=int, default=1, help="Optional number of Users AKA Num Requests (default: 8).")

    args = parser.parse_args()
    print(f"Processing with arguments: {args}")
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

    env_vars = os.environ.copy()
    # start vLLM inference server
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    env_config = EnvironmentConfig()
    # set benchmarking env vars
    # TODO: figure out the correct MESH_DEVICE, use same logic as run vllm script
    env_vars["MESH_DEVICE"] = env_config.mesh_device
    # note: benchmarking script uses capture_traces, which will wait for
    # vLLM health endpoint to return a 200 OK
    # it will wait up to 300s by default.
    benchmark_log_file_path = (
        Path(os.getenv("CACHE_ROOT", "."))
        / "logs"
        / f"2_run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print("running vllm benchmarks client ...")
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
    # wait for benchmark script to finish
    benchmark_process.wait()
    print("✅ vllm benchmarks completed!")
    # terminate and wait for graceful shutdown of vLLM server
    vllm_process.terminate()
    vllm_process.wait()
    print("✅ vllm shutdown.")
    benchmark_log.close()
    vllm_log.close()
    # TODO: extract benchmarking output


if __name__ == "__main__":
    main()
