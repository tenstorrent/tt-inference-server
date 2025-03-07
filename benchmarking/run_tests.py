# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import argparse

from datetime import datetime
import itertools

from tests.tools.essentials import load_env_variables
load_env_variables()
from tests import *

def generate_combinations():
    """Generates argument combinations for benchmark runs."""
    max_seq_values = [8192, 1212]
    continuous_batch_values = [8192, 1212]
    input_size_values = [512, 256]
    output_size_values = [128, 256]
    batch_size_values = [1, 5]
    users_values = [1, 4]

    benchmark_combinations = []

    # Max_seq Mode (Mutually exclusive with batch_size & users)
    for max_seq in max_seq_values:
        for output_size in output_size_values:
            benchmark_combinations.append({
                "max_seq": max_seq,
                "output_size": output_size,
                "input_size": None
            })
        for input_size in input_size_values:
            benchmark_combinations.append({
                "max_seq": max_seq,
                "input_size": input_size,
                "output_size": None
            })


    # Continuous Batch Mode (Explores batch_size and users separately)
    for continuous_batch in continuous_batch_values:
        for input_size in input_size_values + output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "input_size": input_size,
                    "output_size": None,
                    "batch_size": batch_size,
                    "users": users
                })
        for output_size in output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "input_size": None,
                    "output_size": output_size,
                    "batch_size": batch_size,
                    "users": users
                })

    return benchmark_combinations

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_local_server", action="store_true", help="Enable a start_local_server feature.")

    parser.add_argument("--single_execution", action="store_true", help="Enable a single_execution feature.")
    parser.add_argument("--multi_execution", action="store_true", help="Enable a multi_execution feature.")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--max_seq", type=int, help="Run the max_seq process (hyperparameter value)")
    group.add_argument("--continuous_batch", type=int, help="Run the continuous_batch process (hyperparameter value)")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input_size", type=int, help="Input token length")
    group.add_argument("--output_size", type=int, default=1, help="Output token length")

    parser.add_argument("--batch_size", type=int, default=1, help="Optional Batch Size AKA max_concurrent (default: 1).")
    parser.add_argument("--users", type=int, default=1, help="Optional number of Users AKA num_prompts (default: 1).")

    args = parser.parse_args()
    print(f"Processing with arguments: {args}")
    return args

def main():
    args = read_args()
    env_vars = os.environ.copy()
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # if args.start_local_server:
        # start vLLM inference server
    vllm_log, vllm_process = start_server(env_vars, log_timestamp)

    # Run combinations of benchmarks
    if args.multi_execution:
        benchmark_combinations = generate_combinations()
        for benchmark_args in benchmark_combinations:
            mass_benchmark_execution(benchmark_args)


    # note: benchmarking script uses capture_traces, which will wait for
    # vLLM health endpoint to return a 200 OK
    # it will wait up to 300s by default.

    if args.single_execution:
        single_benchmark_execution(vars(args), log_timestamp)

    # wait for benchmark script to finish
    print("✅ vllm benchmarks completed!")
    # terminate and wait for graceful shutdown of vLLM server
    vllm_process.terminate()
    vllm_process.wait()
    print("✅ vllm shutdown.")
    vllm_log.close()

if __name__ == "__main__":
    # The above line is necessary for below because some env_vars are needed to import these functions
    # PYTHONPATH doesn't seem to work for imports when loaded like this. Needs to be defined a priori

    main()
