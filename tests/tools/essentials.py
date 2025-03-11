# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
from datetime import datetime
from pathlib import Path
import itertools


# Load environment variables
ENV_FILE = "model_envs/env_benchmarking.env" # TODO: This isn't ideal and the env_vars might need to be fixed more broadly

# Load environment variables from model_envs/env_benchmarking.env
def load_env_variables():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value


def initialize_and_trace_benchmark(it):
    load_env_variables()  # TODO: Required here because the next two imports need env vars set before being run
    from utils.prompt_configs import EnvironmentConfig
    from utils.prompt_client import PromptClient

    env_config = EnvironmentConfig()
    mesh_device = env_config.mesh_device
    # Create output directory
    cache_dir = Path(os.environ.get("CACHE_ROOT", ""))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = (
            cache_dir
            / "vllm_online_benchmark_results"
            / f"results_{timestamp}_{mesh_device}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt_client = PromptClient(env_config)
    # note: there isnt a better way to pass an api key to the vllm benchmarking script
    os.environ["OPENAI_API_KEY"] = prompt_client._get_authorization()
    # fmt: on
    context_lens = [(it["input_len"], it["output_len"])]
    # de-dupe
    context_lens = list(set(context_lens))
    # pre-capture traces required for benchmarking
    prompt_client.capture_traces(context_lens=context_lens)
    # Run benchmarks
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    isl = it["input_len"]
    osl = it["output_len"]
    max_concurrent = it["max_concurrent"]
    num_prompts = it["num_prompts"]
    # Results output prepare
    result_filename = (
            result_dir
            / f"edge_case_benchmark_{run_timestamp}_{mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
    )
    # Begin Benchmark
    vllm_dir = os.environ.get("vllm_dir")
    return env_config, result_filename, vllm_dir

def process_max_seq(hyperparam):
    # Your logic for the max_seq process
    if hyperparam['input_size'] is not None:
        value = hyperparam['input_size']
    else:
        value = hyperparam['output_size']

    it = {"input_len": hyperparam['max_seq']-value, "output_len": value, "max_concurrent": 1, "num_prompts": 1 * 1}
    if hyperparam.get('input_size') is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def generate_it(hyperparam):
    # Your logic for the continuous_batch process
    if hyperparam.get('input_size') is not None:
        value = hyperparam['input_size']
    else:
        value = hyperparam['output_size']

    if hyperparam.get('max_seq') is not None:
        hyperparam['batch_size'] = 1
        hyperparam['users'] = 1
    else:
        hyperparam['max_seq'] = hyperparam['continuous_batch']

    # it = {"input_len": int(hyperparam['continuous_batch'] / hyperparam['batch_size'] - value), "output_len": value,
    #       "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}
    # TODO: Explore the above and if dispersing max context length across batch or users is appropriate for tests

    it = {"input_len": int(hyperparam['max_seq'] - value), "output_len": value,
              "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}

    if hyperparam.get('input_size') is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def generate_benchmarks(batch_size_values, continuous_batch_values, input_size_values, max_seq_values,
                        output_size_values, users_values):
    benchmark_combinations = []
    # Max_seq Mode (Mutually exclusive with batch_size & users)
    # Continuous Batch Mode (Explores batch_size and users separately)
    for continuous_batch in continuous_batch_values:
        for input_size in input_size_values + output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "input_size": None,
                    "output_size": output_size,
                    "batch_size": batch_size,
                    "users": users
                })
        for output_size in output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "input_size": input_size,
                    "output_size": None,
                    "batch_size": batch_size,
                    "users": users
                })
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

    return benchmark_combinations

