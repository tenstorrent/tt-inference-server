# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import argparse

import logging

import time
import subprocess
from datetime import datetime
from pathlib import Path
import itertools
from typing import Dict

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
        for input_size in input_size_values:
            benchmark_combinations.append({
                "max_seq": max_seq,
                "input_size": input_size
            })
        for output_size in output_size_values:
            benchmark_combinations.append({
                "max_seq": max_seq,
                "output_size": output_size
            })

    # Continuous Batch Mode (Explores batch_size and users separately)
    for continuous_batch in continuous_batch_values:
        for input_size in input_size_values + output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "input_size": input_size,
                    "batch_size": batch_size,
                    "users": users
                })
        for output_size in output_size_values:
            for batch_size, users in itertools.product(batch_size_values, users_values):
                benchmark_combinations.append({
                    "continuous_batch": continuous_batch,
                    "output_size": output_size,
                    "batch_size": batch_size,
                    "users": users
                })

    return benchmark_combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Path to Python environment
PYTHON_ENV = "/home/stisi/tt-metal/python_env/bin/python"

# Load environment variables
ENV_FILE = "model_envs/env_benchmarking.env"

# Load environment variables from model_envs/env_benchmarking.env
def load_env_variables():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

def start_server(env_vars, log_timestamp):
    vllm_log_file_path = (
            Path(os.getenv("CACHE_ROOT", ".")) / "logs" / f"start_vllm_{log_timestamp}.log"
    )
    vllm_log = open(vllm_log_file_path, "w")
    print("running vllm server ...")
    vllm_process = subprocess.Popen(
        ["python", "-u", "/home/stisi/tt-inference-server/vllm-tt-metal-llama3/src/run_vllm_api_server.py"],
        stdout=vllm_log,
        stderr=vllm_log,
        env=env_vars,
    )
    return vllm_log, vllm_process

def mass_benchmark_execution(args, env_vars):
    """Runs a single benchmark process with given arguments."""
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args['max_seq'] is not None:
        it = process_max_seq(args)
    elif args['continuous_batch'] is not None:
        it = process_continuous_batch(args)

    benchmark_log_file_path = (
            Path(os.getenv("CACHE_ROOT", "."))
            / "logs"
            / f"run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print("running vllm benchmarks client ...")
    env_config, result_filename, vllm_dir = initialize_and_trace_benchmark(it)

    print(f"Running benchmark with args: {args}")
    assert vllm_dir is not None, "vllm_dir must be set."
    original_run_benchmark(
        benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
        params=it,
        model=env_config.vllm_model,
        port=env_config.service_port,
        result_filename=result_filename,
    )
    logger.info("Single Benchmark completed")
    benchmark_log.close()
    return

def original_run_benchmark(
    params: Dict[str, int],
    model: str,
    port: int,
    benchmark_script: str,
    result_filename: Path,
) -> None:
    """Run a single benchmark with the given parameters."""
    # fmt: off
    cmd = [
        "python", benchmark_script,
        "--backend", "vllm",
        "--model", model,
        "--port", str(port),
        "--dataset-name", "random",
        "--num-prompts", str(params["num_prompts"]),
        "--random-input-len", str(params["input_len"]),
        "--random-output-len", str(params["output_len"]),
        "--ignore-eos",  # Ignore EOS tokens to force max output length as set
        "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
        "--save-result",
        "--result-filename", str(result_filename)
    ]
    # fmt: on

    logger.info(f"Running benchmark with parameters: {params}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Benchmark completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed with error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during benchmark: {e}")

    # Add a small delay between runs to ensure system stability
    time.sleep(2)


def process_max_seq(hyperparam):
    # Your logic for the max_seq process
    value = hyperparam['input_size'] if hyperparam['input_size'] else hyperparam['output_size']
    it = {"input_len": hyperparam['max_seq']-value, "output_len": value, "max_concurrent": 1, "num_prompts": 1 * 1}
    if hyperparam['input_size'] is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def process_continuous_batch(hyperparam):
    # Your logic for the continuous_batch process
    value = hyperparam['input_size'] if hyperparam['input_size'] else hyperparam['output_size']
    # it = {"input_len": int(hyperparam['continuous_batch'] / hyperparam['batch_size'] - value), "output_len": value,
    #       "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}
    it = {"input_len": int(hyperparam['continuous_batch'] - value), "output_len": value,
          "max_concurrent": hyperparam['batch_size'], "num_prompts": hyperparam['users']}

    if hyperparam["input_size"] is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

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

def single_benchmark_execution(args, log_timestamp):
    if args['max_seq'] is not None:
        it = process_max_seq(args)
    elif args['continuous_batch'] is not None:
        it = process_continuous_batch(args)

    benchmark_log_file_path = (
            Path(os.getenv("CACHE_ROOT", "."))
            / "logs"
            / f"run_vllm_benchmark_client_{log_timestamp}.log"
    )
    benchmark_log = open(benchmark_log_file_path, "w")
    print("running vllm benchmarks client ...")
    env_config, result_filename, vllm_dir = initialize_and_trace_benchmark(it)

    assert vllm_dir is not None, "vllm_dir must be set."
    original_run_benchmark(
        benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
        params=it,
        model=env_config.vllm_model,
        port=env_config.service_port,
        result_filename=result_filename,
    )
    logger.info("Benchmark suite completed")


def initialize_and_trace_benchmark(it):
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
    logger.info(f"\nRunning benchmark")
    # Begin Benchmark
    vllm_dir = os.environ.get("vllm_dir")
    return env_config, result_filename, vllm_dir


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
            mass_benchmark_execution(benchmark_args, env_vars)


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
    load_env_variables() # TODO: Move this back to main() after deciding how/if env_vars are loaded before execution.
    # The above line is necessary for below because some env_vars are needed to import these functions
    # PYTHONPATH doesn't seem to work for imports when loaded like this. Needs to be defined a priori
    from utils.prompt_configs import EnvironmentConfig
    from utils.prompt_client import PromptClient
    main()
