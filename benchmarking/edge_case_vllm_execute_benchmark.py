# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import argparse

import subprocess
import time
import logging
from datetime import datetime
from typing import Dict
from pathlib import Path

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_benchmark(
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
    value = hyperparam.input_size if hyperparam.input_size else hyperparam.output_size
    it = {"input_len": hyperparam.max_seq-value, "output_len": value, "max_concurrent": 1, "num_prompts": 1 * 1}
    if hyperparam.input_size is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def process_continuous_batch(hyperparam):
    # Your logic for the continuous_batch process
    value = hyperparam.input_size if hyperparam.input_size else hyperparam.output_size
    # it = {"input_len": int(hyperparam.continuous_batch / hyperparam.batch_size - value), "output_len": value,
    #       "max_concurrent": hyperparam.batch_size, "num_prompts": hyperparam.users}
    it = {"input_len": int(hyperparam.continuous_batch - value), "output_len": value,
          "max_concurrent": hyperparam.batch_size, "num_prompts": hyperparam.users}

    if hyperparam.input_size is not None:
        it["input_len"], it["output_len"] = it["output_len"], it["input_len"]
    return it

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--max_seq", type=int, help="Run the max_seq process (hyperparameter value)")
    group.add_argument("--continuous_batch", type=int, help="Run the continuous_batch process (hyperparameter value)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_size", type=int, help="Input token length")
    group.add_argument("--output_size", type=int, default=1, help="Output token length")

    parser.add_argument("--batch_size", type=int, default=1, help="Optional Batch Size AKA max_concurrent (default: 1).")
    parser.add_argument("--users", type=int, default=1, help="Optional number of Users AKA num_prompts (default: 1).")

    args = parser.parse_args()
    print(f"Processing with arguments: {args}")

    # Prepare Prompt to be sent for benchmarking

    if args.max_seq is not None:
        it = process_max_seq(args)
    elif args.continuous_batch is not None:
        it = process_continuous_batch(args)

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


    #Results output prepare
    result_filename = (
            result_dir
            / f"3_edge_case_benchmark_{run_timestamp}_{mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrent}_n-{num_prompts}.json"
    )
    logger.info(f"\nRunning benchmark")

    #Begin Benchmark
    vllm_dir = os.environ.get("vllm_dir")
    assert vllm_dir is not None, "vllm_dir must be set."
    run_benchmark(
        benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
        params=it,
        model=env_config.vllm_model,
        port=env_config.service_port,
        result_filename=result_filename,
    )

    logger.info("Benchmark suite completed")



if __name__ == "__main__":
    main()
