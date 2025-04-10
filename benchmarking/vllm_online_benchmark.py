# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
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
        "--max-concurrency", str(params["max_concurrency"]),
        "--num-prompts", str(params["num_prompts"]),
        "--random-input-len", str(params["input_len"]),
        "--random-output-len", str(params["output_len"]),
        "--ignore-eos",  # Ignore EOS tokens to force max output length as set
        "--percentile-metrics", "ttft,tpot,itl,e2el",  # must add e2el in order for it to be logged
        "--save-result",
        "--result-filename", str(result_filename),
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


def main():
    # Configuration
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

    # Get all benchmark combinations using the original function
    # fmt: off
    combinations = [
        # sweeps for batch-1 (max_concurrency=1)
        {"input_len": 128, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 4},
        {"input_len": 128, "output_len": 2048, "max_concurrency": 1, "num_prompts": 32},
        {"input_len": 128, "output_len": 4096, "max_concurrency": 1, "num_prompts": 8},
        {"input_len": 2048, "output_len": 128, "max_concurrency": 1, "num_prompts": 32},
        {"input_len": 2048, "output_len": 2048, "max_concurrency": 1, "num_prompts": 16},
        {"input_len": 3000, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 8},
        {"input_len": 4096, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 4},
        {"input_len": 8192, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 2},
        {"input_len": 16384, "output_len": 128, "max_concurrency": 1, "num_prompts": 32 * 2},
        # sweeps for batch-32 (max_concurrency=32)
        {"input_len": 128, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 16},
        {"input_len": 128, "output_len": 1024, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 2048, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 2048, "output_len": 2048, "max_concurrency": 32, "num_prompts": 32 * 4},
        {"input_len": 3000, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
        {"input_len": 3900, "output_len": 128, "max_concurrency": 32, "num_prompts": 32 * 8},
    ]
    # fmt: on

    context_lens = [(it["input_len"], it["output_len"]) for it in combinations]
    # de-dupe
    context_lens = list(set(context_lens))

    # pre-capture traces required for benchmarking
    prompt_client.capture_traces(context_lens=context_lens, timeout=1200.0)

    # Run benchmarks
    for i, params in enumerate(combinations, 1):
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        isl = params["input_len"]
        osl = params["output_len"]
        max_concurrency = params["max_concurrency"]
        num_prompts = params["num_prompts"]
        result_filename = (
            result_dir
            / f"vllm_online_benchmark_{run_timestamp}_{mesh_device}_isl-{isl}_osl-{osl}_maxcon-{max_concurrency}_n-{num_prompts}.json"
        )
        logger.info(f"\nRunning benchmark {i}/{len(combinations)}")
        vllm_dir = os.environ.get("vllm_dir")
        assert vllm_dir is not None, "vllm_dir must be set."
        run_benchmark(
            benchmark_script=f"{vllm_dir}/benchmarks/benchmark_serving.py",
            params=params,
            model=env_config.vllm_model,
            port=env_config.service_port,
            result_filename=result_filename,
        )

    logger.info("Benchmark suite completed")


if __name__ == "__main__":
    main()
