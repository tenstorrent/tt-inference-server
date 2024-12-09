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

from benchmarking.online_benchmark_prompt_client import get_test_combinations
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
        "--request-rate", "1",
        "--dataset-name", "random",
        "--num-prompts", str(params["batch_size"]),
        "--random-input-len", str(params["input_len"]),
        "--random-output-len", str(params["output_len"]),
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


def main():
    # Configuration
    env_config = EnvironmentConfig()

    # Create output directory
    cache_dir = Path(os.environ.get("CACHE_ROOT", ""))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = cache_dir / "vllm_online_benchmark_results" / f"results_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    prompt_client = PromptClient(env_config)
    # note: there isnt a better way to pass an api key to the vllm benchmarking script
    os.environ["OPENAI_API_KEY"] = prompt_client._get_authorization()

    # Define benchmarking parameters
    typical_context_lens = [
        (128, 128),
        (128, 2048),
        (128, 4096),
        (2048, 128),
        (2048, 2048),
        (1000, 1000),
        (500, 2000),
        (5000, 500),
        (20000, 2000),
    ]
    extra_context_lengths = [
        (128, 2),
        (256, 2),
        (512, 32),
        (1000, 24),
        (2000, 32),
        (4000, 32),
        (8100, 32),
        # (32000, 1024)
    ]

    # Get all benchmark combinations using the original function
    combinations = get_test_combinations(
        context_lens=typical_context_lens + extra_context_lengths,
    )

    # ensure vllm server is ready
    prompt_client.capture_traces()

    # Run benchmarks
    for i, params in enumerate(combinations, 1):
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        isl = params["input_len"]
        osl = params["output_len"]
        bsz = params["batch_size"]
        num_prompts = params["num_prompts"]
        result_filename = (
            result_dir
            / f"vllm_online_benchmark_isl-{isl}_osl-{osl}_bsz-{bsz}_n-{num_prompts}_{run_timestamp}.json"
        )
        logger.info(f"\nRunning benchmark {i}/{len(combinations)}")
        run_benchmark(
            benchmark_script="/home/user/vllm/benchmarks/benchmark_serving.py",
            params=params,
            model=env_config.vllm_model,
            port=env_config.service_port,
            result_filename=result_filename,
        )

    logger.info("Benchmark suite completed")


if __name__ == "__main__":
    main()
