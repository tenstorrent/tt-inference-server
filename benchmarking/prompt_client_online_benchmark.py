#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import logging
import numpy as np
from typing import List, Dict
import json
from datetime import datetime
from pathlib import Path

from utils.prompt_configs import PromptConfig, BatchConfig, EnvironmentConfig
from utils.prompt_client import PromptClient
from utils.batch_processor import BatchProcessor
from utils.prompt_generation import generate_prompts
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_sequence_length_test(
    model: str,
    combinations: List[Dict[str, int]],
    result_dir: str,
    file_prefix: str,
    num_iterations: int = 1,
) -> List[dict]:
    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(result_dir) / f"results_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    all_results = []

    # Initialize configurations
    env_config = EnvironmentConfig(vllm_model=model)
    prompt_client = PromptClient(env_config)

    # Test all combinations
    total_combinations = len(combinations)
    for idx, params in enumerate(combinations, 1):
        input_len = params["input_len"]
        output_len = params["output_len"]
        batch_size = params["batch_size"]
        num_prompts = params["num_prompts"]
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = (
            save_path
            / f"{file_prefix}_{run_timestamp}_isl-{input_len}_osl-{output_len}_bsz-{batch_size}_n-{num_prompts}.json"
        )

        logger.info(
            f"\nTesting combination {idx}/{total_combinations}:\n"
            f"input_len={input_len}, output_len={output_len}, "
            f"batch_size={batch_size}, num_prompts={num_prompts}"
        )

        # Configure prompt generation
        prompt_config = PromptConfig(
            input_seq_len=input_len,
            max_prompt_length=input_len,
            num_prompts=num_prompts,
            distribution="fixed",
            dataset="random",
            tokenizer_model=model,
            template=None,
            save_path=None,
            print_prompts=False,
        )

        # Generate prompts
        prompts, input_seq_lengths = generate_prompts(prompt_config)

        # Configure batch processing
        output_seq_lens = [output_len] * num_prompts
        batch_config = BatchConfig(
            batch_size=batch_size,
            output_seq_lens=output_seq_lens,
            num_full_iterations=num_iterations,
            vary_batch_size=False,
            inter_batch_delay=0,
            stream=True,
        )

        # Initialize processor and tokenizer
        batch_processor = BatchProcessor(prompt_client, batch_config)
        tokenizer = AutoTokenizer.from_pretrained(model)

        # pre-capture traces so benchmark does not include 1st run trace capture time
        prompt_client.capture_traces(context_lens=[(input_len, output_len)])
        # Process batches
        try:
            responses = batch_processor.process_batch(
                prompts=prompts,
                input_seq_lengths=input_seq_lengths,
                tokenizer=tokenizer,
            )
            mean_e2el_ms = np.mean([r["latency"] for r in responses]) * 1000.0
            num_requests = num_prompts * num_iterations
            stats = {
                "model_id": model,
                "backend": "vllm",
                "timestamp": timestamp,
                "input_sequence_length": input_len,
                "output_sequence_length": output_len,
                "batch_size": batch_size,
                "num_requests": num_requests,
                "mean_tpot_ms": np.mean([r["tpot_ms"] for r in responses]),
                "std_tpot_ms": np.std([r["tpot_ms"] for r in responses]),
                "mean_ttft_ms": np.mean([r["ttft_ms"] for r in responses]),
                "std_ttft_ms": np.std([r["ttft_ms"] for r in responses]),
                "total_input_tokens": sum([r["input_seq_len"] for r in responses]),
                "total_output_tokens": sum([r["output_seq_len"] for r in responses]),
                "mean_e2el_ms": mean_e2el_ms,
                "num_iterations": num_iterations,
                "request_throughput": num_requests / mean_e2el_ms,
            }

            all_results.append(stats)

            # Log results
            logger.info(
                f"Results for combination {idx}/{total_combinations}:\n"
                f"Mean TTFT: {stats['mean_ttft_ms']:.4f} ± {stats['std_ttft_ms']:.4f}"
                f"Mean TPOT: {stats['mean_tpot_ms']:.4f} ± "
                f"{stats['std_tpot_ms']:.4f}\n"
            )

            # Save results after each combination
            with open(results_file, "w") as f:
                json.dump(stats, f, indent=4)

        except Exception as e:
            logger.error(f"Error processing combination {idx}: {e}")
            continue

    return all_results


if __name__ == "__main__":
    # fmt: off
    combinations = [
        # sweeps for batch-1
        {"input_len": 128, "output_len": 10, "batch_size": 1, "num_prompts": 64},
        {"input_len": 128, "output_len": 128, "batch_size": 1, "num_prompts": 64},
        {"input_len": 128, "output_len": 1024, "batch_size": 1, "num_prompts": 16},
        {"input_len": 128, "output_len": 2048, "batch_size": 1, "num_prompts": 8},
        {"input_len": 128, "output_len": 4096, "batch_size": 1, "num_prompts": 8},
        {"input_len": 2048, "output_len": 128, "batch_size": 1, "num_prompts": 32},
        {"input_len": 2048, "output_len": 2048, "batch_size": 1, "num_prompts": 8},
        # sweeps for batch-32
        {"input_len": 128, "output_len": 10, "batch_size": 32, "num_prompts": 32 * 16},
        {"input_len": 128, "output_len": 128, "batch_size": 32, "num_prompts": 32 * 16},
        {"input_len": 128, "output_len": 1024, "batch_size": 32, "num_prompts": 32 * 8},
        {"input_len": 128, "output_len": 2048, "batch_size": 32, "num_prompts": 32 * 4},
        {"input_len": 128, "output_len": 4096, "batch_size": 32, "num_prompts": 32 * 4},
        {"input_len": 2048, "output_len": 128, "batch_size": 32, "num_prompts": 32 * 8},
        {"input_len": 2048, "output_len": 2048, "batch_size": 32, "num_prompts": 32 * 4},
    ]
    # fmt: on

    # Create output directory
    cache_dir = Path(os.environ.get("CACHE_ROOT", ""))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = cache_dir / "online_benchmark_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    default_model = "meta-llama/Llama-3.1-70B-Instruct"
    model = os.environ.get("HF_MODEL_REPO_ID", default_model)
    results = run_sequence_length_test(
        model=model,
        combinations=combinations,
        result_dir=result_dir,
        file_prefix="online_benchmark",
    )
