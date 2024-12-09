#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import numpy as np
from typing import List, Dict, Tuple
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


def get_test_combinations(
    context_lens: List[Tuple[int, int]],
) -> List[Dict[str, int]]:
    combinations = []
    for input_len, output_len in context_lens:
        # Skip invalid combinations where output_len > input_len
        context = input_len + output_len
        if context <= 4096:
            bsz = 32
        elif context <= 8192:
            bsz = 16
        else:
            bsz = 1

        num_prompts = max(bsz * 4, 4)
        combinations.append(
            {
                "input_len": input_len,
                "output_len": output_len,
                "batch_size": bsz,
                "num_prompts": num_prompts,
            }
        )

    # Log total number of combinations
    logger.info(f"Generated {len(combinations)} valid test combinations")
    for i, combo in enumerate(combinations, 1):
        logger.info(
            f"Combination {i}: input_len={combo['input_len']}, "
            f"output_len={combo['output_len']}, batch_size={combo['batch_size']}"
        )

    return combinations


def run_sequence_length_test(
    combinations: List[Dict[str, int]],
    save_dir: str,
    file_prefix: str,
    num_iterations: int = 1,
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
) -> List[dict]:
    # Create save directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = Path(save_dir) / f"results_{timestamp}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize results storage
    all_results = []

    # Initialize configurations
    env_config = EnvironmentConfig(vllm_model=model)
    prompt_client = PromptClient(env_config)
    prompt_client.capture_traces()

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

        # Process batches
        try:
            responses = batch_processor.process_batch(
                prompts=prompts,
                input_seq_lengths=input_seq_lengths,
                tokenizer=tokenizer,
            )

            # Calculate statistics
            stats = {
                "input_seq_len": input_len,
                "output_seq_len": output_len,
                "batch_size": batch_size,
                "mean_decode_tps": np.mean([r["decode_tps"] for r in responses]),
                "mean_total_tps": np.mean([r["total_tps"] for r in responses]),
                "mean_ttft": np.mean([r["ttft"] for r in responses]),
                "std_decode_tps": np.std([r["decode_tps"] for r in responses]),
                "std_total_tps": np.std([r["total_tps"] for r in responses]),
                "std_ttft": np.std([r["ttft"] for r in responses]),
                "num_prompts": num_prompts,
                "num_iterations": num_iterations,
                "timestamp": timestamp,
                "combination_index": idx,
            }

            all_results.append(stats)

            # Log results
            logger.info(
                f"Results for combination {idx}/{total_combinations}:\n"
                f"Mean Decode TPS: {stats['mean_decode_tps']:.2f} ± "
                f"{stats['std_decode_tps']:.2f}\n"
                f"Mean Total TPS: {stats['mean_total_tps']:.2f} ± "
                f"{stats['std_total_tps']:.2f}\n"
                f"Mean TTFT: {stats['mean_ttft']:.2f} ± {stats['std_ttft']:.2f}"
            )

            # Save results after each combination
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=4)

        except Exception as e:
            logger.error(f"Error processing combination {idx}: {e}")
            continue

    return all_results


if __name__ == "__main__":
    # Define parameter ranges
    typical_context_lens = [
        # (128, 128),
        # (128, 2048),
        # (128, 4096),
        # (2048, 128),
        # (2048, 2048),
        # (1000, 1000),
        # (500, 2000),
        # (5000, 500),
        # (20000, 2000),
    ]
    extra_context_lengths = [
        # (128, 2),
        # (256, 2),
        # (512, 32),
        # (1000, 24),
        # (2000, 32),
        # (4000, 32),
        # (8100, 32),
        (32760, 1024),
    ]
    # Generate all valid combinations upfront
    combinations = get_test_combinations(
        context_lens=typical_context_lens + extra_context_lengths,
    )

    # Run tests
    results = run_sequence_length_test(
        combinations=combinations,
        save_dir="online_benchmarking",
        file_prefix="online_benchmark_results",
    )
