#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
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
    fixed_batch_size: Optional[int] = None,
) -> List[Dict[str, int]]:
    combinations = []
    for input_len, output_len in context_lens:
        # Skip invalid combinations where output_len > input_len
        context = input_len + output_len
        if not fixed_batch_size:
            if context <= 4096:
                bsz = 32
            elif context <= 8192:
                bsz = 16
            else:
                bsz = 1
        else:
            bsz = fixed_batch_size

        num_prompts = max(bsz * 8, 32)
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
            f"output_len={combo['output_len']}, batch_size={combo['batch_size']}, "
            f"num_prompts={combo['num_prompts']}"
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
            e2e_latency = np.max([r["duration"] for r in responses])
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
                "duration": e2e_latency,
                "num_iterations": num_iterations,
                "request_throughput": num_requests / e2e_latency,
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
    # Define benchmarking context length (isl, osl) pairs
    context_lens = [
        (128, 128),
        # (128, 2048),
        # (128, 4096),
        # (2048, 128),
        # (2048, 2048),
        # (1000, 1000),
        # (500, 2000),
        # (5000, 500),
        # (20000, 2000),
        # (128, 2),
        # (256, 2),
        # (512, 32),
        # (1000, 24),
        # (2000, 32),
        # (4000, 32),
        # (8100, 32),
    ]
    # Generate all valid combinations upfront
    combinations = get_test_combinations(context_lens=context_lens, fixed_batch_size=1)

    # Run tests
    results = run_sequence_length_test(
        combinations=combinations,
        save_dir="online_benchmarking",
        file_prefix="online_benchmark_results",
    )
