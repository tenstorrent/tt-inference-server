# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging
from typing import Dict


import pytest

from benchmarking.prompt_client_online_benchmark import run_sequence_length_test

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Test params
# see: https://github.com/tenstorrent/tt-metal/tree/main/models/demos/t3000/llama3_70b#details

TEST_paramS = [
    {"input_len": 128, "output_len": 128, "batch_size": 32, "num_prompts": 32},
    {"input_len": 2048, "output_len": 2048, "batch_size": 32, "num_prompts": 32},
    {"input_len": 4000, "output_len": 96, "batch_size": 32, "num_prompts": 32},
    {"input_len": 4096, "output_len": 256, "batch_size": 32, "num_prompts": 32},
    {"input_len": 8000, "output_len": 192, "batch_size": 16, "num_prompts": 16},
    {"input_len": 8192, "output_len": 256, "batch_size": 16, "num_prompts": 16},
    {"input_len": 32768, "output_len": 32, "batch_size": 1, "num_prompts": 1},
    {"input_len": 32768, "output_len": 98304, "batch_size": 1, "num_prompts": 1},
]


@pytest.mark.parametrize("param", TEST_paramS)
def test_sequence_length(param: Dict[str, int]):
    # Run the sequence length test
    results = run_sequence_length_test(
        combinations=[param],  # Pass as single-item list for compatibility
        save_dir="vllm_test_seq_lens",
        file_prefix="vllm_test_seq_lens",
        model="meta-llama/Llama-3.1-70B-Instruct",
    )

    # Add assertions to verify the results
    assert results is not None, "Test results should not be None"

    # Verify the results contain expected data
    logger.info(f"Results: {results}")
    assert isinstance(results, list)
    stats = results[0]
    assert "input_seq_len" in stats
    assert "output_seq_len" in stats

    # Verify the specific param parameters were used
    assert stats["input_seq_len"] == param["input_len"]
    assert stats["output_seq_len"] == param["output_len"]
    assert stats["batch_size"] == param["batch_size"]
    assert stats["num_prompts"] == param["num_prompts"]

    # Add specific assertions for the test parameters
    assert stats["total_output_tokens"] > 0
    assert stats["mean_tpot"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
