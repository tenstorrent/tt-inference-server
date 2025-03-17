# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class TestsConfig:
    """Configuration for test setups."""
    hf_model_repo: str
    max_context_length: int


def init_test_configs(device: str, config_list: List[TestsConfig]) -> Dict[str, TestsConfig]:
    """
    Initialize and update test configurations based on the specified device.

    Args:
        device (str): The device to configure (e.g., 'N150', 'N300').
        config_list (List[TestsConfig]): List of test configurations.

    Returns:
        Dict[str, TestsConfig]: Dictionary of updated test configurations.
    """
    _tests_max_context = {
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
            "T3K": 32064 - 960,
            "TG": 128256 - 3000
        },
        "Qwen/Qwen2.5-72B-Instruct": {
            "T3K": 32064 - 960,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.3-70B-Instruct": {
            "T3K": 32064 - 960,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.2-3B-Instruct": {
            "N150": 8192 - 250,
            "N300": 128256 - 3000,
            "T3K": 128256 - 3000,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.2-1B-Instruct": {
            "N150": 128256 - 3000,
            "N300": 128256 - 3000,
            "T3K": 128256 - 3000,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.1-70B-Instruct": {
            "T3K": 32064 - 960,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "N150": 4096 - 128,
            "N300": 64128 - 1500,
            "T3K": 128256 - 3000,
            "TG": 128256 - 3000
        },
        "meta-llama/Llama-3.2-11B-Instruct": {
            "N150": 4096 - 128,
            "N300": 64128 - 1500,
            "T3K": 128256 - 3000,
            "TG": 128256 - 3000
        }
    }

    updated_configs = []
    for config in config_list:
        model = config.hf_model_repo
        max_context = _tests_max_context.get(model, {}).get(device, None)
        updated_config = TestsConfig(hf_model_repo=model, max_context_length=max_context)
        updated_configs.append(updated_config)

    return {config.hf_model_repo: config for config in updated_configs}


# Example usage
test_config_list = [
    TestsConfig(hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="Qwen/Qwen2.5-72B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llama/Llama-3.3-70B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llama/Llama-3.2-3B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llama/Llama-3.2-1B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llinding/Llama-3.1-70B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llama/Llama-3.1-8B-Instruct", max_context_length=128256 - 3000),
    TestsConfig(hf_model_repo="meta-llama/Llama-3.2-11B-Instruct", max_context_length=128256 - 3000)
]

