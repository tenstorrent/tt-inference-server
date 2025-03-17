# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List

from workflows.workflow_config import WorkflowVenvType

@dataclass(frozen=True)
class TestsConfig:
    hf_model_repo: str
    max_context_length: int = 16666
    device: str = 'N300'

_tests_config_list = [
    TestsConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        max_context_length=128256-3000
    ),
    TestsConfig(
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        max_context_length=128256-3000
    ),
    TestsConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        max_context_length=128256-3000
    ),
    # TestsConfig(
    #     hf_model_repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     workflow_venv_type=WorkflowVenvType.TESTS,
    #     max_context_length=128256-3000
    # ),
    TestsConfig(
        hf_model_repo="meta-llama/Llama-3.2-3B-Instruct",
        max_context_length=128256-3000
    ),
    TestsConfig(
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        max_context_length=128256-3000
    ),
    TestsConfig(
        hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
        max_context_length=128256-3000
    ),
    TestsConfig(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        max_context_length=128256-3000
    ),
]

DEVICES = ['N150', 'N300', 'T3K', 'TG']

_tests_max_context_list = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        # "N150": "Not Supported",
        # "N300": "Not Supported",
        "T3K": 32064-960,
        "TG": 128256-3000
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        # "N150": "Not Supported",
        # "N300": "Not Supported",
        "T3K": 32064-960,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        # "N150": "Not Supported",
        # "N300": "Not Supported",
        "T3K": 32064-960,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "N150": 8192-250,
        "N300": 128256-3000,
        "T3K": 128256-3000,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "N150": 128256-3000,
        "N300": 128256-3000,
        "T3K": 128256-3000,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        # "N150": "Not Supported",
        # "N300": "Not Supported",
        "T3K": 32064-960,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "N150": 4096-128,
        "N300": 64128-1500,
        "T3K": 128256-3000,
        "TG": 128256-3000
    },
    "meta-llama/Llama-3.2-11B-Instruct": {
        "N150": 4096-128,
        "N300": 64128-1500,
        "T3K": 128256-3000,
        "TG": 128256-3000
    }
}
updated_configs=[]

def init_configs(device, _tests_config_list):
    for config in _tests_config_list:
        # Get the model's context length for the specified device
        model_context_lengths = _tests_max_context_list.get(config.hf_model_repo, None)
        if model_context_lengths is None:
            # Handle the case where the device is not found for the model
            raise ValueError(f"Device '{device}' not found for model '{config.hf_model_repo}'")
        devices = _tests_max_context_list.get(config.hf_model_repo, {})
        if model_context_lengths is None:
            # Handle the case where the model is not found in the mapping
            raise ValueError(f"Model '{config.hf_model_repo}' not found in MAX_CONTEXT_LENGTHS")

        context_length = model_context_lengths.get(device, None)

        # Create a new configuration with the updated context length
        updated_config = TestsConfig(
            hf_model_repo=config.hf_model_repo,
            max_context_length=context_length
        )
        updated_configs.append(updated_config)

    TESTS_CONFIGS = {config.hf_model_repo: config for config in updated_configs}
    return TESTS_CONFIGS

# Replace the original _tests_config_list with the updated configurations

TESTS_CONFIGS = {config.hf_model_repo: config for config in _tests_config_list}



