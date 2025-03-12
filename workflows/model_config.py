# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import re
from pathlib import Path
from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Set


class DeviceTypes(IntEnum):
    CPU = auto()
    E150 = auto()
    N150 = auto()
    N300 = auto()
    T3K = auto()
    GALAXY = auto()


@dataclass(frozen=True)
class ModelConfig:
    """
    All static configuration and metadata required to execute workflows for a given model.
    """

    device_configurations: Set[DeviceTypes]
    hf_model_repo: str = None
    model_name: str = None  # uses defaults based on hf_model_repo
    model_id: str = None  # uses defaults based on hf_model_repo
    impl_id: str = "tt-metal"  # implementation ID
    version: str = "0.0.1"
    param_count: int = None
    min_disk_gb: int = None
    min_ram_gb: int = None
    repacked: int = 0
    docker_image: str = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        # Note: ONLY run this in __post_init__
        # need to use __setattr__ because instance is frozen
        if not self.model_name:
            # use basename of HF model ID to use same format as tt-transformers
            object.__setattr__(self, "model_name", Path(self.hf_model_repo).name)
        if not self.model_id:
            object.__setattr__(self, "model_id", self.get_default_model_id())

        # use param count to detemine conservative disk and ram minimums
        # these are only checked during initial model setup
        if not self.param_count:
            object.__setattr__(
                self, "param_count", ModelConfig.infer_param_count(self.hf_model_repo)
            )
        if not self.min_disk_gb:
            if self.repacked:
                # 2x for raw fp16 weights hf cache (may already be present)
                # 1x for repacked quantized copy
                # 1x for tt-metal cache
                # 1x for overhead
                object.__setattr__(self, "min_disk_gb", self.param_count * 5)
            else:
                # 2x for raw fp16 weights hf cache (may already be present
                # 2x for copy
                object.__setattr__(self, "min_disk_gb", self.param_count * 4)
        if not self.min_ram_gb:
            object.__setattr__(self, "min_ram_gb", self.param_count * 5)

    def validate_data(self):
        assert (
            self.hf_model_repo or self.model_name
        ), "either hf_model_repo or model_name must be set."

    def get_default_model_id(self):
        return f"id_{self.impl_id}-{self.model_name}-v{self.version}"

    @staticmethod
    def infer_param_count(hf_model_repo: str) -> int:
        """
        Infers the parameter count (in billions) from the hf_model_repo string.

        Examples:
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" -> 70
            "Qwen/Qwen2.5-72B" -> 72
            "meta-llama/Llama-3.1-8B" -> 8

        Returns:
            The inferred parameter count as an int, or None if no pattern is found.
        """
        matches = re.findall(r"(\d+(?:\.\d+)?)B", hf_model_repo)
        if matches:
            # Take the last match which is typically the parameter count
            count_str = matches[-1]
            try:
                # Convert to float first (to handle potential decimals) and then to int.
                count_float = float(count_str)
                return int(count_float)
            except ValueError:
                return None
        return None


config_list = [
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="Qwen/Qwen2.5-72B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="Qwen/Qwen2.5-7B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="Qwen/Qwen2.5-7B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.3-70B",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-11B-Vision",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-1B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-3B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.2-3B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.1-70B",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.1-8B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3-70B",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3-70B-Instruct",
        repacked=1,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3-8B",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
    ModelConfig(
        device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
        hf_model_repo="meta-llama/Llama-3-8B-Instruct",
        repacked=0,
        docker_image="ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.4-v0.56.0-rc39-3429acf14e46",
    ),
]

# Generate a dictionary keyed by the model_name for each ModelConfig instance
MODEL_CONFIGS = {config.model_name: config for config in config_list}
