# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for Mistral and Mixtral model families."""

from dataclasses import dataclass
from typing import Optional

try:
    from .config_qwen2_5 import ModelConfig
except ImportError:
    from .config_qwen2_5 import ModelConfig  # script mode


@dataclass
class MoEModelConfig(ModelConfig):
    num_experts: int = 8
    num_experts_per_tok: int = 2


CONFIGS = {
    "7B":      ModelConfig(hidden_size=4096, num_heads=32, num_kv_heads=8,  head_dim=128, intermediate_size=14336),
    "8x7B":    MoEModelConfig(hidden_size=4096, num_heads=32, num_kv_heads=8, head_dim=128, intermediate_size=14336, num_experts=8, num_experts_per_tok=2),
    "8x22B":   MoEModelConfig(hidden_size=6144, num_heads=48, num_kv_heads=8, head_dim=128, intermediate_size=16384, num_experts=8, num_experts_per_tok=2),
}
