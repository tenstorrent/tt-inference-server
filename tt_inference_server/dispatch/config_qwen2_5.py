# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shape constants for the Qwen 2.5 model family."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    norm_type: str = "rmsnorm"
    activation: str = "silu"
    norm_eps: float = 1e-6

    @classmethod
    def from_hf_config(cls, hf_config) -> "ModelConfig":
        hidden = hf_config.hidden_size
        n_heads = hf_config.num_attention_heads
        n_kv = getattr(hf_config, "num_key_value_heads", n_heads)
        head_dim = getattr(hf_config, "head_dim", None) or (hidden // n_heads)
        intermediate = hf_config.intermediate_size
        return cls(
            hidden_size=hidden,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=intermediate,
        )


# Qwen 2.5 known variants
CONFIGS = {
    "0.5B": ModelConfig(hidden_size=896,  num_heads=14, num_kv_heads=2,  head_dim=64,  intermediate_size=4864),
    "1.5B": ModelConfig(hidden_size=1536, num_heads=12, num_kv_heads=2,  head_dim=128, intermediate_size=8960),
    "3B":   ModelConfig(hidden_size=2048, num_heads=16, num_kv_heads=2,  head_dim=128, intermediate_size=11008),
    "7B":   ModelConfig(hidden_size=3584, num_heads=28, num_kv_heads=4,  head_dim=128, intermediate_size=18944),
    "14B":  ModelConfig(hidden_size=5120, num_heads=40, num_kv_heads=8,  head_dim=128, intermediate_size=13824),
    "32B":  ModelConfig(hidden_size=5120, num_heads=64, num_kv_heads=8,  head_dim=80,  intermediate_size=27648),
    "72B":  ModelConfig(hidden_size=8192, num_heads=64, num_kv_heads=8,  head_dim=128, intermediate_size=29568),
}
