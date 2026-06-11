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

# Per-size dim tables (CONFIGS) were retired in #3 Phase D — dims for listed models now
# live in model_matrix.toml (authoritative); novel models derive via from_hf_config above.
