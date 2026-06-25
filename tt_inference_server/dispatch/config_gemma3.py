# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3 model family config (Google)."""

from dataclasses import dataclass


@dataclass
class Gemma3Config:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    activation: str = "gelu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    @classmethod
    def from_hf_config(cls, hf_config):
        # Multimodal variant (ForConditionalGeneration) nests text params under text_config
        cfg = getattr(hf_config, "text_config", hf_config)
        hidden = cfg.hidden_size
        n_heads = cfg.num_attention_heads
        # Read actual activation (gelu_pytorch_tanh is Gemma's default)
        hf_act = getattr(
            cfg, "hidden_activation", getattr(cfg, "hidden_act", "gelu_pytorch_tanh")
        )
        return cls(
            num_heads=n_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=getattr(cfg, "head_dim", hidden // n_heads),
            hidden_size=hidden,
            intermediate_size=cfg.intermediate_size,
            activation=hf_act,
            norm_eps=getattr(cfg, "rms_norm_eps", getattr(cfg, "layer_norm_eps", 1e-6)),
        )


# Alias so the registry's generic ModelConfig lookup finds this class
ModelConfig = Gemma3Config
