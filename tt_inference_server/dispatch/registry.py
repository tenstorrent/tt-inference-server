# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model family registry — auto-detect config from HuggingFace model config."""

from __future__ import annotations

from . import (
    config_gemma3 as gemma3,
    config_llama3 as llama3,
    config_mistral as mistral,
    config_qwen2_5 as qwen2_5,
    config_qwen3 as qwen3,
)
from .config_qwen2_5 import ModelConfig
from .config_mistral import MoEModelConfig

# Maps HuggingFace architecture name → config module
_FAMILY_MAP = {
    "Qwen2ForCausalLM": qwen2_5,
    "Qwen3ForCausalLM": qwen3,
    "LlamaForCausalLM": llama3,
    "MistralForCausalLM": mistral,
    "MixtralForCausalLM": mistral,
    "Gemma3ForCausalLM": gemma3,
    "Gemma3ForConditionalGeneration": gemma3,
    # Gemma 1 and 2 use same 4-norm pattern — reuse gemma3 config
    "GemmaForCausalLM": gemma3,
    "Gemma2ForCausalLM": gemma3,
}


def detect_model_family(hf_config) -> ModelConfig:
    """Detect model family and return a ModelConfig from a HuggingFace config.

    Falls back to generic derivation from hf_config fields for unknown
    architectures, so models we haven't seen before still get reasonable
    defaults rather than failing outright.
    """
    archs = getattr(hf_config, "architectures", None) or []
    arch = archs[0] if archs else ""

    if arch in _FAMILY_MAP:
        mod = _FAMILY_MAP[arch]
        # Each config module exposes either ModelConfig or a family-named class
        cfg_class = getattr(
            mod,
            "ModelConfig",
            getattr(mod, arch.replace("ForCausalLM", "Config"), None),
        )
        if cfg_class is not None:
            return cfg_class.from_hf_config(hf_config)
        return _from_hf_config_generic(hf_config)

    # Unknown architecture: derive directly from standard HF config fields.
    return _from_hf_config_generic(hf_config)


def _from_hf_config_generic(hf_config) -> ModelConfig:
    """Derive ModelConfig from standard HuggingFace config fields.

    Reads the subset of fields that are present across nearly all transformer
    configs. Missing fields fall back to safe defaults.
    """
    hidden = getattr(hf_config, "hidden_size", 4096)
    n_heads = getattr(hf_config, "num_attention_heads", 32)
    n_kv = getattr(hf_config, "num_key_value_heads", n_heads)
    head_dim = getattr(hf_config, "head_dim", hidden // n_heads)
    intermediate = getattr(hf_config, "intermediate_size", hidden * 4)

    # Determine norm type from config fields. LayerNorm models carry a layer-norm
    # epsilon under one of several field names (Falcon uses `layer_norm_epsilon`)
    # and no `rms_norm_eps`.
    norm_type = "rmsnorm"
    has_ln_eps = hasattr(hf_config, "layer_norm_eps") or hasattr(
        hf_config, "layer_norm_epsilon"
    )
    if has_ln_eps and not hasattr(hf_config, "rms_norm_eps"):
        norm_type = "layernorm"

    # Determine activation. "gelu" is exact-erf (GPTNeoX); the *_new/_fast/_pytorch_tanh
    # variants are the tanh approximation (distinguished from "gelu" so the runner can
    # pick the matching ttnn.gelu mode).
    act_map = {
        "silu": "silu",
        "swish": "silu",
        "gelu": "gelu",
        "gelu_tanh": "gelu_tanh",
        "gelu_new": "gelu_tanh",
        "gelu_fast": "gelu_tanh",
        "gelu_pytorch_tanh": "gelu_tanh",
        "relu": "relu2",
    }
    model_type = getattr(getattr(hf_config, "text_config", hf_config), "model_type", "")
    hf_act = getattr(hf_config, "hidden_act", None)
    if hf_act is None:
        # Some arches carry no hidden_act and hardcode their MLP activation:
        #   BLOOM  -> tanh-approx GELU (BloomGelu)
        #   Falcon -> exact-erf GELU (nn.GELU in FalconMLP)
        hf_act = {"bloom": "gelu_tanh", "falcon": "gelu"}.get(model_type, "silu")
    activation = act_map.get(hf_act, "silu")

    eps = getattr(
        hf_config,
        "rms_norm_eps",
        getattr(
            hf_config, "layer_norm_eps", getattr(hf_config, "layer_norm_epsilon", 1e-6)
        ),
    )

    # MoE fields
    num_experts = getattr(
        hf_config, "num_local_experts", getattr(hf_config, "num_experts", 0)
    )
    top_k = getattr(
        hf_config, "num_experts_per_tok", getattr(hf_config, "top_k_experts", 2)
    )

    if num_experts > 0:
        return MoEModelConfig(
            hidden_size=hidden,
            num_heads=n_heads,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            intermediate_size=intermediate,
            norm_type=norm_type,
            activation=activation,
            norm_eps=eps,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
        )

    return ModelConfig(
        hidden_size=hidden,
        num_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        intermediate_size=intermediate,
        norm_type=norm_type,
        activation=activation,
        norm_eps=eps,
    )
