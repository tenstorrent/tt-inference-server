# SPDX-License-Identifier: Apache-2.0
"""Thin `qwen3_5` config so vLLM's AutoConfig can parse the Qwen3.6-27B
checkpoint. The TT model reads config.json itself; this only needs to surface
the text-tower fields vLLM uses for scheduling + KV-cache sizing.

`qwen3_5` is absent from public transformers (verified against 4.53.0), so we
register it here at plugin import time (module-level call in __init__.py,
guarded by try/except so the plugin stays importable without vllm installed).

Config-file layout
------------------
The checkpoint's config.json nests all text-tower parameters under
`text_config` (there is no top-level `vocab_size`, `hidden_size`, etc.).
`Qwen3_5Config.__init__` promotes them to the top-level PretrainedConfig
namespace so vLLM's attribute reads resolve correctly.
"""
from transformers import PretrainedConfig


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, **kwargs):
        # Promote text-tower fields from the nested text_config dict so that
        # vLLM's standard attribute reads (num_hidden_layers, hidden_size, …)
        # resolve on the top-level config object.
        text_config = kwargs.get("text_config", {}) or {}
        for key in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "tie_word_embeddings",
        ):
            if key in text_config and key not in kwargs:
                kwargs[key] = text_config[key]

        # Drop the nested sub-config dicts. If `text_config` stayed set,
        # PretrainedConfig.get_text_config() would return that raw dict and
        # vLLM (patch_rope_parameters, etc.) would fail trying to set
        # attributes on a dict. With it removed, get_text_config() returns this
        # top-level config (which carries the promoted text fields). Text-only
        # serving, so vision_config is dropped too (keeps is_multimodal off).
        kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)

        super().__init__(**kwargs)

        # Ensure the conditional-generation arch name is set so that
        # platform.py resolves TTQwen3_5ForConditionalGeneration.
        # Fallback only when the checkpoint omits architectures entirely;
        # a present top-level value (set by super().__init__) is preserved.
        if not getattr(self, "architectures", None):
            self.architectures = ["Qwen3_5ForConditionalGeneration"]


def register_qwen3_5_config():
    """Register Qwen3_5Config with transformers AutoConfig (idempotent)."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "qwen3_5" not in CONFIG_MAPPING:
        AutoConfig.register("qwen3_5", Qwen3_5Config)
