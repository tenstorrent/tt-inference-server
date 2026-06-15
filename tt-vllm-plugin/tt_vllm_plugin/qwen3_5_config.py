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


# Text-tower fields promoted from the nested `text_config` dict to the
# top-level config namespace so that vLLM's standard attribute reads
# (num_hidden_layers, hidden_size, …) resolve on the config object.
_PROMOTED_TEXT_FIELDS = (
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
)


def _promote_text_fields(kwargs):
    """Copy the text-tower fields from kwargs['text_config'] up to the top
    level of kwargs (in place), unless already present at the top level."""
    text_config = kwargs.get("text_config", {}) or {}
    for key in _PROMOTED_TEXT_FIELDS:
        if key in text_config and key not in kwargs:
            kwargs[key] = text_config[key]
    return kwargs


class Qwen3_5Config(PretrainedConfig):
    model_type = "qwen3_5"

    def __init__(self, **kwargs):
        # Promote text-tower fields from the nested text_config dict so that
        # vLLM's standard attribute reads (num_hidden_layers, hidden_size, …)
        # resolve on the top-level config object.
        _promote_text_fields(kwargs)

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


def _make_qwen3_6_vl_config_class():
    """Build `Qwen3_6VLConfig` as a SUBCLASS of the real transformers
    `Qwen3VLConfig` so it passes vLLM's `ctx.get_hf_config(Qwen3VLConfig)`
    isinstance check (the native Qwen3VL processor asserts this). Done in a
    factory so the module stays importable when transformers lacks qwen3_vl
    (off-device tooling) — in that case we fall back to a flat PretrainedConfig.

    The Qwen3.6-27B vision tower IS Qwen3VL-architecture, and the text tower is
    Qwen3-family, so the stock Qwen3VLConfig sub-config parsing (text_config ->
    Qwen3VLTextConfig, vision_config -> Qwen3VLVisionConfig) lines up. We keep
    BOTH sub-configs (do not pop) and set a distinct model_type + arch name.
    """
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as _BaseVL
    except Exception:
        _BaseVL = PretrainedConfig

    class Qwen3_6VLConfig(_BaseVL):
        """Multimodal (image+video) Qwen3.6-27B served config.

        Subclasses transformers' Qwen3VLConfig so vLLM's native Qwen3VL
        processor (which does ``isinstance(config, Qwen3VLConfig)``) accepts it,
        while a distinct ``model_type`` ("qwen3_5_vl") keeps the text-only
        ``qwen3_5`` AutoConfig resolution 100% untouched and a distinct
        architecture name routes the plugin to the VL generator class.
        """

        model_type = "qwen3_5_vl"

        def __init__(self, **kwargs):
            # Also promote the text-tower fields to the TOP level so any flat
            # attribute reads (kept for safety) resolve; Qwen3VLConfig still
            # parses text_config/vision_config into proper sub-config objects.
            _promote_text_fields(kwargs)

            # Drop the checkpoint's literal model_type ("qwen3_5") so the class
            # attribute ("qwen3_5_vl") governs AutoConfig round-trips.
            kwargs.pop("model_type", None)

            if _BaseVL is PretrainedConfig:
                # transformers without qwen3_vl: keep a flat config with
                # vision_config wrapped to an attribute-accessible object.
                from types import SimpleNamespace

                vc = kwargs.get("vision_config")
                if isinstance(vc, dict):
                    kwargs["vision_config"] = SimpleNamespace(**vc)
                kwargs.pop("text_config", None)
                PretrainedConfig.__init__(self, **kwargs)
            else:
                # Real Qwen3VLConfig: it parses text_config + vision_config
                # dicts into Qwen3VLTextConfig / Qwen3VLVisionConfig objects.
                _BaseVL.__init__(self, **kwargs)

            # Distinct VL arch name so platform.py resolves the VL generator.
            self.architectures = ["Qwen3_6VLForConditionalGeneration"]

    return Qwen3_6VLConfig


Qwen3_6VLConfig = _make_qwen3_6_vl_config_class()


def register_qwen3_5_config():
    """Register Qwen3_5Config with transformers AutoConfig (idempotent)."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "qwen3_5" not in CONFIG_MAPPING:
        AutoConfig.register("qwen3_5", Qwen3_5Config)


def register_qwen3_6_vl_config():
    """Register Qwen3_6VLConfig with transformers AutoConfig (idempotent)."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if Qwen3_6VLConfig.model_type not in CONFIG_MAPPING:
        AutoConfig.register(Qwen3_6VLConfig.model_type, Qwen3_6VLConfig)
