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


class Qwen3_6VLConfig(Qwen3_5Config):
    """Multimodal (image+video) variant of the Qwen3.6-27B served config.

    Identical to `Qwen3_5Config` for the text tower (it reuses the same
    `_promote_text_fields` promotion logic), but it KEEPS `vision_config` so
    that vLLM treats the model as multimodal: its native Qwen3VL processor
    expands image/video placeholder tokens, and `is_multimodal` stays on.

    A distinct `model_type` ("qwen3_5_vl") registers this under its own key in
    transformers' CONFIG_MAPPING, leaving the text-only `qwen3_5` resolution
    100% untouched. A distinct architecture name lets the plugin's platform
    resolver map this to the VL generator class instead of the text-only one.

    `text_config` is still dropped for the same reason as the text-only config:
    if it stayed set, PretrainedConfig.get_text_config() would return that raw
    dict and vLLM (patch_rope_parameters, etc.) would fail trying to set
    attributes on a dict. With it removed, get_text_config() returns this
    top-level config (which carries the promoted text fields). Note this is a
    Qwen3VL-style "flat" multimodal config: vLLM reads the text params from the
    top level (not via a nested text_config sub-config), and the vision tower
    params from the kept vision_config.
    """

    model_type = "qwen3_5_vl"

    def __init__(self, **kwargs):
        # Promote text-tower fields up to the top level (shared logic).
        _promote_text_fields(kwargs)

        # Drop ONLY the raw text_config dict (so get_text_config() resolves to
        # this top-level config, not a dict). Crucially, do NOT pop
        # vision_config — keeping it is what makes the model multimodal.
        kwargs.pop("text_config", None)

        # Drop the incoming model_type so the class attribute ("qwen3_5_vl")
        # governs. PretrainedConfig.__init__ would otherwise store the
        # checkpoint's literal "qwen3_5" into __dict__, shadowing the class
        # attr and making instances built from the checkpoint mis-report their
        # type (routing to the text-only class on AutoConfig resolution).
        kwargs.pop("model_type", None)

        # Bypass Qwen3_5Config.__init__ (which would pop vision_config); go
        # straight to PretrainedConfig with vision_config intact.
        PretrainedConfig.__init__(self, **kwargs)

        # Distinct VL arch name so platform.py resolves the VL generator class.
        # Override even when the checkpoint carries the text-only arch name
        # (the Qwen3.6-27B checkpoint ships "Qwen3_5ForConditionalGeneration").
        self.architectures = ["Qwen3_6VLForConditionalGeneration"]


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
