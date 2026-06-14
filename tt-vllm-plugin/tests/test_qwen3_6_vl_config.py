# SPDX-License-Identifier: Apache-2.0
"""Tests for the multimodal Qwen3_6VLConfig shim.

Unlike the text-only Qwen3_5Config, the VL config must KEEP `vision_config`
so that vLLM treats the model as multimodal (its Qwen3VL processor expands
image/video tokens) and carry a distinct architecture name so the plugin's
platform resolver maps it to the VL generator class.
"""
import json
import os

import pytest

# Local snapshot of the Qwen3.6-27B checkpoint config (nests text params under
# `text_config` and also carries a `vision_config`).
DEFAULT_SNAPSHOT = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json"
)
CONFIG_JSON = os.getenv("QWEN36_CONFIG_JSON", DEFAULT_SNAPSHOT)


def _load_cfg_dict():
    with open(CONFIG_JSON) as f:
        return json.load(f)


@pytest.mark.skipif(
    not os.path.exists(CONFIG_JSON),
    reason="set QWEN36_CONFIG_JSON to a local Qwen3.6-27B config.json",
)
def test_snapshot_has_text_and_vision_config():
    d = _load_cfg_dict()
    assert "text_config" in d
    assert "vision_config" in d


@pytest.mark.skipif(
    not os.path.exists(CONFIG_JSON),
    reason="set QWEN36_CONFIG_JSON to a local Qwen3.6-27B config.json",
)
def test_vl_config_keeps_vision_and_promotes_text():
    from tt_vllm_plugin.qwen3_5_config import Qwen3_5Config, Qwen3_6VLConfig

    cfg_dict = _load_cfg_dict()

    cfg = Qwen3_6VLConfig(**cfg_dict)

    # Vision kept -> vLLM treats the model as multimodal.
    assert getattr(cfg, "vision_config", None) is not None

    # Text fields still promoted to the top level.
    assert cfg.num_hidden_layers is not None
    assert cfg.hidden_size is not None
    assert cfg.num_attention_heads is not None
    assert cfg.num_hidden_layers == 64
    assert cfg.hidden_size == 5120
    assert cfg.num_attention_heads == 24

    # Distinct architecture so the platform resolver picks the VL generator.
    assert cfg.architectures == ["Qwen3_6VLForConditionalGeneration"]

    # Distinct model_type must survive instantiation from the checkpoint dict
    # (which carries model_type="qwen3_5"); otherwise AutoConfig resolution
    # would route the VL config to the text-only class.
    assert cfg.model_type == "qwen3_5_vl"

    # get_text_config() must not return the raw text_config dict.
    text_cfg = cfg.get_text_config()
    assert not isinstance(text_cfg, dict)

    # Regression guard: the text-only config must still DROP vision_config.
    text_only = Qwen3_5Config(**_load_cfg_dict())
    assert getattr(text_only, "vision_config", None) is None


def test_vl_config_registered():
    import tt_vllm_plugin  # noqa: F401
    from tt_vllm_plugin.qwen3_5_config import register_qwen3_6_vl_config
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    register_qwen3_6_vl_config()  # idempotent
    from tt_vllm_plugin.qwen3_5_config import Qwen3_6VLConfig

    assert Qwen3_6VLConfig.model_type in CONFIG_MAPPING
