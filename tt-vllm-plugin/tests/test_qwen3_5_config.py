# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from transformers import AutoConfig


CKPT = os.getenv("QWEN36_CKPT_DIR", "")


def test_qwen3_5_config_registered():
    # Importing the plugin registers the config at import time (module-level
    # call in __init__.py, guarded by try/except so it's safe without vllm).
    import tt_vllm_plugin  # noqa: F401
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "qwen3_5" in CONFIG_MAPPING


@pytest.mark.skipif(not CKPT, reason="set QWEN36_CKPT_DIR to a local Qwen3.6-27B dir")
def test_autoconfig_parses_checkpoint():
    import tt_vllm_plugin  # noqa: F401

    cfg = AutoConfig.from_pretrained(CKPT)
    assert cfg.num_hidden_layers == 64
    assert cfg.hidden_size == 5120
    assert cfg.num_attention_heads == 24
    # vocab_size lives under text_config in the checkpoint; the config class
    # promotes it to the top-level object.
    assert cfg.vocab_size == 248320
    assert cfg.max_position_embeddings == 262144
    assert cfg.architectures == ["Qwen3_5ForConditionalGeneration"]
