# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""
This module handles the runtime patching of SGLang to use TT-Metal models.
"""

import logging

from sglang.srt.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def register_tt_models():
    """Register TT-Metal models with SGLang's model registry."""
    logger.info("[TT-Plugin] register_tt_models() called")
    try:
        # Import all TT model classes
        from ..models.tt_llm import (
            TTGptOssForCausalLM,
            TTLlamaForCausalLM,
            TTMistralForCausalLM,
            TTQwenForCausalLM,
        )

        logger.info("[TT-Plugin] Imported TT model classes successfully")

        # Mapping from HuggingFace architecture names to TT model classes
        TT_MODEL_REGISTRY = {
            "LlamaForCausalLM": TTLlamaForCausalLM,  # Llama-3.1-8B, Llama-3.1-70B, etc.
            "Qwen2ForCausalLM": TTQwenForCausalLM,  # Qwen2.5-7B, Qwen2.5-14B, etc.
            "MistralForCausalLM": TTMistralForCausalLM,  # Mistral-7B
            "GptOssForCausalLM": TTGptOssForCausalLM,  # GPT-OSS
        }

        # CRITICAL: Directly patch SGLang's ModelRegistry
        ModelRegistry.models.update(TT_MODEL_REGISTRY)
        logger.info(
            f"[TT-Plugin] ✓ Registered {len(TT_MODEL_REGISTRY)} TT models: {list(TT_MODEL_REGISTRY.keys())}"
        )

    except Exception as e:
        logger.error(
            f"[TT-Plugin] Error registering TT models: {type(e).__name__}: {e}"
        )
        import traceback

        traceback.print_exc()
