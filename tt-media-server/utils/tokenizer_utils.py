# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from config.constants import SupportedModels
from transformers import AutoTokenizer, PreTrainedTokenizer


def resolve_model(model: str | None) -> SupportedModels:
    """
    Resolve a model string to SupportedModels enum.

    Accepts HuggingFace path (e.g., "stabilityai/stable-diffusion-xl-base-1.0").

    Raises:
        ValueError: If model is not supported
    """
    if model is None:
        raise ValueError("Model is required")

    for supported_model in SupportedModels:
        if supported_model.value == model:
            return supported_model

    raise ValueError(
        f"Unsupported model: {model}. Supported models: {[m.value for m in SupportedModels]}"
    )


def get_tokenizer(model: str | SupportedModels | None) -> PreTrainedTokenizer:
    """
    Resolve model and get tokenizer.

    Args:
        model: SupportedModels enum or HuggingFace path string

    Returns:
        PreTrainedTokenizer instance

    Raises:
        ValueError: If model is not supported
    """
    if isinstance(model, SupportedModels):
        return AutoTokenizer.from_pretrained(model.value)

    resolved = resolve_model(model)
    return AutoTokenizer.from_pretrained(resolved.value)
