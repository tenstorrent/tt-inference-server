# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Client-side vision token calculator utilities.

This module provides functions to calculate the number of vision tokens
that images will contribute to multimodal model inputs. These calculations
are used client-side to properly account for image tokens when determining
max_concurrency and context length limits.
"""

import math
from typing import Callable, Dict, Optional


def calculate_gemma_vision_tokens(image_height: int, image_width: int) -> int:
    """
    Calculate vision tokens for Gemma/PaliGemma models.
    
    Based on empirical observations from vLLM with SigLIP vision encoder.
    Images are resized to max 1120x1120 while maintaining aspect ratio.
    
    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        
    Returns:
        Number of vision tokens
        
    Example:
        3500x2500 image -> 268 tokens (empirically verified from vLLM logs)
    """
    MAX_DIMENSION = 1120
    
    # Resize logic: maintain aspect ratio, fit within MAX_DIMENSION
    scale = min(MAX_DIMENSION / image_width, MAX_DIMENSION / image_height, 1.0)
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    
    # Empirical pixel-to-token ratio from vLLM observations
    # 3500x2500 -> resized to 800x1120 (896,000 pixels) -> 268 tokens
    PIXELS_PER_TOKEN = 3343
    
    total_pixels = resized_width * resized_height
    tokens = total_pixels // PIXELS_PER_TOKEN
    
    return max(tokens, 1)


def calculate_qwen_vision_tokens(image_height: int, image_width: int) -> int:
    """
    Calculate exact vision tokens for Qwen2.5-VL models.
    
    Based on ViT with dynamic resolution support and 14x14 patches.
    Images are processed with dynamic resolution within pixel limits.
    
    Args:
        image_height: Image height in pixels
        image_width: Image width in pixels
        
    Returns:
        Exact number of vision tokens
    """
    PATCH_SIZE = 14
    MIN_PIXELS = 256 * 28 * 28  # 200704
    MAX_PIXELS = 1280 * 28 * 28  # 1003520
    
    total_pixels = image_height * image_width
    
    # Scale image to fit within model's pixel limits
    if total_pixels < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / total_pixels)
        image_height = int(image_height * scale)
        image_width = int(image_width * scale)
    elif total_pixels > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / total_pixels)
        image_height = int(image_height * scale)
        image_width = int(image_width * scale)
    
    # Calculate number of patches
    num_patches_h = math.ceil(image_height / PATCH_SIZE)
    num_patches_w = math.ceil(image_width / PATCH_SIZE)
    
    return num_patches_h * num_patches_w


# Model ID to vision token calculator mapping
# Keys are model repository IDs from HuggingFace
VISION_TOKEN_CALCULATORS: Dict[str, Callable[[int, int], int]] = {
    # Gemma models
    "google/gemma-3-27b-it": calculate_gemma_vision_tokens,
    "google/paligemma-3b-mix-448": calculate_gemma_vision_tokens,
    
    # Qwen models
    "Qwen/Qwen2-VL-2B-Instruct": calculate_qwen_vision_tokens,
    "Qwen/Qwen2-VL-7B-Instruct": calculate_qwen_vision_tokens,
    "Qwen/Qwen2.5-VL-7B-Instruct": calculate_qwen_vision_tokens,
    "Qwen/Qwen2.5-VL-72B-Instruct": calculate_qwen_vision_tokens,
}


def get_vision_token_calculator(model_id: str) -> Optional[Callable[[int, int], int]]:
    """
    Get the vision token calculator for a given model ID.
    
    Args:
        model_id: HuggingFace model repository ID
        
    Returns:
        Vision token calculator function, or None if model is not a VL model
        or not supported
    """
    return VISION_TOKEN_CALCULATORS.get(model_id)


def calculate_image_tokens(
    model_id: str,
    image_height: int,
    image_width: int,
    images_per_prompt: int = 1
) -> int:
    """
    Calculate total image tokens for a given model and image configuration.
    
    Args:
        model_id: HuggingFace model repository ID
        image_height: Image height in pixels
        image_width: Image width in pixels
        images_per_prompt: Number of images per prompt (default: 1)
        
    Returns:
        Total vision tokens (tokens_per_image * images_per_prompt)
        Returns 0 if model is not a VL model or not supported
        
    Example:
        For Gemma-3-27b-it with 3500x2500 image:
        calculate_image_tokens("google/gemma-3-27b-it", 3500, 2500, 1) -> 268 tokens
    """
    calculator = get_vision_token_calculator(model_id)
    if calculator is None:
        return 0
    
    tokens_per_image = calculator(image_height, image_width)
    return tokens_per_image * images_per_prompt


def is_vision_language_model(model_id: str) -> bool:
    """
    Check if a model is a vision-language model.
    
    Args:
        model_id: HuggingFace model repository ID
        
    Returns:
        True if model is a vision-language model, False otherwise
    """
    return model_id in VISION_TOKEN_CALCULATORS

