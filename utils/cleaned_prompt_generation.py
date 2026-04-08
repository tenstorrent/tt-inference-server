#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Cleaned Prompt Generation Utility

This module provides functions to generate random prompts with specific token lengths,
and process them through encoding/decoding cycles to produce stable token sequences.
It leverages code from the parallel token analysis script.
"""

import random
import torch
import logging
from transformers import AutoTokenizer
from typing import List, Optional, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def tokenize_encode_client(
    prompt: str, tokenizer: Any, max_length: Optional[int], truncation: bool = False
) -> List[int]:
    """Encode a prompt to tokens using the client-side tokenizer"""
    return tokenizer.encode(
        prompt, add_special_tokens=False, truncation=truncation, max_length=max_length
    )


def tokenize_decode_client(encoded_prompt: List[int], tokenizer: Any) -> str:
    """Decode tokens back to a string using the client-side tokenizer"""
    return tokenizer.decode(encoded_prompt)


def tokenize_encode_server(
    prompt: str,
    tokenizer: str,
    max_length: Optional[int],
    client: Any,
    truncation: bool = False,
) -> List[int]:
    """Encode a prompt to tokens using the server-side tokenizer"""
    tokens = client.tokenize(prompt)["tokens"]
    # Truncate tokens if max_length is provided and tokens exceed that length
    if truncation and max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens


def tokenize_decode_server(tokens: List[int], tokenizer: str, client: Any) -> str:
    """Decode tokens back to a string using the server-side tokenizer"""
    prompt = client.detokenize(tokens, tokenizer)["prompt"]
    return prompt


def get_tokenizer(model_name: str, fallback_model: str = "gpt2"):
    """Get tokenizer with fallback if primary model fails"""
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}"
        )
        return tokenizer, model_name
    except Exception as e:
        logger.warning(f"Failed to load primary tokenizer: {str(e)}")
        try:
            logger.info(f"Trying fallback tokenizer: {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            logger.info(
                f"Using fallback tokenizer with vocab size: {tokenizer.vocab_size}"
            )
            return tokenizer, fallback_model
        except Exception as e2:
            logger.error(f"Failed to load fallback tokenizer: {str(e2)}")
            raise RuntimeError("Could not load any tokenizer")


def generate_stable_prompt_tokens(
    input_length: int,
    max_length: int,
    model_name: str,
    server_tokenizer: bool = False,
    client: Any = None,
    fallback_model: str = "gpt2",
    seed: Optional[int] = None,
    preloaded_tokenizer: Any = None,
) -> List[int]:
    """
    Generate a stable sequence of tokens by creating random tokens, then decoding and re-encoding.

    Args:
        input_length: Target number of tokens to generate
        max_length: Maximum allowed token length
        model_name: Name of the model/tokenizer to use
        server_tokenizer: Whether to use server-side tokenization
        client: PromptClient instance for server-side tokenization
        fallback_model: Fallback model to use if the primary model fails
        seed: Random seed for reproducibility
        preloaded_tokenizer: Pre-loaded tokenizer to reuse (avoids loading multiple times)

    Returns:
        A list of integer token IDs
    """
    # Apply vLLM safety buffer (similar to prompt_generation.py)
    # vLLM appears to add extra token on receipt of prompt (likely BOS token)
    safe_input_length = max(1, input_length - 1)
    safe_max_length = max(1, max_length - 1)

    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(random.randint(0, 128000))

    # Load or use pre-loaded tokenizer if using client-side tokenization
    if not server_tokenizer:
        if preloaded_tokenizer is not None:
            # Use pre-loaded tokenizer for efficiency
            tokenizer = preloaded_tokenizer
            actual_model = model_name
        else:
            # Load tokenizer (fallback for backward compatibility)
            tokenizer, actual_model = get_tokenizer(model_name, fallback_model)
        vocab_size = tokenizer.vocab_size
    else:
        tokenizer = model_name  # Just pass the model name for server tokenization
        # Estimate vocab size - could be retrieved from server if available
        vocab_size = 32000  # Default estimate for LLM models

    # Generate random tokens using safe length
    token_ids = torch.randint(0, vocab_size, (safe_input_length,)).tolist()

    # First decoding - convert tokens to text
    if server_tokenizer:
        prompt_text = tokenize_decode_server(token_ids, tokenizer, client)
    else:
        prompt_text = tokenize_decode_client(token_ids, tokenizer)

    # First encoding - convert text back to tokens with truncation using safe length
    if server_tokenizer:
        encoded_tokens = tokenize_encode_server(
            prompt_text, tokenizer, safe_max_length, client, truncation=True
        )
    else:
        encoded_tokens = tokenize_encode_client(
            prompt_text, tokenizer, max_length=safe_max_length, truncation=True
        )

    # Second decoding - convert tokens to text again
    if server_tokenizer:
        decoded_text = tokenize_decode_server(encoded_tokens, tokenizer, client)
    else:
        decoded_text = tokenize_decode_client(encoded_tokens, tokenizer)

    # Final encoding - convert text back to tokens with truncation using safe length
    if server_tokenizer:
        final_tokens = tokenize_encode_server(
            decoded_text, tokenizer, safe_max_length, client, truncation=True
        )
    else:
        final_tokens = tokenize_encode_client(
            decoded_text, tokenizer, max_length=safe_max_length, truncation=True
        )

    return final_tokens


if __name__ == "__main__":
    # Example usage
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    input_length = 1024
    max_length = 2048

    # Generate a stable prompt
    tokens = generate_stable_prompt_tokens(
        input_length=input_length,
        max_length=max_length,
        model_name=model_name,
        server_tokenizer=False,
    )

    print(f"Generated stable token sequence with {len(tokens)} tokens")
