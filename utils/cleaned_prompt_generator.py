#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Cleaned Prompt Generation Utility

This module provides a class-based interface to generate random prompts with specific token lengths,
and process them through encoding/decoding cycles to produce stable token sequences.
It leverages code from the parallel token analysis script.
"""

import random
import torch
import logging
from transformers import AutoTokenizer
from typing import List, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CleanedPromptGenerator:
    """
    A class for generating stable token sequences through multiple encode/decode cycles.
    
    Supports both client-side (local) and server-side tokenization.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        fallback_model: str = "gpt2",
        server_tokenizer: bool = False,
        client: Any = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the CleanedPromptGenerator.
        
        Args:
            model_name: Name of the model/tokenizer to use
            fallback_model: Fallback model to use if the primary model fails
            server_tokenizer: Whether to use server-side tokenization
            client: PromptClient instance for server-side tokenization
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.server_tokenizer = server_tokenizer
        self.client = client
        self.seed = seed
        
        # Initialize tokenizer and model info
        if not self.server_tokenizer:
            self.tokenizer, self.actual_model = self._get_tokenizer()
            self.vocab_size = self.tokenizer.vocab_size
        else:
            self.tokenizer = model_name  # Just pass the model name for server tokenization
            self.actual_model = model_name
            # Estimate vocab size - could be retrieved from server if available
            self.vocab_size = 32000  # Default estimate for LLM models
            
        logger.info(f"Initialized CleanedPromptGenerator with model: {self.actual_model}")
        logger.info(f"Using {'server-side' if self.server_tokenizer else 'client-side'} tokenization")
        logger.info(f"Vocab size: {self.vocab_size}")
    
    def _get_tokenizer(self):
        """Get tokenizer with fallback if primary model fails"""
        try:
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}")
            return tokenizer, self.model_name
        except Exception as e:
            logger.warning(f"Failed to load primary tokenizer: {str(e)}")
            try:
                logger.info(f"Trying fallback tokenizer: {self.fallback_model}")
                tokenizer = AutoTokenizer.from_pretrained(self.fallback_model)
                logger.info(f"Using fallback tokenizer with vocab size: {tokenizer.vocab_size}")
                return tokenizer, self.fallback_model
            except Exception as e2:
                logger.error(f"Failed to load fallback tokenizer: {str(e2)}")
                raise RuntimeError("Could not load any tokenizer")
    
    def _tokenize_encode_client(self, prompt: str, max_length: Optional[int], truncation: bool = False) -> List[int]:
        """Encode a prompt to tokens using the client-side tokenizer"""
        return self.tokenizer.encode(
            prompt, add_special_tokens=False, truncation=truncation, max_length=max_length
        )

    def _tokenize_decode_client(self, encoded_prompt: List[int]) -> str:
        """Decode tokens back to a string using the client-side tokenizer"""
        return self.tokenizer.decode(encoded_prompt)

    def _tokenize_encode_server(self, prompt: str, max_length: Optional[int], truncation: bool = False) -> List[int]:
        """Encode a prompt to tokens using the server-side tokenizer"""
        if self.client is None:
            raise ValueError("Client instance required for server-side tokenization")
        
        tokens = self.client.entokenize(prompt)["tokens"]
        # Truncate tokens if max_length is provided and tokens exceed that length
        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def _tokenize_decode_server(self, tokens: List[int]) -> str:
        """Decode tokens back to a string using the server-side tokenizer"""
        if self.client is None:
            raise ValueError("Client instance required for server-side tokenization")
        
        prompt = self.client.detokenize(tokens, self.tokenizer)["prompt"]
        return prompt
    
    def _encode(self, prompt: str, max_length: Optional[int], truncation: bool = False) -> List[int]:
        """Encode using the configured tokenization method"""
        if self.server_tokenizer:
            return self._tokenize_encode_server(prompt, max_length, truncation)
        else:
            return self._tokenize_encode_client(prompt, max_length, truncation)
    
    def _decode(self, tokens: List[int]) -> str:
        """Decode using the configured tokenization method"""
        if self.server_tokenizer:
            return self._tokenize_decode_server(tokens)
        else:
            return self._tokenize_decode_client(tokens)
    
    def generate_stable_tokens(
        self, 
        input_length: int, 
        max_length: int,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Generate a stable sequence of tokens by creating random tokens, then decoding and re-encoding.
        
        Args:
            input_length: Target number of tokens to generate
            max_length: Maximum allowed token length
            seed: Random seed for reproducibility (overrides instance seed if provided)
            
        Returns:
            A list of integer token IDs
        """
        # Set random seed - use parameter seed if provided, otherwise instance seed
        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            torch.manual_seed(effective_seed)
            random.seed(effective_seed)
            logger.info(f"Using seed: {effective_seed}")
        else:
            # Generate a random seed
            random_seed = random.randint(0, 128000)
            torch.manual_seed(random_seed)
            logger.info(f"Using random seed: {random_seed}")
        
        # Generate random tokens
        token_ids = torch.randint(0, self.vocab_size, (input_length,)).tolist()
        logger.info(f"Generated {len(token_ids)} initial random tokens")
        
        # First decoding - convert tokens to text
        prompt_text = self._decode(token_ids)
        logger.debug(f"First decode completed, text length: {len(prompt_text)}")
        
        # First encoding - convert text back to tokens with truncation
        encoded_tokens = self._encode(prompt_text, max_length=max_length, truncation=True)
        logger.debug(f"First re-encode completed, token count: {len(encoded_tokens)}")
        
        # Second decoding - convert tokens to text again
        decoded_text = self._decode(encoded_tokens)
        logger.debug(f"Second decode completed, text length: {len(decoded_text)}")
        
        # Final encoding - convert text back to tokens with truncation
        final_tokens = self._encode(decoded_text, max_length=max_length, truncation=True)
        logger.info(f"Generated stable token sequence with {len(final_tokens)} tokens")
        
        return final_tokens
    
    def generate_multiple_stable_tokens(
        self, 
        input_length: int, 
        max_length: int,
        num_sequences: int,
        base_seed: Optional[int] = None
    ) -> List[List[int]]:
        """
        Generate multiple stable token sequences.
        
        Args:
            input_length: Target number of tokens to generate for each sequence
            max_length: Maximum allowed token length for each sequence
            num_sequences: Number of token sequences to generate
            base_seed: Base seed for reproducibility (each sequence gets base_seed + i)
            
        Returns:
            A list of lists, each containing integer token IDs
        """
        sequences = []
        
        for i in range(num_sequences):
            # Use incremental seeds for reproducibility while ensuring variety
            seq_seed = (base_seed + i) if base_seed is not None else None
            tokens = self.generate_stable_tokens(input_length, max_length, seed=seq_seed)
            sequences.append(tokens)
            logger.info(f"Generated sequence {i+1}/{num_sequences} with {len(tokens)} tokens")
            
        return sequences

#
# def main():
#     """Example usage of the CleanedPromptGenerator class"""
#     # Example with client-side tokenization
#     print("=== Client-side Tokenization Example ===")
#     generator = CleanedPromptGenerator(
#         model_name="meta-llama/Llama-3.1-8B-Instruct",
#         server_tokenizer=False,
#         seed=42
#     )
#
#     tokens = generator.generate_stable_tokens(
#         input_length=1024,
#         max_length=2048
#     )
#
#     print(f"Generated stable token sequence with {len(tokens)} tokens")
#     print(f"First 20 tokens: {tokens[:20]}")
#     print(f"Last 20 tokens: {tokens[-20:]}")
#
#     # Example generating multiple sequences
#     print("\n=== Multiple Sequences Example ===")
#     sequences = generator.generate_multiple_stable_tokens(
#         input_length=128,
#         max_length=128,
#         num_sequences=3,
#         base_seed=100
#     )
#
#     for i, seq in enumerate(sequences):
#         print(f"Sequence {i+1}: {len(seq)} tokens - {seq[:10]}...")

#
# if __name__ == "__main__":
#     main()