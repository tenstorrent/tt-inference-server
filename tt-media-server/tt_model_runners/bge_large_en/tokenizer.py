# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from transformers import AutoTokenizer


class BGETokenizer:
    """Handles tokenization for BGE Large EN model."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self):
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        return self._tokenizer

    def tokenize(self, text_inputs: list, max_length: int) -> dict:
        """
        Tokenize text inputs.

        :param text_inputs: List of text strings to tokenize
        :param max_length: Maximum sequence length
        :return: Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        return self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def calculate_token_counts(self, tokenized: dict, num_requests: int) -> list:
        """
        Calculate token counts for each request, excluding padding.

        :param tokenized: Tokenized output from tokenizer
        :param num_requests: Number of actual requests (excluding padding)
        :return: List of token counts per request
        """
        input_ids = tokenized["input_ids"]

        if "attention_mask" in tokenized:
            return tokenized["attention_mask"].sum(dim=1).tolist()[:num_requests]

        # Count non-padding tokens
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        return [
            (input_ids[i] != pad_token_id).sum().item()
            for i in range(num_requests)
        ]
