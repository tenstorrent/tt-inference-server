# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from transformers import AutoTokenizer


class EmbeddingTokenizer:
    """Generic tokenizer for embedding models (BGE, Qwen3, etc.)."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def tokenize(self, text_inputs: list[str], max_length: int) -> dict:
        return self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def calculate_token_counts(self, tokenized: dict, num_requests: int) -> list[int]:
        input_ids = tokenized["input_ids"]
        if "attention_mask" in tokenized:
            return tokenized["attention_mask"].sum(dim=1).tolist()[:num_requests]
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )
        return [
            (input_ids[i] != pad_token_id).sum().item() for i in range(num_requests)
        ]
