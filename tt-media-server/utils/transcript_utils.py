# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List
import re

class TranscriptUtils:
    """Utility functions for transcript processing"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing EOS tokens and fixing punctuation spacing"""
        if not isinstance(text, str):
            return str(text)

        # Remove any remaining EOS tokens
        cleaned = text.replace("<EOS>", "")

        # Remove spaces before punctuation
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)
        # Ensure single space after punctuation (but not at end)
        cleaned = re.sub(r'([.,!?;:])(?=[A-Za-z0-9])', r'\1 ', cleaned)

        return cleaned.strip()

    @staticmethod
    def remove_trailing_angle_bracket(text: str) -> str:
        """Remove trailing '<' character if present"""
        if isinstance(text, str) and text.endswith('<'):
            return text[:-1]
        return text

    @staticmethod
    def concatenate_chunks(chunks: List[str]) -> str:
        """Concatenate text chunks into final transcript"""
        texts = []
        for chunk in chunks:
            if not isinstance(chunk, str):
                raise ValueError(f"Expected string chunk but got {type(chunk).__name__}. ")

            clean_text = TranscriptUtils.clean_text(chunk)
            if clean_text:
                texts.append(clean_text)

        return TranscriptUtils.clean_text(" ".join(texts))