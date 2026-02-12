# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import re
from typing import List


class TextUtils:
    """Utility functions for text processing"""

    # Pre-compiled regex patterns (compile once at class load, not per call)
    _SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
    _SPACE_AFTER_PUNCT = re.compile(r"([.,!?;:])(?=[A-Za-z0-9])")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing EOS tokens and fixing punctuation spacing"""
        if not isinstance(text, str):
            return str(text)

        # Remove any remaining EOS tokens
        cleaned = text.replace("<EOS>", "")

        # Remove spaces before punctuation (using pre-compiled pattern)
        cleaned = TextUtils._SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
        # Ensure single space after punctuation (but not at end)
        cleaned = TextUtils._SPACE_AFTER_PUNCT.sub(r"\1 ", cleaned)

        return cleaned.strip()

    @staticmethod
    def strip_eos(text: str) -> str:
        """Minimal cleaning for streaming chunks - only remove EOS tokens"""
        if not isinstance(text, str):
            return str(text)
        return text.replace("<EOS>", "")

    @staticmethod
    def concatenate_chunks(chunks: List[str]) -> str:
        """Concatenate text chunks into final cleaned text"""
        texts = []
        for chunk in chunks:
            if not isinstance(chunk, str):
                raise ValueError(
                    f"Expected string chunk but got {type(chunk).__name__}. "
                )

            clean_text = TextUtils.clean_text(chunk)
            if clean_text:
                texts.append(clean_text)

        return TextUtils.clean_text(" ".join(texts))

    @staticmethod
    def extract_text(chunk) -> str:
        text_chunk = chunk
        start = None
        end = None
        if isinstance(text_chunk, tuple):
            text_chunk = text_chunk[0]
        if isinstance(text_chunk, list) and len(text_chunk) > 0:
            text_chunk = text_chunk[0]
            # we need to check again after getting first element from list
            if isinstance(text_chunk, list) and len(text_chunk) > 0:
                text_chunk = text_chunk[0]
                if "start" in text_chunk:
                    start = text_chunk["start"]
                if "end" in text_chunk:
                    end = text_chunk["end"]
                if "text" in text_chunk:
                    text_chunk = text_chunk["text"]

        if not isinstance(text_chunk, str):
            text_chunk = ""

        return TextUtils.clean_text(text_chunk), start, end
