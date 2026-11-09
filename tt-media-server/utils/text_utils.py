# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import re
from typing import List, Optional, Tuple


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
    def extract_text(chunk) -> Tuple[str, Optional[float], Optional[float]]:
        """Text, start and end of one generator result.

        ``chunk`` is a generator tuple whose first element holds one entry per
        batch item, or a single item from the batch path. An item is a string,
        or with ``return_timestamps`` a list of ``{"text", "start", "end"}``
        segments; the segment texts are joined and the times span the list.
        """
        text_chunk = chunk
        start = None
        end = None
        if isinstance(text_chunk, tuple):
            text_chunk = text_chunk[0]
        if (
            isinstance(text_chunk, list)
            and len(text_chunk) > 0
            and not isinstance(text_chunk[0], dict)
        ):
            text_chunk = text_chunk[0]
        if isinstance(text_chunk, list):
            segments = [s for s in text_chunk if isinstance(s, dict)]
            if segments:
                start = segments[0].get("start")
                end = segments[-1].get("end")
            text_chunk = " ".join(str(s.get("text", "")).strip() for s in segments)

        if not isinstance(text_chunk, str):
            text_chunk = ""

        return TextUtils.clean_text(text_chunk), start, end
