# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Union, List

class TranscriptUtils:
    """Utility functions for transcript processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove EOS tokens and clean text"""
        if not isinstance(text, str):
            return str(text)
        return text.replace("<EOS>", "").strip()
    
    @staticmethod
    def concatenate_chunks(chunks: List[Union[str, dict]]) -> str:
        """Concatenate text chunks into final transcript"""
        texts = []
        for chunk in chunks:
            if isinstance(chunk, str):
                clean_text = TranscriptUtils.clean_text(chunk)
                if clean_text:
                    texts.append(clean_text)
            elif isinstance(chunk, dict) and "text" in chunk:
                clean_text = TranscriptUtils.clean_text(chunk["text"])
                if clean_text:
                    texts.append(clean_text)
        
        return " ".join(texts)