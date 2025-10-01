# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import List

class TranscriptUtils:
    """Utility functions for transcript processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Remove EOS tokens and clean text"""
        if not isinstance(text, str):
            return str(text)
        return text.replace("<EOS>", "").strip()
    
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
        
        return " ".join(texts)