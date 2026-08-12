# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sentence-boundary text chunking shared by the TTS runners.

TTS models degrade (or hard-cap) beyond their trained utterance length, so long request
texts are split at sentence boundaries and synthesized per chunk. Moved verbatim from
speecht5_runner.chunk_text so the TTS runners share one implementation; the only behavioral
extension is the processor-less path for oversized sentences (see split_oversized)."""

import re
from typing import List

DEFAULT_CHUNK_SIZE = 256  # Maximum characters per text chunk


def chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    processor=None,
    max_tokens: int = 250,
) -> List[str]:
    """Split text into chunks that always end at sentence boundaries.

    Sentences are packed greedily into chunks until adding the next sentence would exceed
    max_chunk_size characters. A single sentence longer than the budget is split at clause
    boundaries (, ;): with a `processor` (HF tokenizer, speecht5) the budget is `max_tokens`
    tokens; without one (xtts) the budget is `max_chunk_size` characters, falling back to
    word-boundary splits for a single over-long clause.
    """
    if len(text) <= max_chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]

    def _too_big(candidate: str) -> bool:
        if processor is not None:
            return (
                processor(text=candidate, return_tensors="pt")["input_ids"].shape[1]
                > max_tokens
            )
        return len(candidate) > max_chunk_size

    def _split_giant_clause(clause: str) -> List[str]:
        # processor-less last resort: one clause longer than the char budget
        parts = []
        while len(clause) > max_chunk_size:
            cut = clause.rfind(" ", 0, max_chunk_size)
            cut = cut if cut > 0 else max_chunk_size
            parts.append(clause[:cut].strip())
            clause = clause[cut:].strip()
        if clause:
            parts.append(clause)
        return parts

    def split_oversized(sentence):
        if not _too_big(sentence):
            return [sentence]
        clauses = re.split(r"(?<=[,;])\s+", sentence)
        if processor is None:
            clauses = [p for c in clauses for p in _split_giant_clause(c)]
        parts = []
        current = ""
        for clause in clauses:
            candidate = (current + " " + clause).strip() if current else clause
            if not _too_big(candidate):
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = clause
        if current:
            parts.append(current)
        return parts if parts else [sentence]

    flat_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            flat_sentences.extend(split_oversized(s))

    chunks = []
    current = ""
    for sentence in flat_sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chunk_size:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks
