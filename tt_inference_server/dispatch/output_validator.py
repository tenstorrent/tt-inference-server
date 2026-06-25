# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared output validator for the dispatch benchmark/test harness (issue #46).

Single source of truth for "did this model emit plausible English-prompt output,
or degenerate garbage?". Import-only and dependency-free: NO ttnn, NO torch — so it
can be unit-tested in CI with no Tenstorrent card and no model weights.

The cascade is the one tuned offline against labeled good/garbage 80-tok samples
(see tests/diagnostic/tune_validator.py, 12/12) PLUS the single-token-repetition
check that tune_validator dropped — a documented true-positive that catches
tokenizer-level loops the n-gram check can miss. Both bench harnesses
(tests/_run_model.py and tt-inference-server/bench.py) delegate here.

Signature:  validate_output(text, token_ids) -> (is_valid: bool, reason: str)

The reason-string PREFIXES are a frozen contract — sweep/diff tooling parses them.
Do not change the wording of an existing reason without updating that tooling:
    "empty output"
    "too short: <n> tokens"
    "repetition loop (token)"
    "repetition loop (<n>-gram x<count>)"
    "garbage: <n> non-Latin (CJK/Arabic/etc) chars"
    "garbage: <pct>% punctuation/symbols"
    "ok"

NOTE: the exotic-script / punctuation checks assume an ENGLISH-prompt smoke test.
They gate the *bench verdict* only, never model admission — a deliberately
multilingual or code-heavy novel model may need relaxed thresholds (pass a custom
ValidatorThresholds), and a passing verdict is necessary-not-sufficient for
degeneration-prone models (spot-check actual text; see BUG-MISTRAL-01).
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

# --- Named thresholds (promoted from the inline magic numbers) -------------
VALIDATION_MIN_TOKENS = 10  # reject runs shorter than this many tokens
TOKEN_REPETITION_FRACTION = 0.5  # >50% of token ids being one id => loop
PHRASE_LOOP_MIN_WORDS = 12  # only run the n-gram check on enough words
PHRASE_LOOP_NGRAMS = (1, 2, 3)  # n-gram sizes scanned for phrase loops
PHRASE_LOOP_FRACTION = 0.30  # one n-gram covering >30% of grams => loop
EXOTIC_CHAR_LIMIT = 3  # >this many non-Latin letters => garbage
PUNCT_FRACTION = 0.40  # >40% non-alphanumeric (non-space) => soup


@dataclass(frozen=True)
class ValidatorThresholds:
    """Tunable thresholds for validate_output. Defaults reproduce the tuned cascade."""

    min_tokens: int = VALIDATION_MIN_TOKENS
    token_repetition_fraction: float = TOKEN_REPETITION_FRACTION
    phrase_loop_min_words: int = PHRASE_LOOP_MIN_WORDS
    phrase_loop_ngrams: Tuple[int, ...] = PHRASE_LOOP_NGRAMS
    phrase_loop_fraction: float = PHRASE_LOOP_FRACTION
    exotic_char_limit: int = EXOTIC_CHAR_LIMIT
    punct_fraction: float = PUNCT_FRACTION


DEFAULT_THRESHOLDS = ValidatorThresholds()


def validate_output(
    text: str,
    token_ids: Optional[Sequence[int]],
    thresholds: Optional[ValidatorThresholds] = None,
) -> Tuple[bool, str]:
    """Return (is_valid, reason). Conservative: only fail on strong garbage signals.

    Args:
        text: the decoded model continuation.
        token_ids: the generated token id sequence. When None, the length and
            single-token-repetition checks are SKIPPED (text-only mode, for callers
            that don't have token ids, e.g. the legacy tt-inference-server/bench.py);
            the caller is then responsible for asserting a minimum token count.
        thresholds: optional override of the tuned defaults.
    """
    t = thresholds or DEFAULT_THRESHOLDS

    if not text or not text.strip():
        return False, "empty output"

    # Length + single-token repetition only when token ids are supplied.
    if token_ids is not None:
        n = len(token_ids)
        if n < t.min_tokens:
            return False, f"too short: {n} tokens"
        # 1. single-token repetition (cheap; catches tokenizer-level loops)
        if n:
            most_common_count = Counter(token_ids).most_common(1)[0][1]
            if most_common_count / n > t.token_repetition_fraction:
                return False, "repetition loop (token)"

    s = text.strip()
    words = s.split()

    # 2. phrase-loop: a short n-gram repeated heavily (period > 1 loops)
    if len(words) >= t.phrase_loop_min_words:
        for ngram in t.phrase_loop_ngrams:
            grams = [tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
            if not grams:
                continue
            _top, cnt = Counter(grams).most_common(1)[0]
            if cnt / len(grams) > t.phrase_loop_fraction:
                return False, f"repetition loop ({ngram}-gram x{cnt})"

    # 3. exotic-script chars: correct English output has 0 CJK/Arabic/Devanagari/etc
    exotic = sum(1 for c in s if c.isalpha() and "LATIN" not in unicodedata.name(c, ""))
    if exotic > t.exotic_char_limit:
        return False, f"garbage: {exotic} non-Latin (CJK/Arabic/etc) chars"

    # 4. punctuation/symbol soup (real prose is mostly letters/digits/spaces)
    nonspace = [c for c in s if not c.isspace()]
    if nonspace:
        punct = sum(1 for c in nonspace if not c.isalnum())
        if punct / len(nonspace) > t.punct_fraction:
            return False, f"garbage: {punct / len(nonspace):.0%} punctuation/symbols"

    return True, "ok"


__all__ = [
    "validate_output",
    "ValidatorThresholds",
    "DEFAULT_THRESHOLDS",
    "VALIDATION_MIN_TOKENS",
    "TOKEN_REPETITION_FRACTION",
    "PHRASE_LOOP_MIN_WORDS",
    "PHRASE_LOOP_NGRAMS",
    "PHRASE_LOOP_FRACTION",
    "EXOTIC_CHAR_LIMIT",
    "PUNCT_FRACTION",
]
