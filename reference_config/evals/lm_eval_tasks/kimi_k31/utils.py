# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Task helpers for the Kimi-K3.1 evals: AIME grading that accepts a boxed equation.

The pinned lm-eval fork's AIME grader compares the content of the last \\boxed{} with
the target as normalised strings, so "\\boxed{21+49=70}" is wrong for the target 70.
Here the fork's verdict stands whenever it is "correct". Only when it is "wrong" is the
same comparison repeated on conservative rewrites of the boxed answer, so a sample can
gain a point but never lose one, and nothing is computed or guessed: "\\boxed{21+49}"
stays wrong for 70.
"""

import re
from typing import Dict, List, Optional

from lm_eval.tasks.aime.utils import is_equiv, process_results

_OPEN, _CLOSE = "{([", "})]"
# A box holding several values or an inequality is never reduced to one side of an "=".
_SEVERAL = re.compile(
    r",|;|\\text\{\s*(and|or)\s*\}|\\quad|\\qquad|<|>|\\le|\\ge|\\ne|\\lt|\\gt"
)


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split s at the occurrences of sep outside every brace, paren and bracket."""
    parts, depth, i, start = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s) and s[i + 1] in "{}":
            i += 2  # an escaped brace does not change the depth
            continue
        if c in _OPEN:
            depth += 1
        elif c in _CLOSE:
            depth -= 1
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i])
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(s[start:])
    return parts


def last_boxed(response: str) -> Optional[str]:
    """Content of the last \\boxed{...} after any </think>, or None."""
    text = re.sub(r".*?</think>", "", response, flags=re.DOTALL)
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + 7 : j - 1] if depth == 0 else None


def regrade_candidates(boxed: str) -> List[str]:
    """Rewrites of a boxed answer to try after the grader rejected it."""
    out = []
    s = boxed.strip().rstrip(".")
    # The exact value before a trailing approximation: "\frac{a}{b}\approx 0.03".
    parts = _split_top_level(s, "\\approx")
    if len(parts) > 1 and parts[0].strip():
        s = parts[0].strip()
        out.append(s)
    # The right-hand side of the last "=" of a single equation or chain. Never for a box
    # with several values or an inequality, so "x=3, y=5" is not reduced to "5".
    sides = _split_top_level(s, "=")
    if len(sides) > 1 and not _SEVERAL.search(s) and all(p.strip() for p in sides):
        out.append(sides[-1].strip())
    return out


def process_results_aime(doc: dict, results: List[str]) -> Dict[str, int]:
    """exact_match as the fork reports it, plus boxed answers it rejected for their form."""
    if process_results(doc, results)["exact_match"]:
        return {"exact_match": 1}
    boxed = last_boxed(results[0])
    if boxed is None:
        return {"exact_match": 0}
    target = str(doc[next(k for k in doc if k.lower() == "answer")])
    return {"exact_match": int(any(is_equiv(c, target) for c in regrade_candidates(boxed)))}
