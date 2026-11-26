# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Task helpers for the Kimi-K3.1 evals: math grading that accepts a boxed equation.

The pinned lm-eval fork's math graders compare the content of the last \\boxed{} with
the target, so "\\boxed{21+49=70}" is wrong for the target 70. K3.1 often boxes a
whole equation. Here a grader's verdict stands whenever it is "correct". Only when it
is "wrong" is the same comparison repeated on conservative rewrites of the boxed
answer, so a sample can gain a point but never lose one, and nothing is computed or
guessed: "\\boxed{21+49}" stays wrong for 70.

process_results_aime wraps the fork's aime grader (aime25); process_results_math and
process_results_math_avg wrap its r1_evals grader (r1_aime24, r1_math500, HMMT).
"""

import re
from typing import Dict, List, Optional

from lm_eval.tasks.aime.utils import is_equiv, process_results
from lm_eval.tasks.r1_evals import utils as r1
from lm_eval.tasks.r1_evals.utils import doc_to_target  # noqa: F401  (r1_math500.yaml)

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
    # Matrix row spacing: a row break written "\\[2pt]" is a plain "\\".
    unspaced = re.sub(r"\\\\\s*\[[^\]]*\]", r"\\\\", s)
    if unspaced != s:
        s = unspaced
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


def _plus_minus(s: str) -> Optional[List[str]]:
    """"a\\pm b" with exactly one top-level \\pm -> ["a-b", "a+b"], else None."""
    p = _split_top_level(s, "\\pm")
    if len(p) == 2 and p[0].strip() and p[1].strip():
        return [f"{p[0]}-{p[1]}", f"{p[0]}+{p[1]}"]
    return None


def _r1_equal(a: str, b: str) -> bool:
    return r1.math_equal(r1.strip_answer_string(a), r1.strip_answer_string(b))


def _plus_minus_equal(pred: str, gold: str) -> bool:
    """"a\\pm b" against the two-element list "a-b, a+b", in either order and on either side."""
    for one, other in ((gold, pred), (pred, gold)):
        pair = _plus_minus(one)
        items = [x.strip() for x in _split_top_level(re.sub(r"\\[,;! ]", "", other), ",")]
        items = [x for x in items if x]
        if pair and len(items) == 2:
            if (_r1_equal(items[0], pair[0]) and _r1_equal(items[1], pair[1])) or (
                _r1_equal(items[0], pair[1]) and _r1_equal(items[1], pair[0])
            ):
                return True
    return False


def process_results_math(doc: dict, results: List[str]) -> Dict[str, int]:
    """exact_match as r1_evals reports it, plus boxed answers it rejected for their form."""
    if r1.process_results_math(doc, results)["exact_match"]:
        return {"exact_match": 1}
    boxed = last_boxed(results[0])
    if boxed is None:
        return {"exact_match": 0}
    gold = r1.postprocess_target(doc["answer"])
    for c in [boxed] + regrade_candidates(boxed):
        if r1.math_equal(r1.strip_answer_string(c), gold):
            return {"exact_match": 1}
        if _plus_minus_equal(r1.strip_answer_string(re.sub(r"\\[,;! ]", "", c)), gold):
            return {"exact_match": 1}
    return {"exact_match": 0}


def process_results_math_avg(doc: dict, results: List[str]) -> Dict[str, float]:
    """avg@k: the mean of process_results_math over every sampled answer of a problem."""
    responses = results[0] if results and isinstance(results[0], list) else results
    scores = [process_results_math(doc, [r])["exact_match"] for r in responses]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}
