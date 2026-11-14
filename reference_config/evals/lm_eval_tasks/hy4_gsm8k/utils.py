# SPDX-License-Identifier: Apache-2.0
"""Boxed-aware GSM8K answer extraction for Hy4-Preview (pure Python, no lm_eval import: unit-testable on CPU).

Why: Hy4 answers in LaTeX (``$$ \\boxed{99077} $$``). lm-eval's flexible-extract takes the LAST number-like match ``(-?[$0-9.,]{2,})|(-?[0-9]+)``, which is the
closing ``$$`` (405 of 1319 responses), so flexible-extract scored 0.6566 where the same responses score 0.9644 with this rule (serve-hy4 run, 2026-11-14).
Rule, in order: (1) content of the last ``\\boxed{...}`` (balanced braces), (2) 'answer is N' / 'answer: N' / '#### N', (3) the last number OUTSIDE math
delimiters ($..$, $$..$$, \\(..\\), \\[..\\]), else the last number anywhere. Normalisation: strip $ , % and units, trailing '.', numeric value; compared numerically.
"""
import math
import re
from typing import List, Optional

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_NUM_RE = re.compile(_NUM)
_ANS_RE = re.compile(r"(?:answer\s*(?:is)?\s*[:：]?|####)\s*\**\s*\$?\s*\\?\(?\s*(" + _NUM + ")", re.I)
_MATH_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*?\$|\\\(.*?\\\)|\\\[.*?\\\]", re.S)


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (also \\fbox), balanced braces; None if absent or unterminated."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            best = text[m.end():i - 1]
    return best


def _to_number(s: str) -> Optional[float]:
    s = re.sub(r"\\text\{[^{}]*\}|\\(?:mathrm|textbf)\{([^{}]*)\}", lambda m: m.group(1) or "", s)
    s = s.replace("\\$", "").replace("\\%", "").replace("\\,", "").replace("\\!", "")
    m = _NUM_RE.search(s.replace("$", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_number(resp: str) -> Optional[float]:
    b = last_boxed(resp)
    if b is not None:
        v = _to_number(b)
        if v is not None:
            return v
    m = _ANS_RE.findall(resp)
    if m:
        return _to_number(m[-1])
    outside = _MATH_RE.sub(" ", resp)
    n = _NUM_RE.findall(outside)
    if n:
        return _to_number(n[-1])
    n = _NUM_RE.findall(resp)
    return _to_number(n[-1]) if n else None


def canon(v: Optional[float]) -> str:
    """Canonical string for lm-eval exact_match: integers without '.0', else shortest repr; '' when nothing extracted."""
    if v is None or math.isnan(v):
        return ""
    return str(int(v)) if float(v).is_integer() else repr(round(v, 6))


def same(pred: Optional[float], target: str) -> bool:
    t = _to_number(str(target))
    return pred is not None and t is not None and math.isclose(pred, t, abs_tol=1e-6)


def boxed_filter(resps: List[List[str]], docs=None) -> List[List[str]]:
    """lm-eval ``custom`` filter: one canonical numeric string per response (exact_match then compares with the target after regexes_to_ignore)."""
    return [[canon(extract_number(r)) for r in rs] for rs in resps]
