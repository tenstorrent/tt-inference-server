# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Task helpers for the K2-Horizon model-card evals (HMMT Feb 2026, HLE text-only MC)."""

import contextlib
import io
import re
from typing import Dict, List

# Reuse the r1_evals math equivalence checker of the pinned lm-eval fork, so HMMT is
# graded exactly like r1_aime24 / r1_math500.
from lm_eval.tasks.r1_evals.utils import (  # noqa: F401
    math_equal,
    postprocess_target,
    process_results_math,
    strip_answer_string,
)

# HLE's own response format (system prompt of the official evaluation), for the
# multiple-choice questions it applies to.
HLE_MC_INSTRUCTION = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


def process_hle_text_mc(dataset):
    """Keep text-only multiple-choice questions."""
    return dataset.filter(
        lambda doc: (
            doc.get("answer_type") == "multipleChoice"
            and not (doc.get("image") or "").strip()
        )
    )


def hle_doc_to_text(doc) -> str:
    return f"{HLE_MC_INSTRUCTION}\n\n{doc['question']}"


_ANSWER_LINE = re.compile(
    r"(?im)^\s*\**\s*answer\s*\**\s*[:：]\s*\**\s*\(?\s*([A-Z])\b"
)
_BOXED = re.compile(r"\\boxed\{\s*\(?([A-Z])\)?\s*\}")


def extract_hle_choice(response: str):
    """Return the chosen letter from the last "Answer:" line, else the last \\boxed{X}."""
    text = re.sub(r".*?</think>", "", response or "", flags=re.DOTALL)
    for pattern in (_ANSWER_LINE, _BOXED):
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def process_results_hle_mc(doc: dict, results: List[str]) -> Dict[str, int]:
    gold = str(doc["answer"]).strip().strip("()").upper()[:1]
    return {"exact_match": int(extract_hle_choice(results[0]) == gold)}


# What the r1_evals normaliser (strip_answer_string) cannot read, seen on HMMT Feb 2026
# with DeepSeek V4.1 (17 of 132 correct answers rejected, exabox job 169463): a
# \displaystyle or spacing macro glued to the answer, a label longer than two
# characters (\ell=, BC=, \mathbb E[...]=), an a=b=c chain, an \approx tail, \text{}
# and a percent sign against a fractional gold. Every variant below is sympy-equal to
# the gold; the pre-clean runs only after the fork's grader has said no.
_WORD_MACROS = re.compile(r"\\(displaystyle|quad|qquad)(?![A-Za-z])")
_SPACING_MACROS = re.compile(r"\\[,;!:]")


def last_boxed(text: str):
    """The raw content of the last \\boxed{...} in ``text`` (balanced braces), or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for k in range(open_at, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : k]
    return None


def preclean_candidates(boxed: str) -> List[str]:
    """Normalised readings of a boxed answer: macros and \\text{} removed, the \\approx
    tail dropped, every segment of an ``a=b=c`` chain on its own (the label first, the
    value last), a percent turned into a fraction of 100."""
    text = _WORD_MACROS.sub("", boxed)
    text = _SPACING_MACROS.sub("", text)
    text = re.sub(r"\\approx.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    text = text.strip().rstrip(".")
    segments = [seg.strip() for seg in text.split("=") if seg.strip()] or [text]
    out: List[str] = []
    for seg in segments:
        percent = seg.endswith("\\%") or seg.endswith("%")
        cand = strip_answer_string(seg.rstrip("%").rstrip("\\"))
        if percent:
            cand = f"\\frac{{{cand}}}{{100}}"
        if cand and cand not in out:
            out.append(cand)
    return out


def process_results_math_preclean(doc: dict, results: List[str]) -> Dict[str, int]:
    """The fork's grader first; when it rejects, retry with the pre-cleaned readings of
    the reply's last boxed answer. Additive: nothing the fork accepts is lost."""
    graded = process_results_math(doc, results)
    if graded["exact_match"]:
        return graded
    response = re.sub(r".*?</think>(\\n)*", "", results[0] or "", flags=re.DOTALL)
    raw = last_boxed(response)
    if raw is None:
        return graded
    gold = postprocess_target(doc["answer"])
    for cand in preclean_candidates(raw):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                if math_equal(cand, gold):
                    return {"exact_match": 1}
        except Exception:  # noqa: BLE001  (the grader's parsers raise on exotic LaTeX)
            continue
    return graded


def process_results_math_avg(doc: dict, results: List[str]) -> Dict[str, float]:
    """avg@k: mean math equivalence (pre-cleaned) over every sampled answer of a problem."""
    responses = results[0] if results and isinstance(results[0], list) else results
    scores = [process_results_math_preclean(doc, [r])["exact_match"] for r in responses]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}
