"""Task helpers for the K2-Horizon model-card evals (HMMT Feb 2026, HLE text-only MC)."""

import re
from typing import Dict, List

# Reuse the r1_evals math equivalence checker of the pinned lm-eval fork, so HMMT is
# graded exactly like r1_aime24 / r1_math500.
from lm_eval.tasks.r1_evals.utils import process_results_math  # noqa: F401

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
        lambda doc: doc.get("answer_type") == "multipleChoice" and not (doc.get("image") or "").strip()
    )


def hle_doc_to_text(doc) -> str:
    return f"{HLE_MC_INSTRUCTION}\n\n{doc['question']}"


_ANSWER_LINE = re.compile(r"(?im)^\s*\**\s*answer\s*\**\s*[:：]\s*\**\s*\(?\s*([A-Z])\b")
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


def process_results_math_avg(doc: dict, results: List[str]) -> Dict[str, float]:
    """avg@k: mean r1_evals math equivalence over every sampled answer of a problem."""
    responses = results[0] if results and isinstance(results[0], list) else results
    scores = [process_results_math(doc, [r])["exact_match"] for r in responses]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}
