# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Task helpers for the google/gemma-4-31B-it model-card evals the pinned lm-eval fork
does not ship: AIME 2026 (MathArena), BIG-Bench Extra Hard, and HLE (text-only)."""

import json
import os
import re
import urllib.request
from typing import Dict, List, Optional

# Reuse the r1_evals math equivalence checker of the pinned lm-eval fork, so AIME 2026
# is graded exactly like r1_aime24 / r1_math500.
from lm_eval.tasks.r1_evals.utils import process_results_math  # noqa: F401


def _strip_thinking(response: str) -> str:
    """Drop an inline <think>...</think> block if the server did not split it out."""
    return re.sub(r".*?</think>", "", response or "", flags=re.DOTALL)


# ----------------------------------------------------------------------------------
# AIME 2026 (avg@k)
# ----------------------------------------------------------------------------------
def process_results_math_avg(doc: dict, results: List[str]) -> Dict[str, float]:
    """avg@k: mean r1_evals math equivalence over every sampled answer of a problem."""
    responses = results[0] if results and isinstance(results[0], list) else results
    scores = [process_results_math(doc, [r])["exact_match"] for r in responses]
    return {"exact_match": sum(scores) / len(scores) if scores else 0.0}


# ----------------------------------------------------------------------------------
# BIG-Bench Extra Hard
# The extractor / matcher below is the official google-deepmind/bbeh bbeh/evaluate.py,
# ported verbatim (Apache-2.0) so scores are comparable with the paper.
# ----------------------------------------------------------------------------------
BBEH_ANSWER_INSTRUCTION = (
    "Think step by step, then finish with the final answer on its own line in the "
    "form: The answer is: <answer>"
)


def bbeh_doc_to_text(doc) -> str:
    return f"{doc['input']}\n\n{BBEH_ANSWER_INSTRUCTION}"


def process_bbeh_mini(dataset):
    return dataset.filter(lambda doc: int(doc.get("mini", 0)) == 1)


def _bbeh_strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def _bbeh_extract_answer(sample: str) -> str:
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return _bbeh_strip_latex(answer)


def _bbeh_fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def _bbeh_preprocess_sample(sample: str) -> str:
    prediction = _bbeh_extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def _bbeh_preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def bbeh_evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = _bbeh_preprocess_sample(sample)
    reference = _bbeh_preprocess_reference(reference)
    return _bbeh_fuzzy_match(prediction, reference)


def process_results_bbeh(doc: dict, results: List[str]) -> Dict[str, int]:
    response = _strip_thinking(results[0] if results else "")
    return {"exact_match": int(bbeh_evaluate_correctness(response, doc["target"]))}


# ----------------------------------------------------------------------------------
# Humanity's Last Exam (text-only)
# ----------------------------------------------------------------------------------
# HLE's official response formats (system prompts of the official evaluation).
HLE_MC_INSTRUCTION = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
HLE_EXACT_INSTRUCTION = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

# The official judge prompt (centerforaisafety/hle, hle_eval/run_judge_results.py).
HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def process_hle_text(dataset):
    """Keep text-only questions (no image), both multiple-choice and exact-answer."""
    return dataset.filter(lambda doc: not (doc.get("image") or "").strip())


def hle_doc_to_text(doc) -> str:
    instruction = (
        HLE_MC_INSTRUCTION
        if doc.get("answer_type") == "multipleChoice"
        else HLE_EXACT_INSTRUCTION
    )
    return f"{instruction}\n\n{doc['question']}"


_HLE_ANSWER_LINE = re.compile(
    r"(?im)^\s*\**\s*(?:exact\s+)?answer\s*\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$"
)
_HLE_BOXED = re.compile(r"\\boxed\{\s*(.+?)\s*\}\s*$")


def hle_extract_answer(response: str) -> Optional[str]:
    """Return the text after the last "Answer:" / "Exact Answer:" line, else the last \\boxed{}."""
    text = _strip_thinking(response)
    found = _HLE_ANSWER_LINE.findall(text)
    if found:
        return found[-1].strip()
    found = _HLE_BOXED.findall(text)
    return found[-1].strip() if found else None


def _hle_norm(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.replace("\\,", "").replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def hle_deterministic_match(doc: dict, response: str) -> int:
    extracted = hle_extract_answer(response)
    if extracted is None:
        return 0
    gold = str(doc["answer"]).strip()
    if doc.get("answer_type") == "multipleChoice":
        m = re.match(r"^\(?\s*([A-Za-z])\b", extracted)
        return int(bool(m) and m.group(1).upper() == gold.strip("()").upper()[:1])
    pred_n, gold_n = _hle_norm(extracted), _hle_norm(gold)
    if pred_n == gold_n:
        return 1
    try:
        return int(abs(float(pred_n) - float(gold_n)) <= 1e-6 * max(1.0, abs(float(gold_n))))
    except ValueError:
        return 0


def _hle_judge(doc: dict, response: str) -> Optional[int]:
    """Official LLM-judge verdict via an OpenAI-compatible endpoint, or None if unset."""
    model = os.getenv("HLE_JUDGE_MODEL")
    if not model:
        return None
    base_url = (os.getenv("HLE_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
    api_key = os.getenv("HLE_JUDGE_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    if not base_url:
        return None
    prompt = HLE_JUDGE_PROMPT.format(
        question=doc["question"], response=_strip_thinking(response), correct_answer=doc["answer"]
    )
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 1024}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = json.load(resp)["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 - a judge outage must not abort the eval
        return None
    m = re.search(r"(?im)^\s*\**\s*correct\s*\**\s*[:：]\s*\**\s*(yes|no)", content or "")
    return int(m.group(1).lower() == "yes") if m else None


def process_results_hle(doc: dict, results: List[str]) -> Dict[str, int]:
    response = results[0] if results else ""
    det = hle_deterministic_match(doc, response)
    judged = _hle_judge(doc, response)
    return {"exact_match": det, "judge_match": det if judged is None else judged}


# ----------------------------------------------------------------------------------
# MMMLU (generative, per locale)
# ----------------------------------------------------------------------------------
MMMLU_INSTRUCTION = (
    "Answer the following multiple choice question. Think it through, then finish with "
    "the final answer on its own line in the form: Answer: <letter>"
)


MMMLU_PER_LOCALE_DEFAULT = 300


def process_mmmlu_shuffle(dataset):
    """Fixed-seed shuffle, then keep MMMLU_PER_LOCALE questions (default 300) per locale.

    A full generative pass over 196,588 questions is out of reach on one QB2, so the
    task itself bounds the sample; a further --limit (CI) applies on top of this."""
    k = int(os.getenv("MMMLU_PER_LOCALE", MMMLU_PER_LOCALE_DEFAULT))
    shuffled = dataset.shuffle(seed=1234)
    return shuffled.select(range(min(k, len(shuffled)))) if k > 0 else shuffled


def mmmlu_doc_to_text(doc) -> str:
    return (
        f"{MMMLU_INSTRUCTION}\n\n{doc['Question'].strip()}\n"
        f"A. {doc['A']}\nB. {doc['B']}\nC. {doc['C']}\nD. {doc['D']}"
    )


_LETTER_LINE = re.compile(r"(?im)^\s*\**\s*answer\s*\**\s*[:：]\s*\**\s*\(?\s*([A-D])\b")
_LETTER_BOXED = re.compile(r"\\boxed\{\s*\(?([A-D])\)?\s*\}")


def extract_letter(response: str) -> Optional[str]:
    text = _strip_thinking(response)
    for pattern in (_LETTER_LINE, _LETTER_BOXED):
        found = pattern.findall(text)
        if found:
            return found[-1].upper()
    return None


def process_results_mmmlu(doc: dict, results: List[str]) -> Dict[str, int]:
    gold = str(doc["Answer"]).strip().upper()[:1]
    return {"exact_match": int(extract_letter(results[0] if results else "") == gold)}


# ----------------------------------------------------------------------------------
# LiveCodeBench v6: thin wrappers over the pinned lm-eval fork's livecodebench task so
# the prompt format and the local test-case execution grader stay identical to it.
# ----------------------------------------------------------------------------------
from lm_eval.tasks.livecodebench import utils as _lcb  # noqa: E402


def process_lcb_v6(dataset):
    start = os.getenv("LCB_START_DATE")
    end = os.getenv("LCB_END_DATE")
    if not start and not end:
        return dataset
    return dataset.filter(
        lambda doc: (not start or doc["contest_date"][:10] >= start)
        and (not end or doc["contest_date"][:10] <= end)
    )


def lcb_doc_to_text(doc) -> str:
    return _lcb.doc_to_text_with_format(doc)


def lcb_doc_to_target(doc):
    return _lcb.doc_to_target(doc)


def process_results_lcb(doc: dict, results: List[str]) -> Dict[str, float]:
    # The fork's grader extracts the last ```python block itself; strip an inline
    # thinking block first so reasoning text cannot be mistaken for code.
    return _lcb.process_results(doc, [_strip_thinking(r) for r in results])
