# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Humanity's Last Exam, no tools (centerforaisafety/hle's hle_eval protocol).

Each question goes out with the official answer-format instruction (exact
answer or multiple choice) ahead of the question text, plus its image when it
has one. The official judge prompt then decides each response. The official
judge is o3-mini with structured output; here the judge defaults to the served
model (the only model on the eval host) with thinking off, and its verdict is
read from the prompt's "correct: yes|no" line rather than a JSON schema, which
the TT samplers do not enforce. HLE_JUDGE_MODEL / HLE_JUDGE_BASE_URL /
HLE_JUDGE_API_KEY point it at another OpenAI-compatible judge.
"""

import base64
import io
import os
import random
import re
import time
from functools import lru_cache

from loguru import logger as eval_logger
from PIL import Image

SHUFFLE_SEED = 1234

# hle_eval/run_model_predictions.py
SYSTEM_EXACT_ANSWER = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
SYSTEM_MC = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

# hle_eval/run_judge_results.py
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""

# The judge's "correct: yes|no" line, tolerating Markdown bold around the field.
_VERDICT_RE = re.compile(
    r"^[\s*_#-]*correct[\s*_]*:[\s*_]*(yes|no)\b", re.IGNORECASE | re.MULTILINE
)
JUDGE_ATTEMPTS = 3
JUDGE_MAX_TOKENS = 4096


def process_docs(dataset):
    """The test split is grouped by subject; a fixed shuffle makes any --limit
    prefix a mix of categories (and of image / text-only questions)."""
    order = list(range(len(dataset)))
    random.Random(SHUFFLE_SEED).shuffle(order)
    return dataset.select(order)


def doc_to_visual(doc):
    image = doc.get("image") or ""
    if not image:
        return []
    data = image.split(",", 1)[1] if image.startswith("data:") else image
    return [Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    system = (
        SYSTEM_MC if doc["answer_type"] == "multipleChoice" else SYSTEM_EXACT_ANSWER
    )
    return f"{system}\n\n{doc['question']}"


@lru_cache(maxsize=1)
def _judge():
    """(client, model, extra_body) for the judge; the served model by default."""
    from openai import OpenAI

    base_url = os.environ.get("HLE_JUDGE_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = (
        os.environ.get("HLE_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=1800.0, max_retries=2)
    model = os.environ.get("HLE_JUDGE_MODEL")
    if model:
        return client, model, None
    model = client.models.list().data[0].id
    # The served model judges with thinking off: the verdict is a format task.
    return client, model, {"chat_template_kwargs": {"enable_thinking": False}}


def judge(question: str, response: str, correct_answer: str):
    """True / False, or None when no verdict could be read."""
    client, model, extra_body = _judge()
    prompt = JUDGE_PROMPT.format(
        question=question, response=response, correct_answer=correct_answer
    )
    for attempt in range(1, JUDGE_ATTEMPTS + 1):
        try:
            reply = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=JUDGE_MAX_TOKENS,
                temperature=0.0,
                extra_body=extra_body,
            )
            verdicts = _VERDICT_RE.findall(reply.choices[0].message.content or "")
            if verdicts:
                return verdicts[-1].lower() == "yes"
            eval_logger.warning(f"HLE judge gave no verdict (attempt {attempt})")
        except Exception as e:  # transient server errors: retry
            eval_logger.warning(f"HLE judge request failed (attempt {attempt}): {e}")
            time.sleep(min(2**attempt, 30))
    return None


def process_results(doc, results):
    response = results[0] or ""
    verdict = (
        judge(doc["question"], response, doc["answer"]) if response.strip() else False
    )
    payload = {
        "id": doc["id"],
        "category": doc["category"],
        "has_image": bool(doc.get("image")),
        "correct": bool(verdict),
        "judge_failed": verdict is None,
    }
    return {"hle_accuracy": payload, "hle_judge_failures": payload}


def aggregate_accuracy(results):
    return 100.0 * sum(r["correct"] for r in results) / len(results)


def aggregate_judge_failures(results):
    return sum(r["judge_failed"] for r in results)
