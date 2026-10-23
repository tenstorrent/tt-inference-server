# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""BIG-Bench Extra Hard (BBEH/bbeh): prompt, ordering and the official grader.

``preprocess_sample`` / ``fuzzy_match`` are google-deepmind/bbeh's
``bbeh/evaluate.py``, unchanged apart from formatting.
"""

from itertools import zip_longest

import datasets

# The BBEH paper's answer-format instruction, appended to every question.
ANSWER_INSTRUCTION = (
    "Think step by step, and when you provide the final answer, please use the "
    'prefix "The answer is:" without any modification, and provide the answer '
    "directly, with no formatting, no bolding, and no markup. For instance: "
    '"The answer is: 42" or "The answer is: yes". If the question is multiple '
    "choice with a single correct answer, the final answer must only be the letter "
    'corresponding to the correct answer. For example, "The answer is: (a)"'
)


def process_docs(dataset):
    """Interleave the 23 tasks round-robin, BBEH-mini questions first.

    The dataset is grouped by task, so a --limit prefix would otherwise cover
    one or two tasks. Here the first 460 docs are exactly BBEH mini (20 per
    task) and any shorter prefix takes the same number from every task.
    """
    by_task = {}
    for row in dataset:
        by_task.setdefault(row["task"], []).append(row)
    queues = [
        sorted(rows, key=lambda r: -int(r["mini"]))
        for _, rows in sorted(by_task.items())
    ]
    ordered = [row for group in zip_longest(*queues) for row in group if row]
    return datasets.Dataset.from_list(ordered)


def doc_to_text(doc) -> str:
    return f"{doc['input']}\n\n{ANSWER_INSTRUCTION}"


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
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
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    # Quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    return fuzzy_match(preprocess_sample(sample), preprocess_reference(reference))


def process_results(doc, results):
    return {"exact_match": float(evaluate_correctness(results[0], doc["target"]))}
