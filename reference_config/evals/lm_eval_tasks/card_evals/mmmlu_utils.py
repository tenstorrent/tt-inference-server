# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""MMMLU (openai/MMMLU) as a generative chain-of-thought task.

The pinned fork's openai-mmmlu task scores answer log-likelihoods, which the
chat API cannot return. This version follows openai/simple-evals' MMMLU
protocol instead: the English multiple-choice instruction over the translated
question, and the final answer letter extracted with simple-evals' multilingual
"Answer:" patterns.
"""

import re

QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}"
)

# simple-evals common.MULTILINGUAL_ANSWER_REGEXES / ANSWER_PATTERN_TEMPLATE.
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]
ANSWER_PATTERN_TEMPLATE = (
    r"(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)

# The dataset is ordered by subject; a fixed shuffle makes any --limit prefix a
# mix of subjects instead of the first few alphabetically.
SHUFFLE_SEED = 1234


def process_docs(dataset):
    return dataset.shuffle(seed=SHUFFLE_SEED)


def doc_to_text(doc) -> str:
    return QUERY_TEMPLATE.format(
        question=doc["Question"].strip(),
        a=doc["A"],
        b=doc["B"],
        c=doc["C"],
        d=doc["D"],
    )


def normalize_response(response: str) -> str:
    """simple-evals common.normalize_response: drop markdown/LaTeX wrappers."""
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(extracted_answer: str) -> str:
    """simple-evals common.normalize_extracted_answer: map the Arabic, Bengali
    and full-width letters used for A-D onto A-D."""
    return (
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def extract_answer(response: str):
    text = normalize_response(response)
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        match = re.search(ANSWER_PATTERN_TEMPLATE.format(answer_regex), text)
        if match:
            return normalize_extracted_answer(match.group(1))
    return None


def process_results(doc, results):
    extracted = extract_answer(results[0])
    return {"exact_match": float(extracted == doc["Answer"].strip())}
