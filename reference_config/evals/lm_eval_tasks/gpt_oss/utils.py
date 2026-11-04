# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""AIME grading as in OpenAI's gpt_oss.evals (gpt_oss/evals/aime_eval.py), the harness of gpt-oss-20b's GPU reference.

lm-eval's own aime25 grader reads only a \\boxed{} answer: a correct integer the model writes in display math
(e.g. "\\[ 504 \\]") scores 0. OpenAI's grader takes the last \\boxed{}/\\framebox{} content (after its last comma),
else the last integer in the reply, and compares integers.
"""

import re
from typing import Dict, List


def extract_boxed_text(text: str) -> str:
    """Verbatim from openai/gpt-oss gpt_oss/evals/aime_eval.py."""
    pattern = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(",")[-1].strip()
    pattern = r"\d+"  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    extracted = extract_boxed_text(results[0])
    try:  # all AIME answers are integers, as in aime_eval.py
        extracted = int(extracted)
    except (ValueError, TypeError):
        extracted = None
    answer_key = next(k for k in doc.keys() if k.lower() == "answer")
    return {"exact_match": int(extracted == int(doc[answer_key]))}
