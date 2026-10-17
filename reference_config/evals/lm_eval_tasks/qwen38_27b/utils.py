# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Task helpers for the Qwen/Qwen3.8-27B model-card evals the pinned lm-eval fork does
not ship: IFBench (allenai), HLE text-only, and LiveCodeBench v6 (release-v6 problems).
Ported from lm_eval_tasks/gemma4_31b/utils.py; HLE is judged offline (hle_judge.py)."""

import os
import re
from typing import Dict, List, Optional

from lm_eval.tasks.livecodebench import utils as _lcb


def _strip_thinking(response: str) -> str:
    """Drop an inline <think>...</think> block if the server did not split it out."""
    return re.sub(r".*?</think>", "", response or "", flags=re.DOTALL).strip()


# ----------------------------------------------------------------------------------
# IFBench (allenai/IFBench): 300 prompts, 58 OOD constraints, the official verifiers
# of the `ifbench` package. Strict/loose follow evaluation_lib.py of allenai/IFBench;
# the paper reports prompt-level loose accuracy.
# ----------------------------------------------------------------------------------
def _ifbench_follow(doc: dict, responses: List[str]) -> List[bool]:
    from ifbench import instructions_registry

    following = []
    for index, instruction_id in enumerate(doc["instruction_id_list"]):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        kwargs = {k: v for k, v in (doc["kwargs"][index] or {}).items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=doc["prompt"])
        following.append(any(r.strip() and instruction.check_following(r) for r in responses))
    return following


def _ifbench_loose_variants(response: str) -> List[str]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    variants = [response, remove_first, remove_last, remove_both]
    return variants + [v.replace("*", "") for v in variants]


def ifbench_doc_to_text(doc) -> str:
    return doc["prompt"]


def process_results_ifbench(doc: dict, results: List[str]) -> Dict[str, object]:
    response = _strip_thinking(results[0] if results else "")
    strict = _ifbench_follow(doc, [response])
    loose = _ifbench_follow(doc, _ifbench_loose_variants(response))
    return {
        "prompt_level_strict_acc": int(all(strict)),
        "inst_level_strict_acc": strict,
        "prompt_level_loose_acc": int(all(loose)),
        "inst_level_loose_acc": loose,
    }


def agg_inst_level_acc(items) -> float:
    flat = [x for sub in items for x in sub]
    return sum(flat) / len(flat)


# ----------------------------------------------------------------------------------
# HLE, text-only (no image). exact_match is deterministic; the official LLM-judge
# verdict is produced afterwards from the logged samples by hle_judge.py.
# ----------------------------------------------------------------------------------
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


def process_hle_text(dataset):
    return dataset.filter(lambda doc: not (doc.get("image") or "").strip())


def hle_doc_to_text(doc) -> str:
    instruction = (
        HLE_MC_INSTRUCTION if doc.get("answer_type") == "multipleChoice" else HLE_EXACT_INSTRUCTION
    )
    return f"{instruction}\n\n{doc['question']}"


_HLE_ANSWER_LINE = re.compile(
    r"(?im)^\s*\**\s*(?:exact\s+)?answer\s*\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$"
)
_HLE_BOXED = re.compile(r"\\boxed\{\s*(.+?)\s*\}\s*$")


def hle_extract_answer(response: str) -> Optional[str]:
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


def process_results_hle(doc: dict, results: List[str]) -> Dict[str, int]:
    return {"exact_match": hle_deterministic_match(doc, results[0] if results else "")}


# ----------------------------------------------------------------------------------
# LiveCodeBench v6: the 175 problems of release v6 (test6.jsonl). The card's window
# (contest_date >= 2025-02-01, 131 problems) is a subset; both are reported from the
# same per-problem samples. LCB_START_DATE / LCB_END_DATE narrow the set if needed.
# ----------------------------------------------------------------------------------
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
    return _lcb.process_results(doc, [_strip_thinking(r) for r in results])
