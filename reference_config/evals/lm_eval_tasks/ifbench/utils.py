# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""IFBench scoring with AllenAI's verifier registry.

Mirrors IFBench evaluation_lib.test_instruction_following_{strict,loose}
(allenai/IFBench@1c40f0c): kwargs with None values are dropped, an instruction
whose args include "prompt" is rebuilt with the prompt, and loose scoring tries
the same eight response variants.
"""

from typing import Dict, List

import nltk

# Importing ifbench calls nltk.download() for any resource nltk.data.find() misses
# (including the legacy pickle "punkt" it never uses). The runner proxy refuses
# NLTK's downloader, and scripts/setup_nltk_data.py already staged every resource
# the verifiers read, so skip the network entirely during that import.
_nltk_download = nltk.download
nltk.download = lambda *args, **kwargs: False
try:
    from ifbench import instructions_registry
finally:
    nltk.download = _nltk_download


def _visible(response: str) -> str:
    # The server's reasoning parser already strips thinking; drop any leftover
    # "...</think>" prefix so a stray tag never counts against a constraint.
    response = response or ""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1].lstrip()
    return response


def _loose_variants(response: str) -> List[str]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def _follows(doc: dict, candidates: List[str]) -> List[bool]:
    results = []
    for index, instruction_id in enumerate(doc["instruction_id_list"]):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](
            instruction_id
        )
        kwargs = {k: v for k, v in (doc["kwargs"][index] or {}).items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=doc["prompt"])
        results.append(
            any(c.strip() and instruction.check_following(c) for c in candidates)
        )
    return results


def process_results(doc: dict, results: List[str]) -> Dict[str, object]:
    response = _visible(results[0])
    strict = _follows(doc, [response])
    loose = _follows(doc, _loose_variants(response))
    return {
        "prompt_level_strict_acc": all(strict),
        "inst_level_strict_acc": strict,
        "prompt_level_loose_acc": all(loose),
        "inst_level_loose_acc": loose,
    }


def agg_inst_level_acc(items) -> float:
    flat = [x for item in items for x in item]
    return sum(flat) / len(flat)
