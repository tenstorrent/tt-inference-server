#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import importlib.util
from pathlib import Path

import pytest


def _load_patch_module():
    patch_path = (
        Path(__file__).resolve().parent.parent
        / "evals"
        / "scripts"
        / "lm_eval_streaming_patch"
        / "sitecustomize.py"
    )
    spec = importlib.util.spec_from_file_location("tt_lm_eval_patch", patch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mmlu_pro_extractor_accepts_common_r1_final_answer_formats():
    patch = _load_patch_module()
    doc = {"options": ["a", "b", "c", "d", "e"]}

    assert patch._extract_mmlu_pro_answer("The answer is (C).", doc) == "C"
    assert patch._extract_mmlu_pro_answer("**Answer: D**", doc) == "D"
    assert patch._extract_mmlu_pro_answer("Final Answer: (B)", doc) == "B"
    assert patch._extract_mmlu_pro_answer(r"\boxed{\text{E}}", doc) == "E"
    assert patch._extract_mmlu_pro_answer("The correct option is A.", doc) == "A"


def test_mmlu_pro_extractor_uses_last_valid_final_answer():
    patch = _load_patch_module()
    doc = {"options": ["a", "b", "c", "d"]}

    response = "<think>The answer is (A).</think>\nThe answer is (B).\nFinal Answer: D"

    assert patch._extract_mmlu_pro_answer(response, doc) == "D"


def test_mmlu_pro_extractor_rejects_letters_outside_available_options():
    patch = _load_patch_module()
    doc = {"options": ["a", "b", "c"]}

    assert patch._extract_mmlu_pro_answer("Final answer: J", doc) == "[invalid]"


def test_mmlu_pro_regex_patch_only_changes_known_mmlu_pro_patterns():
    patch = _load_patch_module()

    class FakeRegexFilter:
        def __init__(self, regex_pattern, fallback="[invalid]"):
            self.regex_pattern = regex_pattern
            self.fallback = fallback

        def apply(self, resps, docs):
            return [["ORIGINAL"] for _ in resps]

    patch._patch_mmlu_pro_regex_filter(FakeRegexFilter)

    mmlu_filter = FakeRegexFilter(r"answer is \(?([ABCDEFGHIJ])\)?")
    assert mmlu_filter.apply(
        [["Therefore, **Answer: B**"]],
        [{"options": ["a", "b", "c"]}],
    ) == [["B"]]

    other_filter = FakeRegexFilter(r"#### (\d+)")
    assert other_filter.apply([["#### 42"]], [{}]) == [["ORIGINAL"]]


def test_streaming_consumer_does_not_swallow_keyboard_interrupt():
    patch = _load_patch_module()

    class InterruptedResponse:
        @staticmethod
        def iter_lines(decode_unicode=True):
            yield 'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}'
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        patch._consume_sync_sse_stream(InterruptedResponse())
