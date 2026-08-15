# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import sys
from types import SimpleNamespace

from llm_module.speed_bench_prompts import (
    TIERS,
    build_exact_prompts,
    exact_chat_content,
    render_ids,
    subset_for_length,
)


class _CharTokenizer:
    """One token per character; chat template wraps content in <u>...</u>."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(i) for i in ids)

    def apply_chat_template(
        self, messages, add_generation_prompt, enable_thinking, tokenize
    ):
        assert not tokenize
        assert add_generation_prompt and enable_thinking
        return f"<u>{messages[0]['content']}</u>"


def _texts_for_tier(subset, tier):
    del subset
    # Long enough to cover any target after concatenation, distinct per tier.
    return [f"{tier} corpus text {index} " * 400 for index in range(4)]


def test_subset_selection_tracks_target_length():
    assert subset_for_length(128) == "throughput_1k"
    assert subset_for_length(2048) == "throughput_2k"
    assert subset_for_length(8192) == "throughput_8k"
    assert subset_for_length(16384) == "throughput_16k"
    assert subset_for_length(131072) == "throughput_32k"


def test_exact_chat_content_hits_target_exactly():
    tokenizer = _CharTokenizer()
    source = tokenizer.encode("lorem ipsum dolor sit amet " * 64)
    for target in (64, 128, 500):
        content = exact_chat_content(tokenizer, source, target)
        assert len(render_ids(tokenizer, content)) == target


def test_build_exact_prompts_cycles_tiers_and_hits_isl():
    tokenizer = _CharTokenizer()
    prompts = build_exact_prompts(
        tokenizer=tokenizer,
        target_isl=256,
        num_prompts=4,
        texts_for_tier=_texts_for_tier,
    )
    assert len(prompts) == 4
    for prompt in prompts:
        assert len(render_ids(tokenizer, prompt)) == 256
    # ordinal 0 and 3 are both low_entropy but start at different rows
    assert prompts[0].startswith(TIERS[0].split("_")[0])
    assert prompts[1] != prompts[0]
    assert prompts[3] != prompts[0]


def test_prompt_file_rows_are_custom_dataset_shape(tmp_path, monkeypatch):
    import llm_module.speed_bench_prompts as module

    monkeypatch.setattr(module, "load_subset_texts", _texts_for_tier)

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=False):
            del model, trust_remote_code
            return _CharTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=_AutoTokenizer),
    )
    out = module.write_speed_bench_prompt_file(
        output_path=tmp_path / "prompts.jsonl",
        model="google/diffusiongemma-26B-A4B-it",
        target_isl=128,
        num_prompts=3,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3
    assert all(set(row) == {"prompt"} for row in rows)
