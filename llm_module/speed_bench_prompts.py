# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Exact-length SPEED-Bench prompts for the block-granular benchmark sweep.

The generic sweep benchmarks with ``--dataset-name random``: token-salad
prompts whose denoised canvases never reach the entropy halt, so every block
runs the full 48-step cap and the table reports the worst case rather than
serving behaviour on language. For DiffusionGemma the sweep instead sends real
nvidia/SPEED-Bench text, cut to EXACTLY the sweep point's ISL after the
serving chat template is applied — the same construction the standalone
SPEED-Bench client uses — via ``vllm bench serve --dataset-name custom``.

Rows come from the throughput subsets (1k/2k/8k/16k/32k, whichever is nearest
to the target length) and cycle the three entropy tiers so a sweep point's
requests are not all the same prompt. Short targets trim one row; long targets
concatenate rows before trimming, so 64K/128K sweep points stay real text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Sequence

logger = logging.getLogger(__name__)

SPEED_BENCH_DATASET = "nvidia/SPEED-Bench"
TIERS = ("low_entropy", "mixed", "high_entropy")
_SEPARATOR_TEXT = (
    "\n\n--- Continue with the following related SPEED-Bench context ---\n\n"
)


def subset_for_length(length: int) -> str:
    if length <= 1024:
        return "throughput_1k"
    if length <= 2048:
        return "throughput_2k"
    if length <= 8192:
        return "throughput_8k"
    if length <= 16384:
        return "throughput_16k"
    return "throughput_32k"


def load_subset_texts(subset: str, tier: str) -> List[str]:
    """First conversation turn of every ``tier`` row in ``subset`` (network)."""
    from datasets import load_dataset

    dataset = load_dataset(SPEED_BENCH_DATASET, name=subset, split="test")
    texts = [row["turns"][0] for row in dataset if row["category"] == tier]
    if not texts:
        raise RuntimeError(
            f"{SPEED_BENCH_DATASET}:{subset} has no rows for category={tier}"
        )
    return texts


def render_ids(tokenizer, content: str) -> List[int]:
    """Token ids of ``content`` as the serving chat template renders it."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def _source_ids(
    tokenizer, texts: Sequence[str], *, ordinal: int, minimum_tokens: int
) -> List[int]:
    separator = tokenizer.encode(_SEPARATOR_TEXT, add_special_tokens=False)
    ids: List[int] = []
    index = ordinal % len(texts)
    while len(ids) < minimum_tokens:
        if ids:
            ids.extend(separator)
        ids.extend(tokenizer.encode(texts[index], add_special_tokens=False))
        index = (index + 1) % len(texts)
    return ids


def exact_chat_content(tokenizer, source_ids: Sequence[int], target_length: int) -> str:
    """Content whose chat-template rendering is exactly ``target_length`` tokens."""
    empty_overhead = len(render_ids(tokenizer, ""))
    source_count = max(1, target_length - empty_overhead)

    def candidate(count: int) -> str:
        count = max(1, min(count, len(source_ids)))
        if count == len(source_ids):
            selected = list(source_ids)
        else:
            head = (count * 2) // 3
            tail = count - head
            selected = list(source_ids[:head]) + list(source_ids[-tail:])
        return tokenizer.decode(selected, skip_special_tokens=True)

    seen: set[int] = set()
    for _ in range(32):
        if source_count in seen:
            break
        seen.add(source_count)
        content = candidate(source_count)
        actual = len(render_ids(tokenizer, content))
        if actual == target_length:
            return content
        source_count += target_length - actual

    # SentencePiece boundary effects can make the direct correction oscillate.
    # Pick the closest prompt below the target, then close the small residual
    # gap with ordinary natural-language tokens.
    center = source_count
    bases: List[tuple[int, str]] = []
    for count in range(max(1, center - 128), min(len(source_ids), center + 128) + 1):
        content = candidate(count)
        actual = len(render_ids(tokenizer, content))
        if actual == target_length:
            return content
        if actual < target_length:
            bases.append((actual, content))
    fillers = (" and", " the", " context", " information", " detail", ".", "\n")
    for actual, content in sorted(bases, reverse=True)[:16]:
        while actual < target_length:
            options = []
            for filler in fillers:
                filled = content + filler
                filled_length = len(render_ids(tokenizer, filled))
                if actual < filled_length <= target_length:
                    options.append((filled_length, filled))
            if not options:
                break
            actual, content = max(options, key=lambda item: item[0])
            if actual == target_length:
                return content
    raise RuntimeError(
        f"could not construct an exact {target_length}-token chat prompt"
    )


def build_exact_prompts(
    *,
    tokenizer,
    target_isl: int,
    num_prompts: int,
    texts_for_tier: Callable[[str, str], Sequence[str]] | None = None,
) -> List[str]:
    """``num_prompts`` exact-ISL prompts cycling the SPEED-Bench entropy tiers."""
    if texts_for_tier is None:
        texts_for_tier = load_subset_texts
    subset = subset_for_length(int(target_isl))
    prompts: List[str] = []
    tier_texts: dict[str, Sequence[str]] = {}
    for ordinal in range(int(num_prompts)):
        tier = TIERS[ordinal % len(TIERS)]
        if tier not in tier_texts:
            tier_texts[tier] = texts_for_tier(subset, tier)
        source = _source_ids(
            tokenizer,
            tier_texts[tier],
            ordinal=ordinal,
            # Enough source to trim from; template overhead is well under 64.
            minimum_tokens=int(target_isl) + 64,
        )
        prompts.append(exact_chat_content(tokenizer, source, int(target_isl)))
    return prompts


def write_speed_bench_prompt_file(
    *,
    output_path: Path,
    model: str,
    target_isl: int,
    num_prompts: int,
    trust_remote_code: bool = False,
) -> Path:
    """Write a ``vllm bench serve --dataset-name custom`` JSONL of exact-ISL prompts."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    prompts = build_exact_prompts(
        tokenizer=tokenizer, target_isl=target_isl, num_prompts=num_prompts
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt}) + "\n")
    logger.info(
        "wrote %d exact-%d-token SPEED-Bench prompts to %s",
        len(prompts),
        target_isl,
        output_path,
    )
    return output_path
