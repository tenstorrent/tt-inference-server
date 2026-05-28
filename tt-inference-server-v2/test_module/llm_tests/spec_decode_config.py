# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Spec-decode sweep profiles.

Ports v1's ``SPEC_DECODE_PROFILES`` (``benchmarking/benchmark_config.py``)
to the v2 ``SpecDecodeRunConfig`` dataclass. The catalogue:

  * ``spec_bench`` — hemingkx Spec-Bench, 480 prompts, 13 categories
    (writing, roleplay, reasoning, math, coding, extraction, stem,
    humanities, translation, summarization, qa, math_reasoning, rag).
    aiperf only exposes the whole-dataset slug.
  * ``speed_bench_<category>`` — NVIDIA SPEED-Bench Qualitative split,
    ~80 prompts per category. 11 categories enumerated below.
  * ``speed_bench_throughput_{1k,2k,8k,16k,32k}`` — SPEED-Bench
    Throughput split, used to sweep concurrency at fixed input lengths.
"""

from __future__ import annotations

from typing import Dict, List

from llm_module import SpecDecodeRunConfig

SPEED_BENCH_QUALITATIVE_CATEGORIES = (
    "coding",
    "humanities",
    "math",
    "multilingual",
    "qa",
    "rag",
    "reasoning",
    "roleplay",
    "stem",
    "summarization",
    "writing",
)

SPEED_BENCH_THROUGHPUT_ISLS = ("1k", "2k", "8k", "16k", "32k")

THROUGHPUT_CONCURRENCY_SWEEP = (1, 16, 64)

SPEC_DECODE_PROFILES: Dict[str, List[SpecDecodeRunConfig]] = {
    "smoke": [
        SpecDecodeRunConfig(
            public_dataset="speed_bench_coding",
            max_concurrency=1,
            num_prompts=4,
        ),
    ],
    # Full sweep — output length is left natural:
    #   - speed_bench_<category> × 11 categories × conc=1 (every prompt in
    #     the category; aiperf defaults --request-count to dataset size)
    #   - speed_bench_throughput_{1k..32k} × conc{1,16,64}
    "full": [
        SpecDecodeRunConfig(
            public_dataset=f"speed_bench_{category}",
            max_concurrency=1,
        )
        for category in SPEED_BENCH_QUALITATIVE_CATEGORIES
    ]
    + [
        SpecDecodeRunConfig(
            public_dataset=f"speed_bench_throughput_{isl}",
            max_concurrency=concurrency,
            num_prompts=max(32, 4 * concurrency),
        )
        for isl in SPEED_BENCH_THROUGHPUT_ISLS
        for concurrency in THROUGHPUT_CONCURRENCY_SWEEP
    ],
}


__all__ = [
    "SPEC_DECODE_PROFILES",
    "SPEED_BENCH_QUALITATIVE_CATEGORIES",
    "SPEED_BENCH_THROUGHPUT_ISLS",
    "THROUGHPUT_CONCURRENCY_SWEEP",
]
