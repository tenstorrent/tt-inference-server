# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Sweep definitions for the speculative-decoding benchmark.

Self-contained port of v1 ``benchmarking/spec_decode_common.SpecDecodeRunSpec``
plus the ``SPEC_DECODE_SWEEP`` from ``reference_config/benchmarking/benchmark_config.py``. The
matching AIPerf driver lives in ``llm_module.drivers.aiperf_spec_decode``; the
orchestrator that ties them together is
:mod:`test_module.llm_tests.spec_decode_tests`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# SPEED-Bench qualitative split: ~80 prompts per category.
SPEED_BENCH_QUALITATIVE_CATEGORIES: Tuple[str, ...] = (
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

SPEED_BENCH_THROUGHPUT_ISLS: Tuple[str, ...] = ("1k", "2k", "8k", "16k", "32k")

THROUGHPUT_CONCURRENCY_SWEEP: Tuple[int, ...] = (1, 16, 64)

# The 'ci' preset trims the sweep so a regression run stays short: the
# 'coding' qualitative category plus a single throughput ISL across the full
# concurrency sweep — 32k_maxcon-{1,16,64}.
CI_QUALITATIVE_CATEGORIES: Tuple[str, ...] = ("coding",)
CI_THROUGHPUT_ISLS: Tuple[str, ...] = ("32k",)

# Every SPEED-Bench qualitative category holds exactly 80 prompts. aiperf does
# max(10, concurrency*2) == 10 for conc=1, so the count must be passed
# explicitly to consume the whole category. The default SHUFFLE sampler draws
# without replacement, so a count equal to the category size sends each prompt
# exactly once.
SPEED_BENCH_QUALITATIVE_NUM_PROMPTS = 80

# Cap output tokens on the throughput sweep so a handful of long-decoding
# prompts can't blow up the runtime. Injected as
# ``--extra-inputs max_completion_tokens:<N>`` (a ceiling, not a fixed length).
SPEC_DECODE_MAX_COMPLETION_TOKENS = 8192


@dataclass(frozen=True)
class SpecDecodeRun:
    """One sweep point of the speculative-decoding benchmark.

    ``output_len`` set forces exactly that many output tokens per request
    (``ignore_eos:true``); unset lets the model decode to its natural EOS.
    ``max_completion_tokens`` is an upper bound that still allows early stop
    at EOS — a wall-clock guard rail, not a workload dimension.
    """

    public_dataset: str
    max_concurrency: int
    num_prompts: Optional[int] = None
    output_len: Optional[int] = None
    max_completion_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.public_dataset:
            raise ValueError("public_dataset is required")

    @property
    def slug(self) -> str:
        """Short identifier for use in result filenames.

        ``osl-<N>`` is included only when ``output_len`` is set; omitting it
        signals that the run let the model decode to its natural EOS.
        ``n-<N>`` is included only when ``num_prompts`` is set; omitting it
        signals the run consumed every prompt in the public dataset.
        ``max_completion_tokens`` is intentionally left out: it is a
        wall-clock guard rail, not a workload dimension.
        """
        parts = [self.public_dataset]
        if self.output_len is not None:
            parts.append(f"osl-{self.output_len}")
        parts.append(f"maxcon-{self.max_concurrency}")
        if self.num_prompts is not None:
            parts.append(f"n-{self.num_prompts}")
        return "_".join(parts)


def _qualitative_runs(categories: Tuple[str, ...]) -> List[SpecDecodeRun]:
    return [
        SpecDecodeRun(
            public_dataset=f"speed_bench_{category}",
            max_concurrency=1,
            num_prompts=SPEED_BENCH_QUALITATIVE_NUM_PROMPTS,  # whole category
        )
        for category in categories
    ]


def _throughput_runs(isls: Tuple[str, ...]) -> List[SpecDecodeRun]:
    return [
        SpecDecodeRun(
            public_dataset=f"speed_bench_throughput_{isl}",
            max_concurrency=concurrency,
            num_prompts=max(32, 4 * concurrency),
            max_completion_tokens=SPEC_DECODE_MAX_COMPLETION_TOKENS,
        )
        for isl in isls
        for concurrency in THROUGHPUT_CONCURRENCY_SWEEP
    ]


SPEC_DECODE_SWEEP: List[SpecDecodeRun] = _qualitative_runs(
    SPEED_BENCH_QUALITATIVE_CATEGORIES
) + _throughput_runs(SPEED_BENCH_THROUGHPUT_ISLS)

SPEC_DECODE_CI_SWEEP: List[SpecDecodeRun] = _qualitative_runs(
    CI_QUALITATIVE_CATEGORIES
) + _throughput_runs(CI_THROUGHPUT_ISLS)

SPEC_DECODE_PRESETS = {
    "full": SPEC_DECODE_SWEEP,
    "ci": SPEC_DECODE_CI_SWEEP,
}


def build_runs(preset: str = "full") -> List[SpecDecodeRun]:
    """Return the spec-decode sweep for ``preset``.

    ``full`` (default) runs every qualitative category plus the whole
    throughput ISL x concurrency grid. ``ci`` runs only the 'coding'
    qualitative category plus the 32k throughput ISL across the
    concurrency sweep (32k_maxcon-{1,16,64}).
    """
    if preset not in SPEC_DECODE_PRESETS:
        raise ValueError(
            f"Unknown spec-decode preset: {preset}. "
            f"Available: {sorted(SPEC_DECODE_PRESETS)}"
        )
    return list(SPEC_DECODE_PRESETS[preset])


def summarize_runs(runs: List[SpecDecodeRun]) -> str:
    """One-line-per-run plan summary for the orchestrator log."""
    lines = [f"Spec-decode sweep plan ({len(runs)} run(s)):"]
    for run in runs:
        lines.append(
            f"  - {run.public_dataset}: concurrency={run.max_concurrency}"
            f" num_prompts={run.num_prompts}"
            f" output_len={run.output_len}"
            f" max_completion_tokens={run.max_completion_tokens}"
        )
    return "\n".join(lines)


__all__ = [
    "SPEC_DECODE_CI_SWEEP",
    "SPEC_DECODE_PRESETS",
    "SPEC_DECODE_SWEEP",
    "SpecDecodeRun",
    "build_runs",
    "summarize_runs",
]
