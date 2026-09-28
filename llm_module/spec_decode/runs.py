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

# The SPEED-Bench throughput subsets; --spec-decode-isls picks from these.
SPEED_BENCH_THROUGHPUT_ISLS: Tuple[str, ...] = ("1k", "2k", "8k", "16k", "32k")

THROUGHPUT_CONCURRENCY_SWEEP: Tuple[int, ...] = (1, 8, 16, 32, 64)

# The 'ci' preset trims the sweep so a regression run stays short: the
# 'coding' qualitative category plus a single throughput ISL at three
# concurrencies — 32k_maxcon-{1,16,64}.
CI_QUALITATIVE_CATEGORIES: Tuple[str, ...] = ("coding",)
CI_THROUGHPUT_ISLS: Tuple[str, ...] = ("32k",)
CI_THROUGHPUT_CONCURRENCIES: Tuple[int, ...] = (1, 16, 64)

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


def _throughput_runs(
    isls: Tuple[str, ...], concurrencies: Tuple[int, ...]
) -> List[SpecDecodeRun]:
    return [
        SpecDecodeRun(
            public_dataset=f"speed_bench_throughput_{isl}",
            max_concurrency=concurrency,
            num_prompts=max(32, 4 * concurrency),
            max_completion_tokens=SPEC_DECODE_MAX_COMPLETION_TOKENS,
        )
        for isl in isls
        for concurrency in concurrencies
    ]


@dataclass(frozen=True)
class _Preset:
    qualitative_categories: Tuple[str, ...]
    throughput_isls: Tuple[str, ...]
    throughput_concurrencies: Tuple[int, ...]

    def runs(self) -> List[SpecDecodeRun]:
        return _qualitative_runs(self.qualitative_categories) + _throughput_runs(
            self.throughput_isls, self.throughput_concurrencies
        )


_PRESETS = {
    "full": _Preset(
        SPEED_BENCH_QUALITATIVE_CATEGORIES,
        SPEED_BENCH_THROUGHPUT_ISLS,
        THROUGHPUT_CONCURRENCY_SWEEP,
    ),
    "ci": _Preset(
        CI_QUALITATIVE_CATEGORIES, CI_THROUGHPUT_ISLS, CI_THROUGHPUT_CONCURRENCIES
    ),
    "throughput": _Preset(
        (), SPEED_BENCH_THROUGHPUT_ISLS, THROUGHPUT_CONCURRENCY_SWEEP
    ),
}

SPEC_DECODE_PRESETS = {name: preset.runs() for name, preset in _PRESETS.items()}
SPEC_DECODE_SWEEP: List[SpecDecodeRun] = SPEC_DECODE_PRESETS["full"]
SPEC_DECODE_CI_SWEEP: List[SpecDecodeRun] = SPEC_DECODE_PRESETS["ci"]
SPEC_DECODE_THROUGHPUT_SWEEP: List[SpecDecodeRun] = SPEC_DECODE_PRESETS["throughput"]


def parse_isls(isls: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse ``--spec-decode-isls`` (e.g. ``"1k,8k"``) into bucket names.

    Returns ``None`` when unset so the preset's ISLs apply. Buckets are the
    SPEED-Bench throughput subsets and come back in canonical order.
    """
    if not isls:
        return None
    selected = {s.strip().lower() for s in isls.split(",") if s.strip()}
    bad = sorted(selected - set(SPEED_BENCH_THROUGHPUT_ISLS))
    if bad or not selected:
        raise ValueError(
            f"Unknown spec-decode ISL bucket(s): {bad or [isls]}. "
            f"Available: {list(SPEED_BENCH_THROUGHPUT_ISLS)}"
        )
    return tuple(isl for isl in SPEED_BENCH_THROUGHPUT_ISLS if isl in selected)


def parse_concurrencies(concurrencies: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse ``--spec-decode-concurrencies`` (e.g. ``"1,8,32"``) into ints.

    Returns ``None`` when unset so the preset's concurrencies apply. Any
    positive integer is accepted; values come back sorted and de-duplicated.
    """
    if not concurrencies:
        return None
    try:
        selected = {int(c) for c in concurrencies.split(",") if c.strip()}
    except ValueError:
        selected = set()
    if not selected or min(selected) < 1:
        raise ValueError(
            f"Invalid spec-decode concurrencies: {concurrencies!r}. "
            "Expected a comma-separated list of positive integers, e.g. '1,8,32'."
        )
    return tuple(sorted(selected))


def build_runs(
    preset: str = "full",
    *,
    isls: Optional[str] = None,
    concurrencies: Optional[str] = None,
) -> List[SpecDecodeRun]:
    """Return the spec-decode sweep for ``preset``.

    ``full`` (default) runs every qualitative category plus the whole
    throughput ISL x concurrency grid. ``ci`` runs only the 'coding'
    qualitative category plus the 32k throughput ISL at concurrency
    1/16/64. ``throughput`` runs the full throughput grid and no
    qualitative categories.

    ``isls`` / ``concurrencies`` (comma-separated, from
    ``--spec-decode-isls`` / ``--spec-decode-concurrencies``) replace the
    preset's throughput ISLs / concurrencies; qualitative runs are
    unaffected.
    """
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown spec-decode preset: {preset}. Available: {sorted(_PRESETS)}"
        )
    base = _PRESETS[preset]
    return _Preset(
        base.qualitative_categories,
        parse_isls(isls) or base.throughput_isls,
        parse_concurrencies(concurrencies) or base.throughput_concurrencies,
    ).runs()


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
    "SPEC_DECODE_THROUGHPUT_SWEEP",
    "SPEED_BENCH_THROUGHPUT_ISLS",
    "SpecDecodeRun",
    "build_runs",
    "parse_concurrencies",
    "parse_isls",
    "summarize_runs",
]
