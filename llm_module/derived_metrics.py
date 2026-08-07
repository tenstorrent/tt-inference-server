# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefill quality metrics derived from the benchmark record.

There is no direct prefill measurement anywhere in the stack. ``tt-metal`` has a
``prefill_t/s`` field but only on the demo path, and the normalised serving
record has nothing equivalent. (``WorkflowType.PREFILL_DECODE`` is not a prefill
benchmark either — it serves a mock stack for prefill/decode disaggregation
smoke tests.)

So prefill quality is derived from time to first token across the sweep:

===================== ==================================================
prefill throughput    ``input_sequence_length / mean_ttft_ms``, tokens/s
ttft tail ratio       ``p99_ttft / p50_ttft``
ttft scaling exponent fitted ``b`` in ``log(p50_ttft) = a + b*log(isl)``
===================== ==================================================

The first two are per-point and attach during the run. The exponent needs several
points at one concurrency, so it is fitted after the sweep completes.

Every value is ``None`` when it cannot be computed. Never 0 — for a latency-derived
figure 0 is the *best* possible result, so a missing value coerced to 0 would read
as perfect rather than as unmeasured.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from report_module.schema import Block

logger = logging.getLogger(__name__)

# An exponent fitted through two points is just a line through two points: it
# always fits perfectly and says nothing about curvature, which is the whole
# reason the metric exists.
MIN_POINTS_FOR_FIT = 3

DERIVED_FIELDS = (
    "prefill_throughput_tok_s",
    "ttft_tail_ratio",
    "ttft_scaling_exponent",
)


def _positive(value: Any) -> Optional[float]:
    """Return value as a float when it is a usable positive number, else None.

    Zero and negatives are rejected: they are divide-by-zero hazards here, and a
    zero latency is not a real measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def prefill_throughput(record: Mapping[str, Any]) -> Optional[float]:
    """Input tokens processed per second, from time to first token.

    Meaning depends on concurrency, and both readings are wanted:

    * At concurrency 1 this is isolated prefill compute rate, with no queuing.
    * Above that it is the prefill rate each request actually receives under
      load, which also reflects scheduling and batching efficiency.
    """
    isl = _positive(record.get("input_sequence_length"))
    ttft_ms = _positive(record.get("mean_ttft_ms"))
    if isl is None or ttft_ms is None:
        return None
    return round(isl / (ttft_ms / 1000.0), 4)


def ttft_tail_ratio(record: Mapping[str, Any]) -> Optional[float]:
    """How far the slow tail sits above the typical request.

    Near 1.0 means the slowest requests behave much like the median. Scale-free,
    so it is comparable across operating points in a way raw latency is not.
    """
    p99 = _positive(record.get("p99_ttft"))
    p50 = _positive(record.get("p50_ttft"))
    if p99 is None or p50 is None:
        return None
    return round(p99 / p50, 4)


def fit_scaling_exponent(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Least-squares slope of log(ttft) against log(isl).

    ``points`` is ``(input_sequence_length, ttft_ms)`` pairs. The slope is how
    steeply prefill cost grows with input length: near 1.0 means proportional,
    near 2.0 means quadratic — the signature of attention cost dominating.

    Returns None when there are too few distinct input lengths to fit, or when
    every point shares one input length (a vertical fit has no slope).
    """
    usable = [
        (math.log(isl), math.log(ttft))
        for isl, ttft in points
        if _positive(isl) is not None and _positive(ttft) is not None
    ]
    if len(usable) < MIN_POINTS_FOR_FIT:
        return None

    n = len(usable)
    mean_x = sum(x for x, _ in usable) / n
    mean_y = sum(y for _, y in usable) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in usable)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in usable)
    return round(numerator / denominator, 4)


def apply_derived_metrics(block: Block) -> Block:
    """Return ``block`` with the per-point derived metrics attached.

    Mirrors :func:`llm_module.target_checks.apply_target_checks`: benchmark
    blocks are enriched, everything else passes through untouched.
    """
    from .parsers.base import BENCHMARKS_KIND

    if block.kind != BENCHMARKS_KIND or not isinstance(block.data, Mapping):
        return block

    data = dict(block.data)
    data["prefill_throughput_tok_s"] = prefill_throughput(data)
    data["ttft_tail_ratio"] = ttft_tail_ratio(data)
    return Block(
        kind=block.kind,
        id=block.id,
        title=block.title,
        data=data,
        targets=block.targets,
    )


def apply_scaling_exponents(blocks: List[Block]) -> List[Block]:
    """Fit one exponent per concurrency level and attach it to each block.

    Fitted **separately per concurrency**, not pooled across the whole sweep. A
    pooled fit would average away exactly what the metric is for: a system can
    scale cleanly when idle and degrade under load, and one combined slope hides
    that entirely.
    """
    from .parsers.base import BENCHMARKS_KIND

    by_concurrency: Dict[Any, List[Tuple[float, float]]] = {}
    for block in blocks:
        if block.kind != BENCHMARKS_KIND or not isinstance(block.data, Mapping):
            continue
        concurrency = block.data.get("concurrency")
        isl = block.data.get("input_sequence_length")
        ttft = block.data.get("p50_ttft")
        if concurrency is None:
            continue
        by_concurrency.setdefault(concurrency, []).append((isl, ttft))

    exponents: Dict[Any, Optional[float]] = {}
    for concurrency, points in by_concurrency.items():
        distinct_isls = {isl for isl, _ in points if _positive(isl) is not None}
        exponent = fit_scaling_exponent(points)
        if exponent is None:
            logger.warning(
                "Cannot fit a time-to-first-token scaling exponent at concurrency "
                "%s: %d usable point(s) across %d distinct input length(s), need "
                "at least %d. This sweep cannot grade prefill scaling at this "
                "concurrency level.",
                concurrency,
                len(points),
                len(distinct_isls),
                MIN_POINTS_FOR_FIT,
            )
        exponents[concurrency] = exponent

    out: List[Block] = []
    for block in blocks:
        if block.kind != BENCHMARKS_KIND or not isinstance(block.data, Mapping):
            out.append(block)
            continue
        data = dict(block.data)
        data["ttft_scaling_exponent"] = exponents.get(data.get("concurrency"))
        out.append(
            Block(
                kind=block.kind,
                id=block.id,
                title=block.title,
                data=data,
                targets=block.targets,
            )
        )
    return out
