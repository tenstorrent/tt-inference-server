# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefix-cache figures the bonus rubric scores, stored as data.

Two values are scored: the **decode cache hit rate** and the **reduction in time
to first token** against the matched no-shared-prefix run. Neither was reachable
by a machine before this module. The hit rate was reachable only after
reimplementing the aggregated/disaggregated resolution rule, and the uplift was
computed inside :mod:`report_module.prefix_cache_renderer` while building a
Markdown table -- a display artifact, absent from the report JSON entirely.

Two details here are easy to get wrong and expensive to get wrong:

**Sign.** The renderer's ``ttft_uplift_pct`` is a signed delta,
``(treatment - baseline) / baseline``, so a cache that *helps* produces a
*negative* number -- the usual convention for a latency delta in a table. The
rubric asks for a percentage *reduction*, where higher is better and 0 % means no
uplift. Reading the display field as a score would invert the line completely: a
44 % improvement would read as -44 and score zero, while a cache that made time
to first token 44 % *worse* would read as +44 and score points. The scored field
is therefore named :func:`ttft_reduction_pct` and carries the opposite sign to the
display field, deliberately.

**Zero versus unmeasured.** A reduction of 0 % is a real result -- the cache did
nothing -- and sits exactly at the rubric's qualifying value. An unmeasured
reduction is ``None``. These must never collapse into each other, the same
invariant :mod:`llm_module.derived_metrics` keeps.

Placed in ``report_module`` rather than beside the other derived metrics because
the renderer consumes it, and ``report_module`` importing ``llm_module`` would
close an import cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import Block

logger = logging.getLogger(__name__)

PREFIX_CACHE_KIND = "aiperf_prefix_cache"

#: The scenario every treatment run is compared against: same shape, no shared
#: prefixes. Without it there is no uplift to measure.
BASELINE_SCENARIO = "baseline"

#: Scenarios a Milestone-0 submission must run (RFP Appendix B.3). Only these
#: contribute to the scored summary. ``mooncake_trace`` is optional and is
#: deliberately excluded -- if an optional run could move the score, a Partner
#: would have to weigh whether running it helps or hurts them, which is a
#: perverse thing to make anyone reason about.
REQUIRED_SCENARIOS: Tuple[str, ...] = ("shared_system", "prefix_pool", "multi_turn")

#: Fields this module attaches. Listed so tests and the scorecard agree on names.
SCORED_FIELDS = (
    "decode_prefix_cache_hit_rate",
    "ttft_reduction_pct",
    "baseline_mean_ttft_ms",
)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def baseline_key(record: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    """The shape a treatment run and its baseline must share to be comparable.

    Comparing across different concurrency, arrival pattern or input length would
    measure those differences rather than the cache.
    """
    return (
        record.get("concurrency"),
        record.get("arrival_pattern"),
        record.get("isl_mean"),
    )


def delta_pct(treatment: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """Signed change from baseline, as a percentage. Negative means faster.

    The display convention. :func:`ttft_reduction_pct` is what gets scored.
    """
    t, b = _number(treatment), _number(baseline)
    if t is None or b is None or b == 0:
        return None
    return (t - b) / b * 100.0


def ttft_reduction_pct(
    treatment: Optional[float], baseline: Optional[float]
) -> Optional[float]:
    """How much the cache cut time to first token, as a percentage.

    Positive means the cache helped, which is the direction the rubric scores.
    Exactly the negation of :func:`delta_pct`; kept as its own function so the
    sign is asserted in one place rather than at each call site.
    """
    signed = delta_pct(treatment, baseline)
    return None if signed is None else round(-signed, 4)


def decode_hit_rate(record: Mapping[str, Any]) -> Optional[float]:
    """The hit rate the rubric scores, for either deployment shape.

    A disaggregated deployment reports prefill and decode caches separately and
    only decode is scored -- prefill disproportionately handles misses, so its
    rate is structurally low (RFP G.3.3). An aggregated deployment has a single
    cache, and that cache *is* the decode cache, so the unqualified rate is the
    right value.

    This is the same precedence :func:`llm_module.parsers.aiperf_prefix_cache._compute_sla_checks`
    applies when deciding which hit rate gates the SLA verdict.
    """
    role_rate = _number(record.get("prefix_cache_hit_rate_decode"))
    if role_rate is not None:
        return role_rate
    return _number(record.get("prefix_cache_hit_rate"))


def _is_prefix_cache_block(block: Block) -> bool:
    return block.kind == PREFIX_CACHE_KIND and isinstance(block.data, Mapping)


def _with_data(block: Block, data: Dict[str, Any]) -> Block:
    return Block(
        kind=block.kind,
        id=block.id,
        title=block.title,
        data=data,
        targets=block.targets,
    )


def apply_prefix_cache_uplift(blocks: Sequence[Block]) -> List[Block]:
    """Attach the scored prefix-cache fields to every prefix-cache block.

    Runs after the sweep, not per run: the uplift needs a treatment run and its
    matched baseline, which are separate runs. Mirrors
    :func:`llm_module.derived_metrics.apply_scaling_exponents`.

    Baseline runs and treatment runs with no matching baseline carry the uplift
    fields as ``None``, so the field is always present and its absence never has
    to be distinguished from a genuine zero.
    """
    baselines: Dict[Tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for block in blocks:
        if not _is_prefix_cache_block(block):
            continue
        if block.data.get("scenario") == BASELINE_SCENARIO:
            baselines[baseline_key(block.data)] = block.data

    unmatched: List[str] = []
    out: List[Block] = []
    for block in blocks:
        if not _is_prefix_cache_block(block):
            out.append(block)
            continue

        data = dict(block.data)
        data["decode_prefix_cache_hit_rate"] = decode_hit_rate(data)

        scenario = data.get("scenario")
        base = baselines.get(baseline_key(data))
        if scenario == BASELINE_SCENARIO or base is None:
            data["baseline_mean_ttft_ms"] = None
            data["ttft_reduction_pct"] = None
            if scenario != BASELINE_SCENARIO:
                unmatched.append(f"{scenario}/{data.get('label')}")
        else:
            base_ttft = _number(base.get("mean_ttft_ms"))
            data["baseline_mean_ttft_ms"] = base_ttft
            data["ttft_reduction_pct"] = ttft_reduction_pct(
                data.get("mean_ttft_ms"), base_ttft
            )

        out.append(_with_data(block, data))

    if unmatched:
        logger.warning(
            "No matching %s run for %d prefix-cache run(s): %s. Their time-to-first-"
            "token uplift cannot be measured and will not be scored.",
            BASELINE_SCENARIO,
            len(unmatched),
            ", ".join(sorted(unmatched)),
        )
    if not baselines:
        logger.warning(
            "The prefix-cache sweep contains no %s runs, so no uplift can be "
            "measured at all. RFP Appendix B.3 requires the matched comparison run.",
            BASELINE_SCENARIO,
        )

    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def summarize_prefix_cache_scoring(blocks: Sequence[Block]) -> Dict[str, Any]:
    """Reduce the sweep to the two scalars the bonus rubric scores.

    Averaged across the required scenarios with equal weight. Each probes a
    different reuse pattern and all three are required, so no principle picks one
    over another; taking the worst would let the hardest scenario decide the line,
    and taking the best would reward optimising a single case.

    ``scenarios_missing`` is the part a caller must not ignore. Averaging over
    whichever required scenarios happen to be present would quietly reward
    omitting the weakest one, so a submission missing any required scenario is
    incomplete rather than simply averaged over fewer terms. This function reports
    that; it does not decide what to do about it.
    """
    hit_rates: Dict[str, List[float]] = {}
    reductions: Dict[str, List[float]] = {}

    for block in blocks:
        if not _is_prefix_cache_block(block):
            continue
        scenario = block.data.get("scenario")
        if scenario not in REQUIRED_SCENARIOS:
            continue
        rate = decode_hit_rate(block.data)
        if rate is not None:
            hit_rates.setdefault(scenario, []).append(rate)
        reduction = _number(block.data.get("ttft_reduction_pct"))
        if reduction is not None:
            reductions.setdefault(scenario, []).append(reduction)

    scored = sorted(set(hit_rates) | set(reductions))
    missing = [s for s in REQUIRED_SCENARIOS if s not in scored]

    # Average within a scenario first, then across scenarios, so a scenario run at
    # more operating points than another does not thereby count for more.
    def _across(per_scenario: Mapping[str, List[float]]) -> Optional[float]:
        means = [m for m in (_mean(v) for v in per_scenario.values()) if m is not None]
        return _mean(means)

    return {
        "decode_prefix_cache_hit_rate": _across(hit_rates),
        "ttft_reduction_pct": _across(reductions),
        "scenarios_scored": scored,
        "scenarios_missing": missing,
    }


__all__ = [
    "BASELINE_SCENARIO",
    "PREFIX_CACHE_KIND",
    "REQUIRED_SCENARIOS",
    "SCORED_FIELDS",
    "apply_prefix_cache_uplift",
    "baseline_key",
    "decode_hit_rate",
    "delta_pct",
    "summarize_prefix_cache_scoring",
    "ttft_reduction_pct",
]
