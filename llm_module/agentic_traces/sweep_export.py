# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Measured agentic-traces runs in the requirements ``agenticSweep`` shape.

A requirements document states its expectations for an agentic workload as an
``agenticSweep``: one camelCase object per concurrency. Emitting what we
measured in that same shape makes the two directly comparable — the customer's
expected point and our observed point line up field for field.

One point per run, so a sweep over several concurrencies yields several points.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# agenticSweep key -> the metric key produced by ``parse_aiperf_output``.
#
# Only measurements appear here. ``goodputPct`` is derived rather than mapped:
# see ``_goodput_pct``.
_SWEEP_FIELD_TO_METRIC: Tuple[Tuple[str, str], ...] = (
    ("ttftMeanMs", "mean_ttft_ms"),
    ("ttftP50Ms", "median_ttft_ms"),
    ("ttftP90Ms", "p90_ttft_ms"),
    ("ttftP95Ms", "p95_ttft_ms"),
    ("tpotMeanMs", "mean_tpot_ms"),
    ("tpotP50Ms", "median_tpot_ms"),
    ("tpotP90Ms", "p90_tpot_ms"),
    ("tpotP95Ms", "p95_tpot_ms"),
    ("e2elMeanMs", "mean_e2el_ms"),
    ("e2elP50Ms", "median_e2el_ms"),
    ("e2elP90Ms", "p90_e2el_ms"),
    ("e2elP95Ms", "p95_e2el_ms"),
    ("reqThroughputRps", "request_throughput"),
    ("totalThroughputTps", "total_token_throughput"),
    ("decodeThroughputTps", "output_token_throughput"),
    ("inputTokensMean", "mean_isl"),
    ("inputTokensP95", "p95_isl"),
    ("outputTokensMean", "mean_osl"),
    ("outputTokensP95", "p95_osl"),
    # Percentiles of a rate, so these read the slow tail off the bottom of the
    # distribution; see the note in ``parse_aiperf_output``.
    ("e2eNormIntvtyP75Tps", "p75_e2e_norm_intvty"),
    ("e2eNormIntvtyP90Tps", "p90_e2e_norm_intvty"),
)

# Measured hit rate first; the theoretical rate is what the trace pool implies
# rather than what the server did, so it is only a fallback when the server
# exposed no prefix-cache metrics.
_CACHE_HIT_METRICS: Tuple[str, ...] = (
    "measured_prefix_cache_hit_pct",
    "theoretical_prefix_cache_hit_pct",
)


def _number(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or None if it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _goodput_pct(metrics: Mapping[str, Any]) -> Optional[float]:
    """Percentage of requests that met every SLO, or None if not graded.

    AIPerf reports goodput as a rate (good requests/sec) while the document
    states it as a share of requests, so the two are reconciled against total
    request throughput.

    A zero rate is reported honestly as 0% -- "no request met the SLOs" is a
    measurement, not a gap. That is only distinguishable from "never graded"
    because the run records the SLO bars it used: without them AIPerf emits no
    goodput at all, and there is nothing to report.
    """
    if not str(metrics.get("goodput_slo") or "").strip():
        return None
    good = _number(metrics.get("goodput"))
    total = _number(metrics.get("request_throughput"))
    if good is None or not total:
        return None
    return round(good / total * 100, 2)


def to_agentic_sweep_point(
    metrics: Mapping[str, Any],
    *,
    concurrency: int,
) -> Dict[str, Any]:
    """Render one measured run as an ``agenticSweep`` point.

    Metrics the run did not produce are left out rather than zero-filled, so a
    missing field reads as "not measured" instead of "measured as zero".
    """
    point: Dict[str, Any] = {"concurrency": int(concurrency)}
    for field, metric in _SWEEP_FIELD_TO_METRIC:
        value = _number(metrics.get(metric))
        if value is not None:
            point[field] = value

    goodput_pct = _goodput_pct(metrics)
    if goodput_pct is not None:
        point["goodputPct"] = goodput_pct

    for metric in _CACHE_HIT_METRICS:
        value = _number(metrics.get(metric))
        if value is not None:
            point["kvCacheHitRatePct"] = value
            break

    return point


def to_agentic_sweep(runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Render measured runs as an ``agenticSweep`` list, ordered by concurrency.

    Each entry of ``runs`` is a per-run payload as written by the driver: the
    parsed metrics plus the run config they were measured at.
    """
    points = [
        to_agentic_sweep_point(run, concurrency=int(run.get("concurrency") or 0))
        for run in runs
    ]
    return sorted(points, key=lambda point: point["concurrency"])


__all__ = ["to_agentic_sweep", "to_agentic_sweep_point"]
