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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

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
)

# Emitted after the cache-hit rate, following the document's own field order so
# a measured point and an expected one diff cleanly line for line.
#
# Percentiles of a rate, so these read the slow tail off the bottom of the
# distribution; see the note in ``parse_aiperf_output``.
_INTVTY_FIELD_TO_METRIC: Tuple[Tuple[str, str], ...] = (
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

    for metric in _CACHE_HIT_METRICS:
        value = _number(metrics.get(metric))
        if value is not None:
            point["kvCacheHitRatePct"] = value
            break

    for field, metric in _INTVTY_FIELD_TO_METRIC:
        value = _number(metrics.get(metric))
        if value is not None:
            point[field] = value

    goodput_pct = _goodput_pct(metrics)
    if goodput_pct is not None:
        point["goodputPct"] = goodput_pct

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


# --- grading against a document's expected sweep -----------------------------

# Every field a point can carry, in the document's own emission order. Used to
# pick an expected point back out of a flattened report record.
POINT_FIELDS: Tuple[str, ...] = (
    tuple(field for field, _metric in _SWEEP_FIELD_TO_METRIC)
    + ("kvCacheHitRatePct",)
    + tuple(field for field, _metric in _INTVTY_FIELD_TO_METRIC)
    + ("goodputPct",)
)

# The token-shape four describe the trace mix being replayed, not the server
# under test: the reference curve and this run both replay the same traces, so
# a mismatch says the replays diverged (already surfaced as OSL mismatch in Run
# Health), not that the server is slow. They are shown but never graded.
UNGRADED_POINT_FIELDS: Tuple[str, ...] = (
    "inputTokensMean",
    "inputTokensP95",
    "outputTokensMean",
    "outputTokensP95",
)

# Direction by family: latencies gate at or below the target, rates at or
# above. Exact comparison, tolerance 0, matching how the requirements document
# grades benchmark targets (its comparators are gte/lte).
_LOWER_IS_BETTER_FIELDS = frozenset(
    field for field in POINT_FIELDS if field.startswith(("ttft", "tpot", "e2el"))
)
_HIGHER_IS_BETTER_FIELDS = frozenset(POINT_FIELDS) - _LOWER_IS_BETTER_FIELDS


@dataclass(frozen=True)
class MetricVerdict:
    """One graded field: target, what was measured, and the verdict."""

    field: str
    target: float
    measured: Optional[float]  # None: the run never produced this metric
    lower_is_better: bool

    @property
    def passed(self) -> Optional[bool]:
        """True/False when graded, None when there is no measurement to grade."""
        if self.measured is None:
            return None
        if self.lower_is_better:
            return self.measured <= self.target
        return self.measured >= self.target


@dataclass(frozen=True)
class PointVerdict:
    """Every graded field at one concurrency, plus the point's own verdict."""

    concurrency: int
    verdicts: Tuple[MetricVerdict, ...]

    @property
    def graded(self) -> int:
        return sum(1 for v in self.verdicts if v.passed is not None)

    @property
    def met(self) -> int:
        return sum(1 for v in self.verdicts if v.passed is True)

    @property
    def passed(self) -> bool:
        """A point passes when every graded field does."""
        return self.graded > 0 and self.met == self.graded


def grade_sweep_point(
    measured: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> PointVerdict:
    """Grade one measured point against its expected counterpart.

    Only fields the document states are graded, in document order; a field the
    run did not measure is reported as ungraded rather than failed, so a
    partial export reads as a gap in the measurement, not a regression.
    """
    concurrency = int(expected.get("concurrency") or measured.get("concurrency") or 0)
    verdicts: List[MetricVerdict] = []
    for field in POINT_FIELDS:
        if field in UNGRADED_POINT_FIELDS:
            continue
        target = _number(expected.get(field))
        if target is None:
            continue
        verdicts.append(
            MetricVerdict(
                field=field,
                target=target,
                measured=_number(measured.get(field)),
                lower_is_better=field in _LOWER_IS_BETTER_FIELDS,
            )
        )
    return PointVerdict(concurrency=concurrency, verdicts=tuple(verdicts))


def grade_agentic_sweep(
    measured_points: Sequence[Mapping[str, Any]],
    expected_points: Sequence[Mapping[str, Any]],
) -> Tuple[List[PointVerdict], List[int]]:
    """Grade a measured sweep against the document's expected points.

    Pairing is by concurrency. Returns the per-point verdicts plus the
    concurrencies the document expected but the sweep never measured, so a
    truncated sweep is called out instead of silently scoring a subset.
    """
    measured_by_concurrency = {
        int(point.get("concurrency") or 0): point for point in measured_points
    }
    verdicts: List[PointVerdict] = []
    missing: List[int] = []
    for expected in expected_points:
        concurrency = int(expected.get("concurrency") or 0)
        measured = measured_by_concurrency.get(concurrency)
        if measured is None:
            missing.append(concurrency)
            continue
        verdicts.append(grade_sweep_point(measured, expected))
    return verdicts, missing


def expected_sweep_from_record(record: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The document's full expected sweep attached to a report record, if any.

    Every run carries the whole sweep, not just its own point, so the report
    can call out the concurrencies that were expected but never measured -- a
    truncated sweep must not read as a complete one that simply scored less.
    A list passes the report generator's block merge untouched (only mappings
    are flattened), so the field survives in both the live and the persisted
    shape.
    """
    sweep = record.get("expected_sweep")
    if isinstance(sweep, list) and sweep:
        return [dict(point) for point in sweep if isinstance(point, Mapping)] or None
    return None


def write_agentic_sweep(
    runs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    filename: str = "agentic_sweep.json",
) -> Path:
    """Write ``runs`` as an ``agenticSweep`` JSON file and return its path.

    Wrapped in the document's own ``{"agenticSweep": [...]}`` object so the file
    can be pasted straight into a requirements document beside the expected
    sweep. Always a list, even for a single point, so a file written mid-sweep
    has the same shape as the final one and consumers need no special case.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"agenticSweep": to_agentic_sweep(runs)}, handle, indent=2)
    logger.info("Agentic sweep written to: %s", path)
    return path


__all__ = [
    "POINT_FIELDS",
    "UNGRADED_POINT_FIELDS",
    "MetricVerdict",
    "PointVerdict",
    "expected_sweep_from_record",
    "grade_agentic_sweep",
    "grade_sweep_point",
    "to_agentic_sweep",
    "to_agentic_sweep_point",
    "write_agentic_sweep",
]
