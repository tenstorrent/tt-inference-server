# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Aggregate benchmark Blocks across multiple runs into summary statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from report_module.schema import Block, ReportSchema

BENCHMARK_KIND = "benchmarks"

_TARGET_CHECKS_KEY = "target_checks"
_CHECK_SUFFIX = "_check"


@dataclass(frozen=True)
class MetricStats:
    """Distribution of one metric across N runs."""

    n: int
    mean: float
    median: float
    stdev: float
    minimum: float
    maximum: float
    p50: float
    p90: float
    p99: float
    cov: float


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Aggregated statistics for one logical benchmark across N runs.

    Identity is the tuple that is stable across repeated runs of the same
    benchmark: ``(model, device, task_type, title, block_id)``.
    """

    model: str
    device: str
    task_type: str
    title: str
    block_id: Optional[str]
    run_count: int
    metrics: Dict[str, MetricStats]


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    fraction = rank - low
    return float(
        sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction
    )


def compute_metric_stats(values: Sequence[float]) -> MetricStats:
    if not values:
        raise ValueError("compute_metric_stats requires at least one value")
    numeric = [float(v) for v in values]
    n = len(numeric)
    mean = statistics.fmean(numeric)
    stdev = statistics.stdev(numeric) if n >= 2 else 0.0
    ordered = sorted(numeric)
    return MetricStats(
        n=n,
        mean=mean,
        median=statistics.median(numeric),
        stdev=stdev,
        minimum=ordered[0],
        maximum=ordered[-1],
        p50=_percentile(ordered, 50.0),
        p90=_percentile(ordered, 90.0),
        p99=_percentile(ordered, 99.0),
        cov=(stdev / mean) if mean else 0.0,
    )


def _flatten_numeric(data: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in data.items():
        if key == _TARGET_CHECKS_KEY or key.endswith(_CHECK_SUFFIX):
            continue
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_numeric(value, prefix=f"{path}."))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            out[path] = float(value)
    return out


def _extract_benchmark_metrics(block: Block) -> Dict[str, float]:
    if not isinstance(block.data, Mapping):
        return {}
    return _flatten_numeric(block.data)


def _block_identity(
    block: Block, model: str, device: str
) -> tuple[str, str, str, str, Optional[str]]:
    return (model, device, block.task_type or "", block.title or "", block.id)


def aggregate_benchmark_runs(
    schemas: Sequence[ReportSchema],
) -> List[BenchmarkAggregate]:
    values_by_group: Dict[tuple, Dict[str, List[float]]] = {}
    run_counts: Dict[tuple, int] = {}
    order: List[tuple] = []

    for schema in schemas:
        model = str(schema.metadata.get("model_name", ""))
        device = str(schema.metadata.get("device", ""))
        for block in schema.sections:
            if block.kind != BENCHMARK_KIND:
                continue
            identity = _block_identity(block, model, device)
            if identity not in values_by_group:
                values_by_group[identity] = {}
                run_counts[identity] = 0
                order.append(identity)
            run_counts[identity] += 1
            metric_values = values_by_group[identity]
            for metric, value in _extract_benchmark_metrics(block).items():
                metric_values.setdefault(metric, []).append(value)

    aggregates: List[BenchmarkAggregate] = []
    for identity in order:
        model, device, task_type, title, block_id = identity
        metrics = {
            metric: compute_metric_stats(values)
            for metric, values in values_by_group[identity].items()
        }
        aggregates.append(
            BenchmarkAggregate(
                model=model,
                device=device,
                task_type=task_type,
                title=title,
                block_id=block_id,
                run_count=run_counts[identity],
                metrics=metrics,
            )
        )
    return aggregates


__all__ = [
    "BENCHMARK_KIND",
    "BenchmarkAggregate",
    "MetricStats",
    "aggregate_benchmark_runs",
    "compute_metric_stats",
]
