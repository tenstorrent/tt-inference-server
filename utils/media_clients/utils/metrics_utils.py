# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Metrics aggregation for benchmark status.

Status objects define _METRIC_ATTRS and get_metrics(). Use aggregate_metrics_from_status_list
for means (one pass); use percentiles_from_metric for P90/P95 etc.
"""

import logging
import math
from itertools import chain
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Incremental aggregator for metrics (Welford mean).

    Attributes:
        _counts: Number of values seen per metric key.
        _means: Running mean per metric key.
    """

    __slots__ = ("_counts", "_means")

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._means: Dict[str, float] = {}

    def reset(self) -> None:
        """Clear state so the aggregator can be reused for another run."""
        self._counts.clear()
        self._means.clear()

    def add(self, metrics: Dict[str, float]) -> None:
        """Update running mean per key (Welford)."""
        for key, value in metrics.items():
            if key in self._counts:
                self._counts[key] += 1
                n = self._counts[key]
                self._means[key] += (value - self._means[key]) / n
            else:
                self._counts[key] = 1
                self._means[key] = value

    def result(self, n_requests: int) -> Dict[str, float]:
        """Same shape as aggregate_metrics_from_status_list output."""
        if n_requests == 0:
            return {}
        return {"num_requests": float(n_requests), **self._means}


def aggregate_metrics_from_status_list(
    status_list: List[Any],
) -> Dict[str, float]:
    """
    Aggregate metrics from status list in single pass.

    Prefer MetricsAggregator during benchmark loop to avoid this post-hoc call.
    """
    if not status_list:
        return {}

    counts: Dict[str, int] = {}
    means: Dict[str, float] = {}

    for key, value in chain.from_iterable(
        st.get_metrics().items() for st in status_list
    ):
        if key in counts:
            counts[key] += 1
            means[key] += (value - means[key]) / counts[key]
        else:
            counts[key] = 1
            means[key] = value

    return {"num_requests": float(len(status_list)), **means}


def percentiles_from_metric(
    status_list: List[Any],
    metric_key: str,
    percentiles: Tuple[float, ...] = (0.90, 0.95),
) -> Tuple[float, ...]:
    """
    Compute percentiles for one metric (e.g., P90/P95 for ttft_ms).
    """
    values = [
        m[metric_key]
        for st in status_list
        if (m := st.get_metrics()) and metric_key in m
    ]

    if not values:
        return tuple(0.0 for _ in percentiles)

    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile_index(p: float) -> int:
        idx = math.ceil(n * p) - 1
        return max(0, min(idx, n - 1))

    return tuple(sorted_values[percentile_index(p)] for p in percentiles)
