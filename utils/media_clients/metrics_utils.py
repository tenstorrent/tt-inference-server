# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Metrics calculation utilities for benchmark status aggregation.

Provides functions for calculating common benchmark metrics like TTFT,
RTR, and percentiles from lists of test status objects.
"""

import logging
import math
from itertools import chain
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Incremental aggregator for metrics (Welford mean).

    Update in the same loop where you build status_list; then call .result()
    so report generation does not need a second pass over status_list.

    """

    __slots__ = ("_counts", "_means")

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._means: Dict[str, float] = {}

    def add(self, metrics: Dict[str, float]) -> None:
        """Update running mean per key (Welford). O(k) for k keys."""
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
    Aggregate metrics in one pass, O(N) time, O(m) memory.

    Uses Welford's online mean; single loop over all (key, value) pairs.

    Args:
        status_list: List of status objects with get_metrics() method.

    Returns:
        Dict with num_requests and averaged metric values.
        Empty dict if status_list is empty.

    Complexity:
        Time: O(N) — one pass over all metric entries.
        Memory: O(m) — counts and means per unique key.
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


def _extract_numeric_value(
    obj: Any,
    attributes: Tuple[str, ...],
) -> float | None:
    """
    Extract first available numeric value from object attributes.

    Args:
        obj: Object to extract value from.
        attributes: Attribute names to try in order.

    Returns:
        Float value or None if not found/invalid.
    """
    for attr in attributes:
        if (val := getattr(obj, attr, None)) is None:
            continue

        if isinstance(val, (int, float)):
            return float(val)

        logger.warning(
            f"Expected numeric value for '{attr}', got {type(val).__name__}",
        )

    return None


def _collect_valid_values(
    status_list: List[Any],
    attributes: Tuple[str, ...],
) -> List[float]:
    """
    Collect all valid numeric values from status list.

    Args:
        status_list: List of status objects.
        attributes: Attribute names to try for each status.

    Returns:
        List of extracted float values.
    """
    return [
        val
        for status in status_list
        if (val := _extract_numeric_value(status, attributes)) is not None
    ]


def _safe_mean(values: List[float], default: float = 0.0) -> float:
    """
    Calculate mean with empty list handling.

    Args:
        values: List of numeric values.
        default: Value to return if list is empty.

    Returns:
        Mean of values or default.
    """
    return sum(values) / len(values) if values else default


def _percentile_index(n: int, percentile: float) -> int:
    """
    Calculate array index for percentile value.

    Args:
        n: Length of sorted array.
        percentile: Percentile value (0.0 to 1.0).

    Returns:
        Valid index into array of length n.
    """
    if n == 0:
        return 0

    raw_idx = math.ceil(n * percentile) - 1
    return max(0, min(raw_idx, n - 1))


def _validate_percentile(percentile: float) -> None:
    """
    Validate percentile is in range [0, 1].

    Args:
        percentile: Value to validate.

    Raises:
        ValueError: If percentile is outside valid range.
    """
    if not 0 <= percentile <= 1:
        raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")


def calculate_ttft(
    status_list: List[Any],
    attributes: Tuple[str, ...] = ("ttft_ms", "ttft", "elapsed"),
) -> float:
    """
    Calculate average Time To First Token (TTFT).

    Args:
        status_list: List of status objects with timing attributes.
        attributes: Attribute names to try in order of preference.

    Returns:
        Average TTFT value, or 0.0 if no valid values found.

    Example:
        >>> statuses = [Status(ttft_ms=100), Status(ttft_ms=200)]
        >>> calculate_ttft(statuses)
        150.0
    """
    if not status_list:
        return 0.0

    return _safe_mean(_collect_valid_values(status_list, attributes))


def calculate_rtr(
    status_list: List[Any],
    attribute: str = "rtr",
) -> float:
    """
    Calculate average Real-Time Ratio (RTR).

    RTR = audio_duration / processing_time.
    Values > 1.0 indicate faster than real-time processing.

    Args:
        status_list: List of status objects with RTR attribute.
        attribute: Name of the RTR attribute.

    Returns:
        Average RTR value, or 0.0 if no valid values found.

    Example:
        >>> statuses = [Status(rtr=1.5), Status(rtr=2.0)]
        >>> calculate_rtr(statuses)
        1.75
    """
    if not status_list:
        return 0.0

    return _safe_mean(_collect_valid_values(status_list, (attribute,)))


def calculate_tail_latency(
    status_list: List[Any],
    attributes: Tuple[str, ...] = ("ttft_ms", "ttft", "elapsed"),
    percentiles: Tuple[float, float] = (0.90, 0.95),
) -> Tuple[float, float]:
    """
    Calculate tail latency percentiles (P90, P95 by default).

    Args:
        status_list: List of status objects with timing attributes.
        attributes: Attribute names to try in order of preference.
        percentiles: Two percentile values to calculate.

    Returns:
        Tuple of (first_percentile, second_percentile) values.

    Raises:
        ValueError: If percentiles are not between 0 and 1.

    Example:
        >>> statuses = [Status(ttft_ms=i) for i in range(1, 101)]
        >>> calculate_tail_latency(statuses)
        (90.0, 95.0)
    """
    p1, p2 = percentiles
    _validate_percentile(p1)
    _validate_percentile(p2)

    if not status_list:
        return 0.0, 0.0

    values = _collect_valid_values(status_list, attributes)
    if not values:
        return 0.0, 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    return (
        sorted_values[_percentile_index(n, p1)],
        sorted_values[_percentile_index(n, p2)],
    )


def calculate_average(
    status_list: List[Any],
    attribute: str,
    default: float = 0.0,
) -> float:
    """
    Calculate average of any numeric attribute.

    Args:
        status_list: List of status objects.
        attribute: Name of the attribute to average.
        default: Value to return if no valid values found.

    Returns:
        Average value of the attribute.

    Example:
        >>> statuses = [Status(custom=10), Status(custom=20)]
        >>> calculate_average(statuses, "custom")
        15.0
    """
    if not status_list:
        return default

    return _safe_mean(_collect_valid_values(status_list, (attribute,)), default)


def calculate_percentile(
    values: List[float],
    percentile: float,
) -> float:
    """
    Calculate a single percentile from a list of numbers.

    Args:
        values: List of numeric values.
        percentile: Percentile to calculate (0.0 to 1.0).

    Returns:
        The percentile value, or 0.0 if values is empty.

    Raises:
        ValueError: If percentile is not between 0 and 1.

    Example:
        >>> calculate_percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9)
        9
    """
    if not values:
        return 0.0

    _validate_percentile(percentile)

    sorted_values = sorted(values)
    return sorted_values[_percentile_index(len(sorted_values), percentile)]
