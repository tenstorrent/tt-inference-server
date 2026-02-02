# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Metrics calculation utilities for media clients.

Pure functions for calculating benchmark metrics like TTFT, RTR, percentiles, etc.
These functions are stateless and can be used anywhere without class inheritance.

Example usage:
    from utils.media_clients.metrics_utils import calculate_ttft, calculate_rtr
    
    ttft = calculate_ttft(status_list)
    rtr = calculate_rtr(status_list)
"""

import logging
import math
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _extract_numeric_value(obj: Any, attributes: Tuple[str, ...]) -> float | None:
    """
    Extract first available numeric value from object attributes.
    
    Args:
        obj: Object to extract value from
        attributes: Attribute names to try in order
        
    Returns:
        Numeric value or None if not found/invalid
    """
    for attr in attributes:
        val = getattr(obj, attr, None)
        if val is not None:
            if isinstance(val, (int, float)):
                return float(val)
            else:
                logger.warning(
                    f"Expected numeric value for {attr}, got {type(val).__name__}"
                )
    return None


def calculate_ttft(
    status_list: List[Any],
    attributes: Tuple[str, ...] = ("ttft_ms", "ttft", "elapsed"),
) -> float:
    """
    Calculate average TTFT (Time To First Token) value.
    
    Args:
        status_list: List of status objects with timing attributes
        attributes: Tuple of attribute names to try in order of preference
        
    Returns:
        Average TTFT value, or 0.0 if no valid values found
        
    Example:
        >>> statuses = [Status(ttft_ms=100), Status(ttft_ms=200)]
        >>> calculate_ttft(statuses)
        150.0
    """
    if not status_list:
        return 0.0

    valid_values = []
    for s in status_list:
        val = _extract_numeric_value(s, attributes)
        if val is not None:
            valid_values.append(val)

    return sum(valid_values) / len(valid_values) if valid_values else 0.0


def calculate_rtr(
    status_list: List[Any],
    attribute: str = "rtr",
) -> float:
    """
    Calculate average RTR (Real-Time Ratio) value.
    
    RTR = audio_duration / processing_time
    Values > 1.0 mean faster than real-time.
    
    Args:
        status_list: List of status objects with RTR attribute
        attribute: Name of the RTR attribute (default: "rtr")
        
    Returns:
        Average RTR value, or 0.0 if no valid values found
    """
    if not status_list:
        return 0.0

    valid_values = []
    for s in status_list:
        val = _extract_numeric_value(s, (attribute,))
        if val is not None:
            valid_values.append(val)

    return sum(valid_values) / len(valid_values) if valid_values else 0.0


def calculate_tail_latency(
    status_list: List[Any],
    attributes: Tuple[str, ...] = ("ttft_ms", "ttft", "elapsed"),
    percentiles: Tuple[float, float] = (0.90, 0.95),
) -> Tuple[float, float]:
    """
    Calculate tail latency percentiles (P90, P95 by default).
    
    Args:
        status_list: List of status objects with timing attributes
        attributes: Tuple of attribute names to try in order of preference
        percentiles: Tuple of two percentile values (default: 90th and 95th)
        
    Returns:
        Tuple of (p90_value, p95_value) in same units as input
        
    Raises:
        ValueError: If percentiles are not between 0 and 1
    """
    if not status_list:
        return 0.0, 0.0

    p1, p2 = percentiles
    if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
        raise ValueError(f"Percentiles must be between 0 and 1, got {percentiles}")

    valid_values = []
    for s in status_list:
        val = _extract_numeric_value(s, attributes)
        if val is not None:
            valid_values.append(val)

    if not valid_values:
        return 0.0, 0.0

    sorted_values = sorted(valid_values)
    n = len(sorted_values)

    p1_idx = min(math.ceil(n * p1) - 1, n - 1)
    p2_idx = min(math.ceil(n * p2) - 1, n - 1)

    return sorted_values[max(0, p1_idx)], sorted_values[max(0, p2_idx)]


def calculate_average(
    status_list: List[Any],
    attribute: str,
    default: float = 0.0,
) -> float:
    """
    Generic average calculation for any numeric attribute.
    
    Args:
        status_list: List of status objects
        attribute: Name of the attribute to average
        default: Default value if no valid values found
        
    Returns:
        Average value of the attribute
    """
    if not status_list:
        return default

    valid_values = []
    for s in status_list:
        val = _extract_numeric_value(s, (attribute,))
        if val is not None:
            valid_values.append(val)

    return sum(valid_values) / len(valid_values) if valid_values else default


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate a single percentile value from a list of numbers.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0.0 to 1.0)
        
    Returns:
        The percentile value
        
    Raises:
        ValueError: If percentile is not between 0 and 1
    """
    if not values:
        return 0.0

    if not (0 <= percentile <= 1):
        raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")

    sorted_values = sorted(values)
    n = len(sorted_values)
    idx = min(math.ceil(n * percentile) - 1, n - 1)

    return sorted_values[max(0, idx)]
