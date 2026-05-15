# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the throughput / tail-latency helpers on ``BaseMediaStrategy``.

These helpers are pure functions of their inputs (no I/O, no perf-targets
JSON), so the tests don't need a real strategy subclass.
"""

import unittest

from utils.media_clients.base_strategy_interface import (
    MIN_TAIL_LATENCY_SAMPLES,
    TAIL_LATENCY_PERCENTILES,
    BaseMediaStrategy,
)


class TestCalculateTailLatencies(unittest.TestCase):
    """Tests for ``BaseMediaStrategy._calculate_tail_latencies``."""

    def _expected_keys(self):
        return {f"latency_p{p}" for p in TAIL_LATENCY_PERCENTILES}

    def test_returns_none_for_each_percentile_on_empty_input(self):
        result = BaseMediaStrategy._calculate_tail_latencies([])
        assert set(result) == self._expected_keys()
        assert all(v is None for v in result.values())

    def test_returns_none_below_min_samples(self):
        # 9 samples is one short of the default threshold (10).
        result = BaseMediaStrategy._calculate_tail_latencies(
            [float(i) for i in range(9)]
        )
        assert all(v is None for v in result.values())

    def test_returns_values_at_exactly_min_samples(self):
        # 10 ascending samples: nearest-rank p50 = sorted[ceil(10*0.5)-1] = sorted[4] = 4.0
        # p90 = sorted[8] = 8.0, p95 = sorted[9] = 9.0
        result = BaseMediaStrategy._calculate_tail_latencies(
            [float(i) for i in range(10)]
        )
        assert result["latency_p50"] == 4.0
        assert result["latency_p90"] == 8.0
        assert result["latency_p95"] == 9.0

    def test_skips_none_entries(self):
        # 10 valid + a few Nones; Nones must not pollute the percentile.
        values = [float(i) for i in range(10)] + [None, None, None]
        result = BaseMediaStrategy._calculate_tail_latencies(values)
        assert result["latency_p50"] == 4.0
        assert result["latency_p90"] == 8.0
        assert result["latency_p95"] == 9.0

    def test_unsorted_input_is_sorted_internally(self):
        values = [9, 0, 5, 2, 7, 1, 8, 3, 6, 4]
        result = BaseMediaStrategy._calculate_tail_latencies(values)
        assert result["latency_p50"] == 4
        assert result["latency_p90"] == 8
        assert result["latency_p95"] == 9

    def test_p95_clamps_to_last_index_for_small_n(self):
        # With n=10, ceil(10 * 0.95) - 1 = 9 (the last index). The clamp
        # protects against off-by-one when n is small.
        values = list(range(10))
        result = BaseMediaStrategy._calculate_tail_latencies(values)
        assert result["latency_p95"] == values[-1]

    def test_custom_min_samples_threshold(self):
        # Lower threshold lets us compute percentiles on small samples.
        result = BaseMediaStrategy._calculate_tail_latencies(
            [1.0, 2.0, 3.0], min_samples=3
        )
        assert result["latency_p50"] == 2.0
        assert result["latency_p90"] == 3.0
        assert result["latency_p95"] == 3.0

    def test_default_min_samples_constant_matches_module(self):
        # Defends against silently changing the default by accident.
        assert MIN_TAIL_LATENCY_SAMPLES == 10


class TestCalculateThroughputRps(unittest.TestCase):
    """Tests for ``BaseMediaStrategy._calculate_throughput_rps``."""

    def test_basic_division(self):
        assert BaseMediaStrategy._calculate_throughput_rps(10, 2.0) == 5.0

    def test_returns_none_when_num_requests_is_zero(self):
        assert BaseMediaStrategy._calculate_throughput_rps(0, 2.0) is None

    def test_returns_none_when_wall_clock_is_zero(self):
        assert BaseMediaStrategy._calculate_throughput_rps(10, 0.0) is None

    def test_returns_none_when_wall_clock_is_negative(self):
        # Defensive: monotonic clock should never produce this, but if a
        # producer ever feeds a bogus duration we don't want -inf.
        assert BaseMediaStrategy._calculate_throughput_rps(10, -1.0) is None

    def test_returns_none_when_wall_clock_is_none(self):
        assert BaseMediaStrategy._calculate_throughput_rps(10, None) is None

    def test_fractional_throughput(self):
        # 1 request in 4 seconds → 0.25 rps; we don't round.
        assert BaseMediaStrategy._calculate_throughput_rps(1, 4.0) == 0.25


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
