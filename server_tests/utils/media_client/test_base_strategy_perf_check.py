# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the perf-check helper on ``BaseMediaStrategy``.

These tests target the pure helper logic (``PerfCheck`` dataclass +
``calculate_performance_check`` method) so they don't need a real strategy
subclass or any I/O.
"""

import unittest
from typing import Optional
from unittest.mock import MagicMock

from utils.media_clients.base_strategy_interface import (
    DEFAULT_PERF_CHECK_TOLERANCE,
    BaseMediaStrategy,
    PerfCheck,
)
from workflows.workflow_types import ReportCheckTypes


class _BareStrategy(BaseMediaStrategy):
    """Concrete subclass that satisfies the ABC contract for test fixtures."""

    def run_eval(self) -> None:  # pragma: no cover - not exercised
        pass

    def run_benchmark(self):  # pragma: no cover - not exercised
        return []


def _make_strategy() -> _BareStrategy:
    return _BareStrategy(
        all_params=MagicMock(),
        model_spec=MagicMock(),
        device=MagicMock(),
        output_path="/tmp/out",
        service_port=8000,
    )


def _check(
    name: str,
    measured: Optional[float],
    target: Optional[float],
    lower_is_better: bool,
) -> PerfCheck:
    return PerfCheck(
        name=name,
        measured=measured,
        target=target,
        lower_is_better=lower_is_better,
    )


class TestCalculatePerformanceCheck(unittest.TestCase):
    """Tests for ``BaseMediaStrategy.calculate_performance_check``."""

    def test_returns_na_when_no_checks_supplied(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(checks=[])
        assert result == ReportCheckTypes.NA

    def test_returns_na_when_all_measurements_missing(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[
                _check("latency", None, 1.0, True),
                _check("rtr", None, 2.0, False),
            ]
        )
        assert result == ReportCheckTypes.NA

    def test_returns_na_when_all_targets_missing(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[
                _check("latency", 0.5, None, True),
                _check("rtr", 2.5, None, False),
            ]
        )
        assert result == ReportCheckTypes.NA

    def test_pass_when_lower_is_better_within_tolerance(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[_check("latency", 1.04, 1.0, True)],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.PASS

    def test_fail_when_lower_is_better_exceeds_tolerance(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[_check("latency", 1.10, 1.0, True)],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.FAIL

    def test_pass_when_higher_is_better_within_tolerance(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[_check("rtr", 1.96, 2.0, False)],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.PASS

    def test_fail_when_higher_is_better_below_tolerance(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[_check("rtr", 1.50, 2.0, False)],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.FAIL

    def test_skips_checks_with_missing_pieces(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[
                _check("latency", 0.9, 1.0, True),
                _check("missing-target", 5.0, None, True),
                _check("missing-measured", None, 10.0, True),
            ],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.PASS

    def test_fail_when_any_check_fails(self):
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[
                _check("latency", 0.9, 1.0, True),
                _check("rtr", 0.5, 2.0, False),
            ],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.FAIL

    def test_uses_default_tolerance_when_none(self):
        strategy = _make_strategy()
        edge = 1.0 * (1 + DEFAULT_PERF_CHECK_TOLERANCE)
        result = strategy.calculate_performance_check(
            checks=[_check("latency", edge, 1.0, True)],
            tolerance=None,
        )
        assert result == ReportCheckTypes.PASS

    def test_zero_measurement_is_not_treated_as_missing(self):
        """0.0 must be a valid measurement (regression for boolean-falsy bug)."""
        strategy = _make_strategy()
        result = strategy.calculate_performance_check(
            checks=[_check("latency", 0.0, 1.0, True)],
            tolerance=0.05,
        )
        assert result == ReportCheckTypes.PASS


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
