"""Tests for the defensive coercion helpers added to workflows/run_reports.py.

These guard the report generator against benchmark rows that produced
no real numbers (e.g. when the upstream server returned 401 and the
benchmark wrote ``"N/A"`` / ``NaN`` placeholders into mean_tps,
mean_ttft_ms, and tps_decode_throughput).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# The full run_reports module pulls in heavy benchmarking dependencies
# at import time. Tests for the small helpers only need the functions
# themselves, so we lift them by name via importlib.
import importlib  # noqa: E402

run_reports = importlib.import_module("workflows.run_reports")

_coerce_metric = run_reports._coerce_metric
_is_zero_token_row = run_reports._is_zero_token_row


# ---------------------------------------------------------------------------
# _coerce_metric
# ---------------------------------------------------------------------------


def test_coerce_metric_passes_floats_through():
    assert _coerce_metric({"x": 3.14}, "x") == 3.14


def test_coerce_metric_promotes_ints_to_float():
    assert _coerce_metric({"x": 5}, "x") == 5.0
    assert isinstance(_coerce_metric({"x": 5}, "x"), float)


def test_coerce_metric_rejects_bool_to_avoid_silent_truthy_coercion():
    # ``True`` would otherwise coerce to 1.0 - undesirable for a metric.
    assert _coerce_metric({"x": True}, "x") is None
    assert _coerce_metric({"x": False}, "x") is None


def test_coerce_metric_parses_numeric_strings():
    assert _coerce_metric({"x": "12.5"}, "x") == 12.5
    assert _coerce_metric({"x": "  7 "}, "x") == 7.0
    assert _coerce_metric({"x": "0"}, "x") == 0.0


def test_coerce_metric_returns_none_for_na_placeholders():
    for marker in ("N/A", "n/a", "NA", "nan", "NaN", "None", "", "   "):
        assert _coerce_metric({"x": marker}, "x") is None, marker


def test_coerce_metric_returns_none_for_none_and_missing_keys():
    assert _coerce_metric({"x": None}, "x") is None
    assert _coerce_metric({}, "x") is None


def test_coerce_metric_returns_none_for_unparseable_strings():
    assert _coerce_metric({"x": "abc"}, "x") is None
    assert _coerce_metric({"x": "12.3.4"}, "x") is None


def test_coerce_metric_filters_nan_and_inf_floats():
    assert _coerce_metric({"x": float("nan")}, "x") is None
    assert _coerce_metric({"x": float("inf")}, "x") is None
    assert _coerce_metric({"x": float("-inf")}, "x") is None
    # And the string versions:
    assert _coerce_metric({"x": "inf"}, "x") is None


def test_coerce_metric_rejects_arbitrary_objects():
    assert _coerce_metric({"x": [1, 2, 3]}, "x") is None
    assert _coerce_metric({"x": {"nested": 1}}, "x") is None


# ---------------------------------------------------------------------------
# _is_zero_token_row
# ---------------------------------------------------------------------------


def test_is_zero_token_row_detects_zero_total_generated():
    assert _is_zero_token_row({"total_generated_tokens": 0})
    assert _is_zero_token_row({"total_generated_tokens": "0"})
    assert _is_zero_token_row({"total_generated_tokens": 0.0})


def test_is_zero_token_row_false_for_nonzero():
    assert not _is_zero_token_row({"total_generated_tokens": 5})
    assert not _is_zero_token_row({"total_generated_tokens": "100"})


def test_is_zero_token_row_false_when_metric_missing_or_unparseable():
    # Missing entirely - cannot confirm zero, do not classify as zero-token.
    assert not _is_zero_token_row({})
    # Unparseable string - same: caller will get N/A elsewhere via _coerce_metric.
    assert not _is_zero_token_row({"total_generated_tokens": "N/A"})


def test_is_zero_token_row_accepts_alternate_keys():
    assert _is_zero_token_row({"total_output_tokens": 0})
    assert _is_zero_token_row({"output_tokens": 0})


def test_is_zero_token_row_handles_nan_as_not_zero():
    # NaN is neither zero nor a real number; do not falsely flag.
    assert not _is_zero_token_row({"total_generated_tokens": math.nan})
