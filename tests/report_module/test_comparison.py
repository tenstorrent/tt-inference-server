# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the reproduction comparison.

This decides whether a Partner's reported numbers stand, so the bias is toward
failing. The two rules worth guarding hardest: medians are taken across runs
rather than trusting any single one, and a value that is missing on either side
fails rather than being treated as agreement.
"""

from __future__ import annotations

import pytest

from report_module.comparison import (
    collect_points,
    compare,
    render_markdown,
)


def _report(*records, kind="benchmarks"):
    return {
        "metadata": {"model_name": "m", "device": "d"},
        "sections": [
            {"kind": kind, "id": "m_d", "targets": {}, "data": dict(r)} for r in records
        ],
    }


def _point(conc=1, isl=8192, osl=1024, **metrics):
    base = {
        "concurrency": conc,
        "input_sequence_length": isl,
        "output_sequence_length": osl,
        "mean_ttft_ms": 100.0,
        "tput_user": 35.0,
    }
    base.update(metrics)
    return base


# --------------------------------------------------------------------------
# identifying benchmark sections
# --------------------------------------------------------------------------


def test_benchmark_sections_are_found_by_content_not_kind():
    """`kind` is not stable: older reports carry the tool name, not "benchmarks"."""
    for kind in ("benchmarks", "vllm", "aiperf", "some_future_tool"):
        points = collect_points([_report(_point(), kind=kind)])
        assert len(points) == 1, f"kind={kind} was not recognised"


def test_non_benchmark_sections_are_ignored():
    report = {
        "sections": [
            {"kind": "evals", "data": {"task": "gpqa", "score": 80.0}},
            {"kind": "aiperf_prefix_cache", "data": {"prefix_cache_hit_rate": 0.9}},
        ]
    }
    assert collect_points([report]) == {}


# --------------------------------------------------------------------------
# medians across runs
# --------------------------------------------------------------------------


def test_median_is_taken_across_runs_not_the_first_or_last():
    runs = [
        _report(_point(mean_ttft_ms=90.0)),
        _report(_point(mean_ttft_ms=100.0)),
        _report(_point(mean_ttft_ms=200.0)),  # outlier
    ]
    result = compare(runs, runs, margin=0.05)
    metric = next(m for m in result.points[0].metrics if m.metric == "mean_ttft_ms")
    assert metric.reported == 100.0  # median, not mean (130) or first (90)


def test_an_outlier_in_one_run_does_not_fail_the_comparison():
    reported = [_report(_point(mean_ttft_ms=100.0)) for _ in range(3)]
    measured = [
        _report(_point(mean_ttft_ms=100.0)),
        _report(_point(mean_ttft_ms=101.0)),
        _report(_point(mean_ttft_ms=900.0)),  # one bad run
    ]
    assert compare(reported, measured, margin=0.05).passed


# --------------------------------------------------------------------------
# the margin
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "measured,expected",
    [
        (100.0, True),  # identical
        (104.9, True),  # just inside +5 %
        (105.0, True),  # exactly on the margin
        (105.1, False),  # just outside
        (95.0, True),  # 5 % better still counts as reproduced
        (80.0, False),  # far better is still not a reproduction
    ],
)
def test_margin_is_symmetric_and_inclusive(measured, expected):
    result = compare(
        [_report(_point(mean_ttft_ms=100.0, tput_user=35.0))],
        [_report(_point(mean_ttft_ms=measured, tput_user=35.0))],
        margin=0.05,
    )
    assert result.passed is expected


def test_reporting_far_better_than_measured_also_fails():
    """Reproduction means agreement, not 'at least as good'."""
    result = compare(
        [_report(_point(tput_user=100.0))],
        [_report(_point(tput_user=35.0))],
        margin=0.05,
    )
    assert not result.passed


# --------------------------------------------------------------------------
# missing data fails
# --------------------------------------------------------------------------


def test_metric_missing_on_the_measured_side_fails():
    reported = _point()
    measured = {k: v for k, v in _point().items() if k != "tput_user"}
    result = compare([_report(reported)], [_report(measured)], margin=0.05)
    assert not result.passed
    bad = next(m for m in result.points[0].metrics if m.metric == "tput_user")
    assert "measured" in bad.note


def test_metric_explicitly_null_on_one_side_fails():
    result = compare(
        [_report(_point(mean_ttft_ms=100.0))],
        [_report(_point(mean_ttft_ms=None))],
        margin=0.05,
    )
    assert not result.passed


def test_metric_unmeasured_on_both_sides_is_recorded_but_does_not_fail():
    """Nothing to disagree about — that is an acceptance gap, not a mismatch."""
    result = compare(
        [_report(_point(error_request_count=None))],
        [_report(_point(error_request_count=None))],
        margin=0.05,
    )
    assert result.passed
    assert "error_request_count" in result.points[0].unmeasured
    assert "Captured by neither side" in render_markdown(result)


def test_a_report_compared_against_itself_always_passes():
    """The sanity invariant the whole tool rests on.

    Real reports contain nulls — error_request_count is null throughout the
    gpt-oss-120b release run. If a null were a failure regardless of side, a
    report would fail against itself and no reproduction could ever pass.
    """
    report = _report(
        _point(isl=128, error_request_count=None),
        _point(isl=8192, mean_ttft_ms=1498.0, tput_user=None),
    )
    assert compare([report], [report], margin=0.05).passed


def test_operating_point_only_one_side_ran_is_a_blocking_note():
    result = compare(
        [_report(_point(isl=128), _point(isl=8192))],
        [_report(_point(isl=128))],
        margin=0.05,
    )
    assert not result.passed
    assert any("never measured" in n for n in result.notes)


def test_empty_input_does_not_pass_vacuously():
    assert not compare([], [], margin=0.05).passed
    assert not compare([_report(_point())], [], margin=0.05).passed


def test_reported_zero_against_non_zero_fails_rather_than_dividing():
    result = compare(
        [_report(_point(error_request_count=0))],
        [_report(_point(error_request_count=3))],
        margin=0.05,
    )
    assert not result.passed
    bad = next(m for m in result.points[0].metrics if m.metric == "error_request_count")
    assert bad.relative_difference is None


def test_zero_on_both_sides_agrees():
    result = compare(
        [_report(_point(error_request_count=0))],
        [_report(_point(error_request_count=0))],
        margin=0.05,
    )
    assert result.passed


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def test_markdown_names_the_failing_metric_and_the_verdict():
    result = compare(
        [_report(_point(mean_ttft_ms=100.0))],
        [_report(_point(mean_ttft_ms=200.0))],
        margin=0.05,
    )
    out = render_markdown(result)
    assert "**FAIL**" in out
    assert "mean_ttft_ms" in out
    assert "100.00 %" in out  # the relative difference


def test_markdown_reports_pass_when_everything_agrees():
    runs = [_report(_point())]
    assert "**PASS**" in render_markdown(compare(runs, runs, margin=0.05))
