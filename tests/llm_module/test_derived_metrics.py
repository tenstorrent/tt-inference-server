# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the derived prefill quality metrics.

These three values carry 17 of the 55 prefill points in the Milestone-0 grading
rubric, and none of them is measured directly by any benchmark tool — they are
computed here. The invariant that matters most is that an unavailable value is
``None`` rather than 0: for a latency-derived figure 0 is the *best* possible
result, so a coerced zero would score as perfect rather than as unmeasured.
"""

from __future__ import annotations

import math

import pytest

from llm_module.derived_metrics import (
    MIN_POINTS_FOR_FIT,
    apply_derived_metrics,
    apply_scaling_exponents,
    fit_scaling_exponent,
    prefill_throughput,
    ttft_tail_ratio,
)
from report_module.schema import Block


def _block(**data) -> Block:
    from llm_module.parsers.base import BENCHMARKS_KIND as KIND

    return Block(kind=KIND, id="m_d", title="t", data=dict(data), targets={})


# --------------------------------------------------------------------------
# prefill throughput
# --------------------------------------------------------------------------


def test_prefill_throughput_is_isl_over_ttft_seconds():
    # 8192 tokens in 44.4 ms -> ~184.5k tok/s
    assert prefill_throughput(
        {"input_sequence_length": 8192, "mean_ttft_ms": 44.4}
    ) == pytest.approx(184504.5, rel=1e-4)


@pytest.mark.parametrize(
    "record",
    [
        {"input_sequence_length": 8192},  # no ttft
        {"mean_ttft_ms": 44.4},  # no isl
        {"input_sequence_length": 8192, "mean_ttft_ms": 0},  # zero ttft
        {"input_sequence_length": 0, "mean_ttft_ms": 44.4},  # zero isl
        {"input_sequence_length": 8192, "mean_ttft_ms": None},
    ],
)
def test_prefill_throughput_is_none_when_uncomputable(record):
    assert prefill_throughput(record) is None


# --------------------------------------------------------------------------
# tail ratio
# --------------------------------------------------------------------------


def test_tail_ratio_is_p99_over_p50():
    assert ttft_tail_ratio({"p99_ttft": 2093.6, "p50_ttft": 786.0}) == pytest.approx(
        2.6636, rel=1e-4
    )


def test_tail_ratio_of_one_means_no_tail():
    assert ttft_tail_ratio({"p99_ttft": 500.0, "p50_ttft": 500.0}) == 1.0


@pytest.mark.parametrize(
    "record",
    [{"p99_ttft": 2093.6}, {"p50_ttft": 786.0}, {"p99_ttft": 2093.6, "p50_ttft": 0}],
)
def test_tail_ratio_is_none_when_uncomputable(record):
    assert ttft_tail_ratio(record) is None


# --------------------------------------------------------------------------
# scaling exponent
# --------------------------------------------------------------------------


def test_exponent_of_perfectly_linear_prefill_is_one():
    """ttft = k * isl -> slope 1.0 in log-log."""
    points = [(isl, 0.005 * isl) for isl in (128, 1024, 8192, 32768)]
    assert fit_scaling_exponent(points) == pytest.approx(1.0, abs=1e-6)


def test_exponent_of_quadratic_prefill_is_two():
    """ttft = k * isl^2 — the signature of attention cost dominating."""
    points = [(isl, 1e-6 * isl**2) for isl in (128, 1024, 8192, 32768)]
    assert fit_scaling_exponent(points) == pytest.approx(2.0, abs=1e-6)


def test_exponent_below_one_when_fixed_overhead_dominates():
    """Short inputs are overhead-bound, so cost grows slower than input."""
    points = [(isl, 20.0 + 0.005 * isl) for isl in (128, 1024, 8192, 32768)]
    exponent = fit_scaling_exponent(points)
    assert 0.0 < exponent < 1.0


def test_exponent_needs_at_least_three_points():
    two = [(128, 1.0), (1024, 8.0)]
    assert len(two) < MIN_POINTS_FOR_FIT
    assert fit_scaling_exponent(two) is None


def test_exponent_is_none_when_every_input_length_is_identical():
    """A vertical fit has no slope; must not raise or return a bogus value."""
    assert fit_scaling_exponent([(8192, 40.0), (8192, 44.0), (8192, 42.0)]) is None


def test_exponent_ignores_unusable_points():
    points = [(128, 1.0), (1024, 8.0), (8192, None), (32768, 256.0), (None, 5.0)]
    assert fit_scaling_exponent(points) is not None


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_apply_derived_metrics_enriches_a_benchmark_block():
    block = apply_derived_metrics(
        _block(
            input_sequence_length=8192,
            mean_ttft_ms=44.4,
            p50_ttft=40.0,
            p99_ttft=60.0,
            concurrency=1,
        )
    )
    assert block.data["prefill_throughput_tok_s"] == pytest.approx(184504.5, rel=1e-4)
    assert block.data["ttft_tail_ratio"] == pytest.approx(1.5)
    # original fields survive
    assert block.data["mean_ttft_ms"] == 44.4


def test_apply_derived_metrics_leaves_non_benchmark_blocks_alone():
    other = Block(
        kind="aiperf_prefix_cache", id="x", title="t", data={"a": 1}, targets={}
    )
    assert apply_derived_metrics(other) is other


def test_apply_derived_metrics_emits_none_not_zero_when_unmeasured():
    block = apply_derived_metrics(_block(concurrency=1))
    assert block.data["prefill_throughput_tok_s"] is None
    assert block.data["ttft_tail_ratio"] is None


def test_scaling_exponent_is_fitted_separately_per_concurrency():
    """The point of the metric: a system can scale well idle and badly loaded.

    Concurrency 1 is linear (exponent ~1.0); concurrency 32 is quadratic
    (~2.0). A single pooled fit would land between the two and hide it.
    """
    blocks = [
        _block(concurrency=1, input_sequence_length=isl, p50_ttft=0.005 * isl)
        for isl in (128, 1024, 8192, 32768)
    ] + [
        _block(concurrency=32, input_sequence_length=isl, p50_ttft=1e-6 * isl**2)
        for isl in (128, 1024, 8192, 32768)
    ]

    out = apply_scaling_exponents(blocks)
    at_1 = [b.data["ttft_scaling_exponent"] for b in out if b.data["concurrency"] == 1]
    at_32 = [
        b.data["ttft_scaling_exponent"] for b in out if b.data["concurrency"] == 32
    ]

    # every block at a level carries that level's exponent
    assert len(set(at_1)) == 1 and len(set(at_32)) == 1
    assert at_1[0] == pytest.approx(1.0, abs=1e-3)
    assert at_32[0] == pytest.approx(2.0, abs=1e-3)

    pooled = fit_scaling_exponent(
        [(b.data["input_sequence_length"], b.data["p50_ttft"]) for b in blocks]
    )
    assert not math.isclose(pooled, 1.0, abs_tol=0.1)
    assert not math.isclose(pooled, 2.0, abs_tol=0.1)


def test_scaling_exponent_is_none_when_a_concurrency_has_too_few_points(caplog):
    """Issue #64: the sweep must carry >=3 input lengths at every graded level."""
    blocks = [
        _block(concurrency=1, input_sequence_length=isl, p50_ttft=0.005 * isl)
        for isl in (128, 1024, 8192)
    ] + [
        _block(concurrency=32, input_sequence_length=isl, p50_ttft=0.16 * isl)
        for isl in (128, 1024)
    ]

    with caplog.at_level("WARNING"):
        out = apply_scaling_exponents(blocks)

    assert all(
        b.data["ttft_scaling_exponent"] is None
        for b in out
        if b.data["concurrency"] == 32
    )
    assert all(
        b.data["ttft_scaling_exponent"] is not None
        for b in out
        if b.data["concurrency"] == 1
    )
    assert "concurrency 32" in caplog.text
