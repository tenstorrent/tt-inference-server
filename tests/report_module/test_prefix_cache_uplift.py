# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the scored prefix-cache figures.

These two values carry the whole 20-point prefix-cache bonus, and the one that
bites is the sign. The renderer's ``ttft_uplift_pct`` is negative when the cache
helps, because it is a latency delta; the scored ``ttft_reduction_pct`` is
positive when the cache helps, because the rubric asks for a reduction. Reading
one as the other inverts the line, so the sign is asserted directly rather than
left to follow from the arithmetic.
"""

from __future__ import annotations

import pytest

from report_module.prefix_cache_uplift import (
    PREFIX_CACHE_KIND,
    REQUIRED_SCENARIOS,
    apply_prefix_cache_uplift,
    decode_hit_rate,
    delta_pct,
    summarize_prefix_cache_scoring,
    ttft_reduction_pct,
)
from report_module.schema import Block


def _block(scenario="shared_system", **data) -> Block:
    base = {
        "scenario": scenario,
        "label": f"{scenario}-c8",
        "concurrency": 8,
        "arrival_pattern": "poisson",
        "isl_mean": 8192,
        "mean_ttft_ms": 1000.0,
    }
    base.update(data)
    return Block(kind=PREFIX_CACHE_KIND, id="m_d", title="t", data=base, targets={})


def _sweep(treatment_ttft=560.0, baseline_ttft=1000.0, hit_rate=0.92):
    """One baseline plus one treatment run per required scenario."""
    blocks = [_block("baseline", mean_ttft_ms=baseline_ttft, prefix_cache_hit_rate=0.0)]
    blocks += [
        _block(s, mean_ttft_ms=treatment_ttft, prefix_cache_hit_rate=hit_rate)
        for s in REQUIRED_SCENARIOS
    ]
    return blocks


# --------------------------------------------------------------------------
# the sign
# --------------------------------------------------------------------------


def test_a_cache_that_helps_gives_a_positive_reduction():
    """1000 ms -> 560 ms is a 44 % reduction, the Appendix F.2 worked value."""
    assert ttft_reduction_pct(560.0, 1000.0) == pytest.approx(44.0)


def test_reduction_is_the_opposite_sign_to_the_displayed_delta():
    """The trap. The table shows -44 %; the rubric scores +44 %."""
    assert delta_pct(560.0, 1000.0) == pytest.approx(-44.0)
    assert ttft_reduction_pct(560.0, 1000.0) == pytest.approx(44.0)


def test_a_cache_that_hurts_gives_a_negative_reduction():
    """Must land below the 0 % qualifying value, not above it."""
    assert ttft_reduction_pct(1440.0, 1000.0) == pytest.approx(-44.0)


def test_a_cache_that_does_nothing_is_zero_not_none():
    """0 % is a real result sitting exactly at qualifying, not a missing one."""
    assert ttft_reduction_pct(1000.0, 1000.0) == 0.0


@pytest.mark.parametrize(
    "treatment,baseline",
    [(None, 1000.0), (560.0, None), (560.0, 0), (560.0, "x")],
)
def test_reduction_is_none_when_uncomputable(treatment, baseline):
    assert ttft_reduction_pct(treatment, baseline) is None


# --------------------------------------------------------------------------
# which hit rate is scored
# --------------------------------------------------------------------------


def test_decode_rate_wins_when_the_deployment_is_disaggregated():
    """Prefill is structurally low and explicitly not scored (RFP G.3.3)."""
    record = {
        "prefix_cache_hit_rate": 0.55,
        "prefix_cache_hit_rate_prefill": 0.20,
        "prefix_cache_hit_rate_decode": 0.93,
    }
    assert decode_hit_rate(record) == 0.93


def test_the_single_cache_is_the_decode_cache_when_aggregated():
    assert decode_hit_rate({"prefix_cache_hit_rate": 0.91}) == 0.91


def test_decode_rate_is_none_when_no_cache_metrics_were_captured():
    assert decode_hit_rate({"mean_ttft_ms": 1000.0}) is None


# --------------------------------------------------------------------------
# attaching to blocks
# --------------------------------------------------------------------------


def test_treatment_blocks_get_the_scored_fields():
    out = apply_prefix_cache_uplift(_sweep())
    treatment = next(b for b in out if b.data["scenario"] == "shared_system")
    assert treatment.data["ttft_reduction_pct"] == pytest.approx(44.0)
    assert treatment.data["baseline_mean_ttft_ms"] == 1000.0
    assert treatment.data["decode_prefix_cache_hit_rate"] == 0.92


def test_baseline_blocks_carry_the_fields_as_none():
    """Always present, so absent never has to be told apart from zero."""
    out = apply_prefix_cache_uplift(_sweep())
    baseline = next(b for b in out if b.data["scenario"] == "baseline")
    assert baseline.data["ttft_reduction_pct"] is None
    assert baseline.data["baseline_mean_ttft_ms"] is None


def test_baselines_are_matched_on_shape_not_just_taken_from_any_run():
    """Comparing across concurrency would measure concurrency, not the cache."""
    blocks = [
        _block("baseline", concurrency=1, mean_ttft_ms=100.0),
        _block("baseline", concurrency=64, mean_ttft_ms=2000.0),
        _block("shared_system", concurrency=64, mean_ttft_ms=1000.0),
    ]
    out = apply_prefix_cache_uplift(blocks)
    treatment = next(b for b in out if b.data["scenario"] == "shared_system")
    assert treatment.data["baseline_mean_ttft_ms"] == 2000.0
    assert treatment.data["ttft_reduction_pct"] == pytest.approx(50.0)


def test_a_treatment_run_with_no_matching_baseline_is_none_and_warns(caplog):
    blocks = [
        _block("baseline", concurrency=1),
        _block("shared_system", concurrency=64),
    ]
    with caplog.at_level("WARNING"):
        out = apply_prefix_cache_uplift(blocks)
    treatment = next(b for b in out if b.data["scenario"] == "shared_system")
    assert treatment.data["ttft_reduction_pct"] is None
    assert "shared_system" in caplog.text


def test_a_sweep_with_no_baseline_at_all_warns_loudly(caplog):
    with caplog.at_level("WARNING"):
        apply_prefix_cache_uplift([_block("shared_system")])
    assert "no baseline runs" in caplog.text.lower()


def test_non_prefix_cache_blocks_pass_through_untouched():
    other = Block(kind="benchmarks", id="x", title="t", data={"a": 1}, targets={})
    out = apply_prefix_cache_uplift([other])
    assert out[0] is other


# --------------------------------------------------------------------------
# the submission-level summary
# --------------------------------------------------------------------------


def test_summary_averages_the_required_scenarios():
    summary = summarize_prefix_cache_scoring(apply_prefix_cache_uplift(_sweep()))
    assert summary["ttft_reduction_pct"] == pytest.approx(44.0)
    assert summary["decode_prefix_cache_hit_rate"] == pytest.approx(0.92)
    assert summary["scenarios_missing"] == []


def test_summary_reports_a_missing_required_scenario():
    """Silently averaging the rest would reward dropping the weakest scenario."""
    blocks = [_block("baseline"), _block("shared_system"), _block("prefix_pool")]
    summary = summarize_prefix_cache_scoring(apply_prefix_cache_uplift(blocks))
    assert summary["scenarios_missing"] == ["multi_turn"]


def test_the_optional_trace_scenario_cannot_move_the_score():
    """Running an optional scenario must never be a strategic decision."""
    without = summarize_prefix_cache_scoring(apply_prefix_cache_uplift(_sweep()))
    with_trace = summarize_prefix_cache_scoring(
        apply_prefix_cache_uplift(
            _sweep() + [_block("mooncake_trace", mean_ttft_ms=10.0)]
        )
    )
    assert with_trace == without


def test_a_scenario_run_at_more_points_does_not_thereby_count_for_more():
    """Average within a scenario first, then across scenarios."""
    blocks = _sweep()
    # shared_system measured three times, the others once each
    blocks += [_block("shared_system", mean_ttft_ms=560.0, prefix_cache_hit_rate=0.10)]
    blocks += [_block("shared_system", mean_ttft_ms=560.0, prefix_cache_hit_rate=0.10)]
    summary = summarize_prefix_cache_scoring(apply_prefix_cache_uplift(blocks))
    # shared_system mean = (0.92+0.10+0.10)/3 = 0.3733; across three scenarios:
    # (0.3733 + 0.92 + 0.92)/3
    assert summary["decode_prefix_cache_hit_rate"] == pytest.approx(0.7378, abs=1e-4)


def test_summary_is_none_not_zero_when_nothing_was_measured():
    summary = summarize_prefix_cache_scoring([])
    assert summary["ttft_reduction_pct"] is None
    assert summary["decode_prefix_cache_hit_rate"] is None
    assert summary["scenarios_missing"] == list(REQUIRED_SCENARIOS)
