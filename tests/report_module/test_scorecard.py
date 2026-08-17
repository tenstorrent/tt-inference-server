# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the Milestone-0 rubric scoring.

The centrepiece is the golden reproduction of the RFP's own worked examples.
Appendix F.2 and F.3 are published to Partners, who are told they can compute
their own score before submitting; if this module and those tables ever disagree,
one of them is wrong and a Partner will find out during a bid dispute. So the
published figures are asserted to the decimal place they are printed at.
"""

from __future__ import annotations

import pytest

from report_module.scorecard import (
    CONCURRENCY_SHARES,
    GROUPS,
    LINE_WEIGHTS,
    GradedPoint,
    LineScore,
    Scorecard,
    Submission,
    fraction,
    point_weights,
    render_markdown,
    score,
    to_dict,
)

# --------------------------------------------------------------------------
# The Appendix F.3 submission, transcribed from the published tables
# --------------------------------------------------------------------------

# (concurrency, input length, mean TTFT target ms, measured p99 ms)
F3_P99 = [
    (1, 128, 25.0, 21.0),
    (1, 1024, 100.0, 90.0),
    (1, 8192, 650.0, 620.0),
    (1, 32768, 2800.0, 2800.0),
    (1, 131072, 13000.0, 13650.0),
    (128, 128, 700.0, 700.0),
    (128, 1024, 3400.0, 3570.0),
    (128, 8192, 22000.0, 24200.0),
    (128, 32768, 88000.0, 101200.0),
    (128, 131072, 350000.0, 420000.0),
]

# Published per-point weights, F.3 step 1.
F3_WEIGHTS = {
    (1, 128): 0.0282,
    (1, 1024): 0.0403,
    (1, 8192): 0.0524,
    (1, 32768): 0.0605,
    (1, 131072): 0.0685,
    (128, 128): 0.0847,
    (128, 1024): 0.1210,
    (128, 8192): 0.1573,
    (128, 32768): 0.1815,
    (128, 131072): 0.2056,
}

# Published per-point fractions for the p99 line, F.3 step 2.
F3_P99_FRACTIONS = [
    0.756,
    0.667,
    0.587,
    0.519,
    0.444,
    0.519,
    0.444,
    0.370,
    0.296,
    0.222,
]


def _f3_points() -> list[GradedPoint]:
    return [
        GradedPoint(
            concurrency=c, input_length=isl, target_ttft_ms=target, p99_ttft=measured
        )
        for c, isl, target, measured in F3_P99
    ]


# --------------------------------------------------------------------------
# F.3 — per-point weights
# --------------------------------------------------------------------------


def test_per_point_weights_reproduce_appendix_f3():
    weights = point_weights([(c, isl) for c, isl, _, _ in F3_P99])
    for key, published in F3_WEIGHTS.items():
        assert weights[key] == pytest.approx(published, abs=5e-5), key


def test_per_point_weights_sum_to_one():
    weights = point_weights([(c, isl) for c, isl, _, _ in F3_P99])
    assert sum(weights.values()) == pytest.approx(1.0)


def test_the_heaviest_point_is_the_longest_input_under_full_load():
    """The stated intent of the weighting, so worth pinning."""
    weights = point_weights([(c, isl) for c, isl, _, _ in F3_P99])
    assert max(weights, key=lambda k: weights[k]) == (128, 131072)


def test_the_loaded_corner_carries_three_quarters():
    weights = point_weights([(c, isl) for c, isl, _, _ in F3_P99])
    loaded = sum(w for (c, _), w in weights.items() if c == 128)
    assert loaded == pytest.approx(0.75)


def test_weights_need_exactly_two_concurrency_levels():
    """Published shares describe two corners; three would invent weights."""
    with pytest.raises(ValueError, match="two concurrency levels"):
        point_weights([(1, 128), (32, 128), (128, 128)])


# --------------------------------------------------------------------------
# F.3 — the p99 line, worked in full
# --------------------------------------------------------------------------


def test_p99_per_point_fractions_reproduce_appendix_f3():
    card = score(Submission(points=_f3_points()))
    got = [p.fraction for p in card.lines["ttft_p99"].points]
    for actual, published in zip(got, F3_P99_FRACTIONS):
        assert actual == pytest.approx(published, abs=5e-4)


def test_p99_line_fraction_and_score_reproduce_appendix_f3():
    card = score(Submission(points=_f3_points()))
    line = card.lines["ttft_p99"]
    assert round(line.fraction, 4) == 0.3962
    assert round(line.score, 2) == 8.72


def test_every_point_is_judged_against_its_own_range():
    """24 ms and 478 000 ms both land inside 0-1; no global threshold could."""
    card = score(Submission(points=_f3_points()))
    fractions = [p.fraction for p in card.lines["ttft_p99"].points]
    assert all(0.0 < f < 1.0 for f in fractions)


def test_qualifying_and_excellence_are_derived_not_authored():
    card = score(Submission(points=_f3_points()))
    first = card.lines["ttft_p99"].points[0]
    assert first.qualifying == pytest.approx(25.0 * 1.35)  # 33.75 ms
    assert first.excellence == pytest.approx(33.75 * 0.50)  # 16.875 ms


# --------------------------------------------------------------------------
# F.3 — scaling quality, fitted per concurrency level
# --------------------------------------------------------------------------


def test_scaling_quality_reproduces_appendix_f3():
    card = score(Submission(scaling_exponents={1: 0.956, 128: 0.916}))
    line = card.lines["scaling_quality"]
    assert line.fraction == pytest.approx(1.0)
    assert round(line.score, 2) == 3.00


def test_scaling_well_when_idle_and_badly_under_load_scores_only_the_idle_share():
    """The reason the fits are kept separate rather than pooled."""
    card = score(Submission(scaling_exponents={1: 1.0, 128: 2.0}))
    assert card.lines["scaling_quality"].fraction == pytest.approx(0.25)


def test_an_unfittable_level_contributes_zero_and_says_so():
    card = score(Submission(scaling_exponents={1: 1.0, 128: None}))
    line = card.lines["scaling_quality"]
    assert line.fraction == pytest.approx(0.25)
    assert "concurrency [128]" in line.note


# --------------------------------------------------------------------------
# F.2 — the full scorecard
# --------------------------------------------------------------------------


def _f2_submission() -> Submission:
    """The F.2 submission, with per-point lines pinned to their F.3 fractions.

    F.2 rolls up F.3, so the prefill fractions it prints are inputs here rather
    than things to recompute; only the p99 line is recomputed from raw points
    (above), because that is the one F.3 works through point by point.
    """
    return Submission(
        partner="Example Partner",
        model="google/gemma-4-31B-it",
        scaling_exponents={1: 0.956, 128: 0.916},
        once={
            "agentic_eval": 1.09,
            "standard_eval": 1.04,
            "run_to_run_cov": 0.05,
            "contribution_quality": 3.0,
            "technical_assistance": 3.0,
            "prefix_cache_hit_rate": 0.78,
            "ttft_uplift": 44.0,
        },
        reproduced_first_attempt=True,
    )


@pytest.mark.parametrize(
    "key,fraction_,score_",
    [
        ("agentic_eval", 0.6000, 7.20),
        ("standard_eval", 0.2667, 2.67),
        ("reproduced_first_attempt", 1.0000, 3.00),
        ("run_to_run_cov", 0.7692, 1.54),
        ("contribution_quality", 1.0000, 3.00),
        ("technical_assistance", 0.7500, 1.50),
        ("prefix_cache_hit_rate", 0.6667, 8.00),
        ("ttft_uplift", 0.7333, 5.87),
    ],
)
def test_once_scored_lines_reproduce_appendix_f2(key, fraction_, score_):
    card = score(_f2_submission())
    assert round(card.lines[key].fraction, 4) == fraction_
    assert round(card.lines[key].score, 2) == score_


def test_f2_bonus_and_group_totals():
    card = score(_f2_submission())
    assert round(card.group_score("quality"), 2) == 9.87
    assert round(card.bonus_total, 2) == 13.87


# Every line fraction printed in F.2. The per-point ones are rolled up from F.3
# rather than recomputed, which is exactly how F.2 presents them.
F2_LINE_FRACTIONS = {
    "ttft_p99": 0.3962,
    "ttft_p90": 0.6054,
    "ttft_p50": 0.4879,
    "prefill_throughput": 0.4139,
    "tail_discipline": 0.5000,
    "scaling_quality": 1.0000,
    "tput_user_median": 0.5000,
    "decode_throughput": 0.2000,
    "agentic_eval": 0.6000,
    "standard_eval": 0.2667,
    "reproduced_first_attempt": 1.0000,
    "run_to_run_cov": 0.7692,
    "contribution_quality": 1.0000,
    "technical_assistance": 0.7500,
    "prefix_cache_hit_rate": 0.6667,
    "ttft_uplift": 0.7333,
}


def _f2_card() -> Scorecard:
    return Scorecard(
        partner="Example Partner",
        model="google/gemma-4-31B-it",
        lines={
            key: LineScore(key=key, weight=LINE_WEIGHTS[key], fraction=frac)
            for key, frac in F2_LINE_FRACTIONS.items()
        },
    )


@pytest.mark.parametrize(
    "group,expected",
    [("prefill", 26.95), ("decode", 4.70), ("quality", 9.87), ("engineering", 9.04)],
)
def test_group_subtotals_reproduce_appendix_f2(group, expected):
    assert round(_f2_card().group_score(group), 2) == expected


def test_core_bonus_and_overall_reproduce_appendix_f2():
    card = _f2_card()
    assert round(card.core_total, 2) == 50.56
    assert round(card.bonus_total, 2) == 13.87
    assert round(card.overall, 2) == 64.43


def test_totals_are_summed_at_full_precision_not_from_rounded_subtotals():
    """Rounding is a display step. If it fed back into the arithmetic, a submission's
    total would depend on how many decimals the scorecard happens to print, and the
    error would compound with every line.

    Asserted as an exact identity rather than against a worked example: whether the
    rounded and unrounded sums visibly disagree depends on the numbers in play, so
    an example that happens to agree would silently stop testing anything.
    """
    card = _f2_card()
    core_keys = [
        k for g in ("prefill", "decode", "quality", "engineering") for k in GROUPS[g]
    ]
    assert card.core_total == sum(card.lines[k].score for k in core_keys)
    for group in ("prefill", "decode", "quality", "engineering"):
        assert card.group_score(group) == sum(
            card.lines[k].score for k in GROUPS[group]
        )


def test_the_core_weights_sum_to_one_hundred_and_the_bonus_to_twenty():
    card = _f2_card()
    assert (
        sum(
            card.group_weight(g)
            for g in ("prefill", "decode", "quality", "engineering")
        )
        == 100
    )
    assert card.group_weight("bonus") == 20


def test_prefill_dominates_the_core_score_as_the_rubric_intends():
    """Prefill carries 55 of the 100 core points and must remain the largest single
    contribution. Asserted as "largest group" rather than against a fixed share: the
    share moves with whatever the example submission scores, but the ordering is the
    design intent."""
    card = _f2_card()
    scores = {
        g: card.group_score(g) for g in ("prefill", "decode", "quality", "engineering")
    }
    assert max(scores, key=lambda g: scores[g]) == "prefill"
    assert card.group_weight("prefill") == 55
    assert (
        card.group_weight("prefill")
        > sum(card.group_weight(g) for g in ("decode", "quality")) / 2
    )


def test_lower_is_better_lines_need_no_special_case():
    """Coefficient of variation: excellence 0.02 sits below qualifying 0.15."""
    assert fraction(0.05, 0.15, 0.02) == pytest.approx(0.7692, abs=5e-5)
    assert fraction(0.15, 0.15, 0.02) == 0.0
    assert fraction(0.01, 0.15, 0.02) == 1.0


# --------------------------------------------------------------------------
# The scoring formula
# --------------------------------------------------------------------------


def test_meeting_the_qualifying_value_scores_nothing():
    """The central property of the rubric (K.2)."""
    assert fraction(100.0, 100.0, 50.0) == 0.0


def test_reaching_excellence_scores_full_marks_and_beyond_is_clamped():
    assert fraction(50.0, 100.0, 50.0) == 1.0
    assert fraction(10.0, 100.0, 50.0) == 1.0


def test_worse_than_qualifying_is_clamped_to_zero_not_negative():
    assert fraction(500.0, 100.0, 50.0) == 0.0


def test_an_unmeasured_line_is_none_not_zero():
    """Zero means "measured, and at the bar". None means nobody measured."""
    assert fraction(None, 100.0, 50.0) is None


def test_a_degenerate_range_is_unscoreable_rather_than_dividing_by_zero():
    assert fraction(1.0, 5.0, 5.0) is None


def test_a_boolean_is_not_a_measurement():
    assert fraction(True, 0.0, 1.0) is None


# --------------------------------------------------------------------------
# K.8 — the two rules that remove any incentive to game the process
# --------------------------------------------------------------------------


def test_a_line_failing_reproduction_scores_zero_not_a_reduced_score():
    sub = _f2_submission()
    baseline = score(sub).lines["agentic_eval"].score
    assert baseline > 0

    sub.failed_reproduction = ("agentic_eval",)
    line = score(sub).lines["agentic_eval"]
    assert line.score == 0.0
    assert "failed reproduction" in line.note
    # The scorecard still shows what it would have been, so the cost is visible.
    assert "7.20" in line.note


def test_a_waived_line_scores_zero_and_stays_in_the_denominator():
    """A waiver protects qualification; it must never improve rank."""
    sub = _f2_submission()
    sub.waived = ("standard_eval",)
    card = score(sub)
    assert card.lines["standard_eval"].score == 0.0
    # Weight 10 is still counted against the submission, not removed.
    assert card.group_weight("quality") == 22
    assert "never improves rank" in card.lines["standard_eval"].note


def test_waiving_a_line_can_only_lower_the_overall():
    sub = _f2_submission()
    before = score(sub).overall
    sub.waived = ("standard_eval",)
    assert score(sub).overall < before


# --------------------------------------------------------------------------
# Unscoreable lines
# --------------------------------------------------------------------------


def test_a_submission_with_no_points_scores_zero_on_every_per_point_line():
    card = score(Submission())
    assert card.lines["ttft_p99"].score == 0.0
    assert "No graded points" in card.notes[0]


def test_unscoreable_lines_are_listed_separately_from_lines_that_scored_zero():
    """A line at qualifying and a line never measured both score 0, and differ."""
    sub = _f2_submission()
    sub.once["agentic_eval"] = 1.00  # exactly at qualifying -> a real zero
    sub.once["standard_eval"] = None  # never measured
    card = score(sub)
    assert card.lines["agentic_eval"].score == 0.0
    assert card.lines["standard_eval"].score == 0.0
    assert "agentic_eval" not in card.unscoreable
    assert "standard_eval" in card.unscoreable


def test_a_point_missing_its_target_cannot_be_scored_and_is_reported():
    points = _f3_points()
    points[0].target_ttft_ms = None
    line = score(Submission(points=points)).lines["ttft_p99"]
    assert line.points[0].fraction is None
    assert "1 of 10 points could not be scored" in line.note


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_markdown_carries_the_totals_and_the_group_subtotals():
    out = render_markdown(score(_f2_submission()))
    assert "Core total" in out and "Bonus total" in out and "Overall" in out
    assert "13.87" in out


def test_json_keeps_every_per_point_intermediate_for_audit():
    card = score(Submission(points=_f3_points()))
    data = to_dict(card)
    points = data["lines"]["ttft_p99"]["points"]
    assert len(points) == 10
    assert set(points[0]) >= {
        "concurrency",
        "input_length",
        "qualifying",
        "excellence",
        "measured",
        "fraction",
        "weight",
        "contribution",
    }


def test_concurrency_shares_are_the_published_pair():
    assert CONCURRENCY_SHARES == (0.25, 0.75)
