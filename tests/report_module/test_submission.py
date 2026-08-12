# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.submission``.

The module's job is transcription, so the tests are mostly about not inventing
data: a value that was not measured must not arrive at the scorecard as a number.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from report_module.scorecard import score, submission_from_dict
from report_module.submission import (
    SubmissionError,
    build,
    find_report,
    graded_points,
    merge_points,
    run_to_run_cov,
    scaling_exponents,
    validate,
)


def _point(concurrency: int, isl: int, *, ttft: float = 100.0, mean: float = 100.0):
    """A benchmark block shaped like the runners emit."""
    return {
        "kind": "benchmarks",
        "title": f"AIPerf Benchmark Targets — ISL {isl}, concurrency {concurrency}",
        "data": {
            "concurrency": concurrency,
            "input_sequence_length": isl,
            "mean_ttft_ms": mean,
            "p50_ttft": ttft,
            "p90_ttft": ttft * 1.1,
            "p99_ttft": ttft * 1.2,
            "prefill_throughput_tok_s": 5000.0,
            "ttft_tail_ratio": 1.2,
            "tput_user": 35.0,
            "tps_decode_throughput": 35.0,
            "ttft_scaling_exponent": 0.8,
            "error_request_count": 0,
            "target_checks": {
                "functional": {"ttft": ttft * 10},
                "target": {"ttft": 90.0, "tput_user": 35.0, "tput": 35.0},
            },
        },
    }


def _report(*blocks: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metadata": {"model_name": "google/gemma-4-31B-it"},
        "sections": list(blocks),
    }


def _sweep(mean: float = 100.0) -> Dict[str, Any]:
    """A full two-corner sweep: 5 input lengths at concurrency 1 and 32."""
    return _report(
        *[
            _point(c, isl, mean=mean)
            for c in (1, 32)
            for isl in (128, 1024, 8192, 32768, 131072)
        ]
    )


# --- extraction ------------------------------------------------------------


def test_targets_come_from_the_points_own_target_tier():
    (point,) = graded_points(_report(_point(1, 128)))
    assert point["target_ttft_ms"] == 90.0
    assert point["target_tput_user"] == 35.0
    assert point["target_decode_throughput"] == 35.0
    assert point["concurrency"] == 1 and point["input_length"] == 128


def test_ungraded_points_are_skipped_not_defaulted():
    """A point with no targets has no qualifying value; inventing one would score
    something nobody agreed to grade."""
    block = _point(1, 128)
    del block["data"]["target_checks"]
    assert graded_points(_report(block)) == []


def test_a_block_with_targets_but_no_operating_point_is_an_error():
    block = _point(1, 128)
    del block["data"]["concurrency"]
    with pytest.raises(SubmissionError, match="cannot be placed in the sweep"):
        graded_points(_report(block))


def test_scaling_exponents_are_keyed_by_concurrency():
    assert scaling_exponents(_sweep()) == {1: 0.8, 32: 0.8}


# --- merging across runs ---------------------------------------------------


def test_two_corner_runs_merge_into_one_sweep():
    """A Partner may run each concurrency corner separately."""
    corner1 = _report(*[_point(1, isl) for isl in (128, 1024)])
    corner32 = _report(*[_point(32, isl) for isl in (128, 1024)])
    points = merge_points([corner1, corner32])
    assert sorted({p["concurrency"] for p in points}) == [1, 32]
    assert len(points) == 4


def test_repeat_runs_do_not_duplicate_points_and_the_first_run_wins():
    """Scored figures must come from one consistent run, not a mix of runs."""
    points = merge_points([_sweep(mean=100.0), _sweep(mean=999.0)])
    assert len(points) == 10
    assert {p["p50_ttft"] for p in points} == {100.0}


# --- run-to-run variation --------------------------------------------------


def test_cov_is_none_when_nothing_was_measured_twice():
    """One measurement is not a variation. Reporting 0.0 would award full marks
    for having done nothing."""
    value, evidence = run_to_run_cov([_sweep()])
    assert value is None and evidence == {}


def test_cov_is_none_when_runs_cover_different_corners():
    """Splitting a sweep across runs must not manufacture a stable-looking score."""
    corner1 = _report(_point(1, 128))
    corner32 = _report(_point(32, 128))
    value, _ = run_to_run_cov([corner1, corner32])
    assert value is None


def test_cov_is_computed_from_repeats_and_the_worst_point_governs():
    stable = _report(_point(1, 128, mean=100.0), _point(32, 128, mean=100.0))
    varied = _report(_point(1, 128, mean=101.0), _point(32, 128, mean=140.0))
    value, evidence = run_to_run_cov([stable, varied])
    assert set(evidence) == {"conc1_isl128", "conc32_isl128"}
    # The unstable point, not the average of the two.
    assert value == max(evidence.values())
    assert value == pytest.approx(evidence["conc32_isl128"])


# --- validation ------------------------------------------------------------


def test_validate_rejects_a_sweep_that_is_not_two_corners():
    problems = validate(graded_points(_report(_point(1, 128))))
    assert any("exactly two" in p for p in problems)
    assert any("ONLY_BENCHMARK_TARGETS=1" in p for p in problems)


def test_validate_rejects_an_empty_sweep():
    assert any("No graded points" in p for p in validate([]))


def test_validate_names_the_missing_field():
    block = _point(1, 128)
    block["data"]["tput_user"] = None
    points = graded_points(_report(block)) + graded_points(_report(_point(32, 128)))
    assert any("tput_user" in p for p in validate(points))


# --- end to end ------------------------------------------------------------


def _write(tmp_path, name, report):
    d = tmp_path / name
    d.mkdir()
    (d / "report_data_x.json").write_text(json.dumps(report))
    return d


def test_find_report_errors_with_an_actionable_message(tmp_path):
    with pytest.raises(SubmissionError, match="Point --run at the directory"):
        find_report(tmp_path)


def test_build_produces_a_document_the_scorecard_can_score(tmp_path):
    run = _write(tmp_path, "run1", _sweep())
    doc = build([run], partner="Acme", once={"agentic_eval": 1.05})
    assert doc["partner"] == "Acme"
    assert doc["model"] == "google/gemma-4-31B-it"
    assert len(doc["points"]) == 10
    assert doc["scaling_exponents"] == {"1": 0.8, "32": 0.8}
    # The real contract: it round-trips through the scorer without adjustment.
    card = score(submission_from_dict(doc))
    assert card.lines["ttft_p99"].fraction is not None
    assert card.lines["agentic_eval"].fraction is not None


def test_build_records_evidence_for_supplied_lines(tmp_path):
    run = _write(tmp_path, "run1", _sweep())
    doc = build([run], partner="Acme")
    assert doc["_evidence"]["error_request_count_per_point"] == [0] * 10
    assert doc["_evidence"]["runs_used_for_cov"] == 1


def test_build_states_how_cov_was_derived_and_honours_an_override(tmp_path):
    runs = [
        _write(tmp_path, "run1", _sweep(mean=100.0)),
        _write(tmp_path, "run2", _sweep(mean=110.0)),
    ]
    doc = build(runs, partner="Acme")
    assert doc["once"]["run_to_run_cov"] > 0
    assert "stdev / mean" in doc["_evidence"]["cov_definition"]

    overridden = build(runs, partner="Acme", cov_override=0.01)
    assert overridden["once"]["run_to_run_cov"] == 0.01
    assert overridden["_evidence"]["cov_definition"] == "supplied"


def test_build_refuses_a_sweep_the_scorecard_could_not_weight(tmp_path):
    run = _write(tmp_path, "run1", _report(_point(1, 128), _point(1, 1024)))
    with pytest.raises(SubmissionError, match="exactly two"):
        build([run], partner="Acme")


def test_build_requires_at_least_one_run():
    with pytest.raises(SubmissionError, match="At least one --run"):
        build([], partner="Acme")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
