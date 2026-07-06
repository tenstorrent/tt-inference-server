#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for mode-aware, sample-count-aware reference selection.

Under --ci-mode (--limit-samples-mode ci-nightly) only a fixed subset of each
task runs, so the accuracy check must compare the subset score against a
subset-specific reference (EvalTaskScore.mode_reference_scores) using a
sample-count-aware integer floor. See evals.eval_config.resolve_eval_reference
and evals.eval_config.accept_eval_score.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.eval_config import (
    EvalTaskScore,
    ModeReferenceScore,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.run_reports import _resolve_eval_reference, collect_sample_counts
from workflows.workflow_types import EvalLimitMode


def _make_score(**overrides):
    base = dict(
        published_score=84.3,
        published_score_ref="card",
        score_func=lambda *a, **k: 0.0,
        gpu_reference_score=83.33,
        gpu_reference_score_ref="full (198)",
        tolerance=0.05,
    )
    base.update(overrides)
    return EvalTaskScore(**base)


# --- reference selection ----------------------------------------------------


def test_run_reports_reuses_shared_resolver():
    # run_reports aliases the shared resolver (single source of truth).
    assert _resolve_eval_reference is resolve_eval_reference


def test_no_limit_mode_falls_back_to_full_reference():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
        }
    )

    ref = resolve_eval_reference(score, None)

    assert ref["is_mode_ref"] is False
    assert ref["reference_score"] == 83.33
    assert ref["reference_ref"] == "full (198)"
    assert ref["tolerance"] == 0.05


def test_limit_mode_without_matching_entry_falls_back():
    score = _make_score(mode_reference_scores={})

    ref = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["is_mode_ref"] is False
    assert ref["reference_score"] == 83.33


def test_ci_nightly_uses_subset_reference_and_tolerance():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                72.5, ref="ci-nightly doc_ids 0-39", tolerance=0.10
            )
        }
    )

    ref = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["is_mode_ref"] is True
    assert ref["reference_score"] == 72.5
    assert ref["tolerance"] == 0.10
    assert "ci-nightly doc_ids 0-39" in ref["reference_ref"]
    assert "[CI_NIGHTLY subset]" in ref["reference_ref"]


def test_mode_reference_tolerance_none_falls_back_to_task_tolerance():
    score = _make_score(
        tolerance=0.07,
        mode_reference_scores={EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5)},
    )

    ref = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["tolerance"] == 0.07


# --- acceptance: sample-count-aware -----------------------------------------


def test_full_reference_uses_ratio_check():
    score = _make_score()
    ref = resolve_eval_reference(score, None)
    # 72.5 / 83.33 = 0.87 < 0.95 -> FAIL (full-set ratio, n ignored)
    assert accept_eval_score(ref, 72.5, n_total=40) is False
    # 80 / 83.33 = 0.96 >= 0.95 -> PASS
    assert accept_eval_score(ref, 80.0, n_total=40) is True


def test_gpqa_subset_passes_sample_aware_but_fails_full():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
        }
    )
    full = resolve_eval_reference(score, None)
    ci = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    # 70% on 40 -> 28 correct. Full ref FAIL; subset (threshold floor(40*0.725*0.9)=26) PASS.
    assert accept_eval_score(full, 70.0, n_total=40) is False
    assert accept_eval_score(ci, 70.0, n_total=40) is True


def test_tiny_subset_tolerates_one_flip_without_abs_margin():
    # 5-item agentic subset, reference 40% (=2/5), tol 10%.
    # threshold = floor(5 * 0.40 * 0.90) = floor(1.8) = 1 -> need >= 1/5.
    score = _make_score(
        gpu_reference_score=44.94,
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(40.0, tolerance=0.10)
        },
    )
    ci = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert accept_eval_score(ci, 40.0, n_total=5) is True  # 2/5
    assert accept_eval_score(ci, 20.0, n_total=5) is True  # 1/5 (one flip)
    assert accept_eval_score(ci, 0.0, n_total=5) is False  # 0/5


def test_mode_reference_without_sample_count_falls_back_to_ratio():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(40.0, tolerance=0.10)
        }
    )
    ci = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
    # No n_total -> ratio: 20/40 = 0.5 < 0.9 -> FAIL; 40/40 = 1.0 -> PASS.
    assert accept_eval_score(ci, 20.0, n_total=None) is False
    assert accept_eval_score(ci, 40.0, n_total=None) is True


def test_no_reference_returns_none():
    score = _make_score(gpu_reference_score=None)
    ref = resolve_eval_reference(score, None)
    assert accept_eval_score(ref, 50.0, n_total=40) is None


# --- v1 sample-count plumbing -----------------------------------------------


def test_collect_sample_counts_reads_effective(tmp_path):
    # The v1 report path must recover effective sample counts from lm-eval
    # result JSONs so the subset acceptance check is sample-count-aware
    # (matching the v2 scorer) instead of always falling back to the ratio.
    result_json = tmp_path / "results_0.json"
    result_json.write_text(
        json.dumps(
            {
                "results": {"r1_gpqa_diamond": {"exact_match,none": 0.7}},
                "n-samples": {
                    "r1_gpqa_diamond": {"original": 198, "effective": 40}
                },
            }
        )
    )

    counts = collect_sample_counts([str(result_json)])

    assert counts == {"r1_gpqa_diamond": 40}


def test_collect_sample_counts_ignores_files_without_field(tmp_path):
    no_field = tmp_path / "results_1.json"
    no_field.write_text(json.dumps({"results": {"foo": {"acc": 0.5}}}))

    assert collect_sample_counts([str(no_field)]) == {}
