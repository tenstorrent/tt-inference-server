#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for mode-aware reference selection in the evals release report.

Under --ci-mode (--limit-samples-mode ci-nightly) only a fixed subset of each
task runs, so the accuracy check must compare the subset score against a
subset-specific reference (EvalTaskScore.mode_reference_scores) instead of the
full-dataset gpu_reference_score. See evals.eval_config.ModeReferenceScore and
workflows.run_reports._resolve_eval_reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.eval_config import EvalTaskScore, ModeReferenceScore
from workflows.run_reports import _resolve_eval_reference
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


def test_no_limit_mode_falls_back_to_full_reference():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
        }
    )

    ref = _resolve_eval_reference(score, None)

    assert ref["is_mode_ref"] is False
    assert ref["reference_score"] == 83.33
    assert ref["reference_ref"] == "full (198)"
    assert ref["tolerance"] == 0.05
    assert ref["abs_margin"] is None


def test_limit_mode_without_matching_entry_falls_back():
    # Task has no CI_NIGHTLY reference -> full-set baseline is used.
    score = _make_score(mode_reference_scores={})

    ref = _resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

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

    ref = _resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["is_mode_ref"] is True
    assert ref["reference_score"] == 72.5
    # Subset-specific tolerance overrides the task default.
    assert ref["tolerance"] == 0.10
    assert ref["abs_margin"] is None
    assert "ci-nightly doc_ids 0-39" in ref["reference_ref"]
    assert "[CI_NIGHTLY subset]" in ref["reference_ref"]


def test_mode_reference_tolerance_none_falls_back_to_task_tolerance():
    score = _make_score(
        tolerance=0.07,
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5)
        },
    )

    ref = _resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["tolerance"] == 0.07


def test_abs_margin_is_propagated_for_tiny_subsets():
    score = _make_score(
        gpu_reference_score=44.94,
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                40.0, ref="5 tasks", abs_margin=20.0
            )
        },
    )

    ref = _resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["is_mode_ref"] is True
    assert ref["reference_score"] == 40.0
    assert ref["abs_margin"] == 20.0


def _ratio_pass(score, ref):
    """Mirror the run_reports ratio-based PASS condition."""
    return (score / ref["reference_score"]) >= (1.0 - ref["tolerance"])


def _abs_margin_pass(score, ref):
    """Mirror the run_reports absolute-margin PASS condition."""
    return score >= ref["reference_score"] - ref["abs_margin"]


def test_subset_score_passes_against_subset_but_fails_full_reference():
    # GPQA: 72.5 on the 40-question CI subset. Full ref 83.33 would FAIL.
    score_obj = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
        }
    )
    observed = 72.5

    full_ref = _resolve_eval_reference(score_obj, None)
    assert _ratio_pass(observed, full_ref) is False  # 72.5/83.33 = 0.87 < 0.95

    mode_ref = _resolve_eval_reference(score_obj, EvalLimitMode.CI_NIGHTLY)
    assert _ratio_pass(observed, mode_ref) is True  # 72.5/72.5 = 1.0 >= 0.90


def test_abs_margin_pass_and_fail_boundaries():
    score_obj = _make_score(
        gpu_reference_score=44.94,
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(40.0, abs_margin=20.0)
        },
    )
    mode_ref = _resolve_eval_reference(score_obj, EvalLimitMode.CI_NIGHTLY)

    # One agentic task flips (40 -> 20): still within the 20pt margin -> PASS.
    assert _abs_margin_pass(20.0, mode_ref) is True
    # Two tasks flip (40 -> 0): below ref - margin -> FAIL.
    assert _abs_margin_pass(0.0, mode_ref) is False
