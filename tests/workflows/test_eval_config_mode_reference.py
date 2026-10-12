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

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reference_config.evals.eval_config import (
    EvalTaskScore,
    ModeReferenceScore,
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
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


def test_no_limit_mode_falls_back_to_full_reference():
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(72.5, tolerance=0.10)
        }
    )

    ref = resolve_eval_reference(score, None)

    assert ref["is_subset_reference"] is False
    assert ref["reference_score"] == 83.33
    assert ref["reference_ref"] == "full (198)"
    assert ref["tolerance"] == 0.05


def test_limit_mode_without_matching_entry_falls_back():
    score = _make_score(mode_reference_scores={})

    ref = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)

    assert ref["is_subset_reference"] is False
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

    assert ref["is_subset_reference"] is True
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


def test_diffusiongemma_gpqa_requires_more_than_67_percent_in_all_modes():
    tasks = _eval_config_map["google/diffusiongemma-26B-A4B-it"].tasks
    score = next(
        task.score for task in tasks if task.task_name == "gpqa_diamond_cot_zeroshot"
    )

    assert score.gpu_reference_score == 70.0
    assert score.tolerance == 3 / 70

    full = resolve_eval_reference(score, None)
    assert accept_eval_score(full, 132 / 198 * 100, n_total=198) is False
    assert accept_eval_score(full, 133 / 198 * 100, n_total=198) is True

    nightly = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
    assert nightly["is_subset_reference"] is False
    assert accept_eval_score(nightly, 60.0, n_total=10) is False
    assert accept_eval_score(nightly, 70.0, n_total=10) is True


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


# --- gemma-4 release entries: deterministic subset gate + ci-long preset ------


def _gemma4_task(model, task_name):
    return next(t for t in _eval_config_map[model].tasks if t.task_name == task_name)


def test_gemma4_12b_gpqa_subset_gate_is_the_measured_deterministic_score():
    """The 40-question ci-nightly subset is deterministic for a fixed seed on TT
    (26/40 in three same-seed runs), so it is gated as a regression reference at
    its measured 65.0 with a 5% band: 24/40 passes, 23/40 fails."""
    score = _gemma4_task("google/gemma-4-12B-it", "r1_gpqa_diamond").score
    ref = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
    assert ref["is_subset_reference"] is True
    assert ref["reference_score"] == 65.0
    assert ref["tolerance"] == 0.05
    assert accept_eval_score(ref, 26 / 40 * 100, n_total=40) is True
    assert accept_eval_score(ref, 24 / 40 * 100, n_total=40) is True
    assert accept_eval_score(ref, 23 / 40 * 100, n_total=40) is False
    # The full set is still judged against the published score (ratio check).
    full = resolve_eval_reference(score, EvalLimitMode.CI_LONG)
    assert full["is_subset_reference"] is False


def test_gemma4_release_models_define_the_ci_long_preset():
    """ci-long = full gpqa (no CI_LONG limit), the nightly mmlu_pro subset and the
    nightly agentic task lists, so the preset proves accuracy without running the
    12k-question mmlu_pro or 500-task swe-bench sets."""
    for model in ("google/gemma-4-31B-it", "google/gemma-4-12B-it"):
        gpqa = _gemma4_task(model, "r1_gpqa_diamond")
        assert EvalLimitMode.CI_LONG not in gpqa.limit_samples_map
        mmlu = _gemma4_task(model, "mmlu_pro")
        assert (
            mmlu.limit_samples_map[EvalLimitMode.CI_LONG]
            == mmlu.limit_samples_map[EvalLimitMode.CI_NIGHTLY]
            == 0.07
        )
        agentic = [t for t in _eval_config_map[model].tasks if t.agentic_eval_config]
        for task in agentic:
            names = task.agentic_eval_config.task_names_map
            assert names[EvalLimitMode.CI_LONG] == names[EvalLimitMode.CI_NIGHTLY]
            assert len(names[EvalLimitMode.CI_LONG]) == 5
    # The 12B entry carries the two informational agentic tasks.
    assert {
        t.task_name
        for t in _eval_config_map["google/gemma-4-12B-it"].tasks
        if t.agentic_eval_config
    } == {
        "terminal_bench_2",
        "swe_bench_verified",
    }
