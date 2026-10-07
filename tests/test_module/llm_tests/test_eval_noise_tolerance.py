# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Noise-aware acceptance for low-baseline accuracy tasks.

A task passes when score / reference >= 1 - tolerance (the ratio rule) OR when
its shortfall is within NOISE_Z binomial SEs of the reference. The examples
below are real Quetzal reports; the expected verdicts follow from the fixed
z = 1.96, not from the scores.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reference_config.evals.eval_utils import (
    score_multilevel_keys_mean,
    score_task_keys_mean,
    score_task_single_key,
)
from test_module._test_common import ReportCheckTypes
from test_module.llm_tests import llm_eval_tests as mod

_MOD = "test_module.llm_tests.llm_eval_tests"

IFEVAL_KEY = "prompt_level_strict_acc,none"
IFEVAL_N = 541
# Open LLM Leaderboard v2 MATH Lvl 5 subtask sizes (sum 1324).
MATH_HARD_SIZES = {
    "leaderboard_math_algebra_hard": 307,
    "leaderboard_math_counting_and_prob_hard": 123,
    "leaderboard_math_geometry_hard": 132,
    "leaderboard_math_intermediate_algebra_hard": 280,
    "leaderboard_math_num_theory_hard": 154,
    "leaderboard_math_prealgebra_hard": 193,
    "leaderboard_math_precalculus_hard": 135,
}
FULL_REF = {"reference_score": None, "tolerance": 0.05}


def _task(task_name, published, result_keys, score_func=score_task_single_key):
    return SimpleNamespace(
        task_name=task_name,
        score=SimpleNamespace(
            published_score=published,
            published_score_ref="ref://published",
            gpu_reference_score=None,
            tolerance=0.05,
            score_func=score_func,
            score_func_kwargs={"result_keys": list(result_keys), "unit": "percent"},
        ),
    )


def _ifeval(published, score, n=IFEVAL_N):
    task = _task("leaderboard_ifeval", published, [IFEVAL_KEY])
    results = {"leaderboard_ifeval": {IFEVAL_KEY: score / 100.0}}
    counts = {"leaderboard_ifeval": n} if n else {}
    return mod._grade_one(
        task, results, "leaderboard_ifeval", FULL_REF, sample_counts=counts
    )


def _math_hard(published, score, sizes=MATH_HARD_SIZES):
    keys = [(name, "exact_match,none") for name in MATH_HARD_SIZES]
    task = _task("leaderboard_math_hard", published, keys, score_multilevel_keys_mean)
    results = {"leaderboard_math_hard": {"exact_match,none": score / 100.0}}
    results.update({name: {"exact_match,none": score / 100.0} for name in sizes})
    return mod._grade_one(
        task, results, "leaderboard_math_hard", FULL_REF, sample_counts=dict(sizes)
    )


def test_z_is_fixed_at_95_percent():
    assert mod.NOISE_Z == 1.96


# --- high-baseline tasks are unchanged ---------------------------------------


@pytest.mark.parametrize(
    "published,score,n,expected",
    [
        # mmlu_pro (n = 12032): ratio 0.935 fails, SE 0.45 pt cannot rescue it.
        (57.77, 54.0, 12032, ReportCheckTypes.FAIL),
        # ratio 0.98 passes by the ratio rule as before.
        (57.77, 56.62, 12032, ReportCheckTypes.PASS),
        # ifeval at 80%: 5% of the reference (4.0 pt) exceeds z*SE (3.37 pt),
        # so the ratio rule alone decides on either side of 76.0.
        (80.0, 76.1, IFEVAL_N, ReportCheckTypes.PASS),
        (80.0, 75.9, IFEVAL_N, ReportCheckTypes.FAIL),
    ],
)
def test_high_baseline_verdicts_match_the_ratio_rule(published, score, n, expected):
    task = _task("t", published, ["exact_match,custom-extract"])
    results = {"t": {"exact_match,custom-extract": score / 100.0}}
    _, ratio, _, check, evidence = mod._grade_one(
        task, results, "t", FULL_REF, sample_counts={"t": n}
    )
    assert check == expected
    assert check == ReportCheckTypes.from_result(ratio >= 0.95)
    assert evidence["accuracy_rule"] in {mod.RULE_RATIO, mod.RULE_FAIL}


def test_noise_term_is_below_five_percent_for_high_baseline_ifeval():
    # Crossover: 0.05 * 100p == z * 100 * sqrt(p(1-p)/n)  =>  p/(1-p) = z^2/(n 0.05^2)
    odds = mod.NOISE_Z**2 / (IFEVAL_N * 0.05**2)
    p_cross = odds / (1 + odds)
    assert 0.73 < p_cross < 0.75
    for ref in (75.0, 85.0, 95.0):
        p = ref / 100
        assert mod.NOISE_Z * 100 * math.sqrt(p * (1 - p) / IFEVAL_N) < 0.05 * ref


# --- the reported low-baseline cases -----------------------------------------


def test_tinyllama_math_hard_passes_within_noise():
    # TinyLlama-1.1B-Chat: 1.01 vs 1.51, ratio 0.67.
    score, ratio, _, check, evidence = _math_hard(1.51, 1.0131960628369427)
    assert ratio == pytest.approx(0.671, abs=1e-3)
    n_eff = 49 / sum(1 / n for n in MATH_HARD_SIZES.values())
    assert evidence["noise_n"] == pytest.approx(n_eff)
    assert 1177 < n_eff < 1178
    se = 100 * math.sqrt(0.0151 * 0.9849 / n_eff)
    assert evidence["noise_se"] == pytest.approx(se)
    assert 1.51 - score <= 1.96 * se  # 0.50 <= 0.70
    assert check == ReportCheckTypes.PASS
    assert evidence["accuracy_rule"] == mod.RULE_WITHIN_NOISE
    assert evidence["noise_z"] == 1.96


def test_llama32_3b_ifeval_rerun_passes_within_noise():
    # 8.32 vs 9.24 on 541 prompts: ratio 0.90, about 0.74 SE short.
    _, _, _, check, evidence = _ifeval(9.24, 8.317929759704251)
    assert check == ReportCheckTypes.PASS
    assert evidence["accuracy_rule"] == mod.RULE_WITHIN_NOISE
    assert evidence["noise_n"] == IFEVAL_N


def test_pythia_160m_ifeval_still_fails():
    # 8.87 vs 12.57 on n = 541: SE 1.43 pt, z*SE 2.79 pt, shortfall 3.70 pt.
    _, _, _, check, evidence = _ifeval(12.57, 8.872458410351202)
    assert evidence["noise_se"] == pytest.approx(1.4253, abs=1e-3)
    assert 12.57 - 8.87 > 1.96 * evidence["noise_se"]
    assert check == ReportCheckTypes.FAIL
    assert evidence["accuracy_rule"] == mod.RULE_FAIL


def test_pythia_160m_math_hard_still_fails():
    # 0.37 vs 0.92: shortfall 0.55 pt > z*SE 0.54 pt.
    _, _, _, check, evidence = _math_hard(0.92, 0.3713596780139358)
    assert check == ReportCheckTypes.FAIL
    assert evidence["accuracy_rule"] == mod.RULE_FAIL


# --- edge cases --------------------------------------------------------------


def test_boundary_shortfall_equal_to_z_se_passes():
    p, n = 0.10, 400
    se = 100 * math.sqrt(p * (1 - p) / n)
    passed, rule, got_se = mod.noise_aware_check(False, 10.0 - 1.96 * se, 10.0, n)
    assert got_se == pytest.approx(se)
    assert (passed, rule) == (True, mod.RULE_WITHIN_NOISE)
    passed, rule, _ = mod.noise_aware_check(False, 10.0 - 1.96 * se - 1e-6, 10.0, n)
    assert (passed, rule) == (False, mod.RULE_FAIL)


def test_unknown_sample_count_falls_back_to_ratio_rule():
    _, _, _, check, evidence = _ifeval(9.24, 8.32, n=None)
    assert check == ReportCheckTypes.FAIL
    assert evidence == {
        "accuracy_rule": mod.RULE_FAIL,
        "noise_n": None,
        "noise_se": None,
        "noise_z": None,
    }


def test_missing_subtask_count_disables_the_rule():
    sizes = dict(MATH_HARD_SIZES)
    del sizes["leaderboard_math_geometry_hard"]
    _, _, _, check, evidence = _math_hard(1.51, 1.0131960628369427, sizes=sizes)
    assert check == ReportCheckTypes.FAIL
    assert evidence["noise_n"] is None


@pytest.mark.parametrize(
    "result_keys",
    [
        ["score,none"],  # LongBench F1 / ROUGE / accuracy mix
        ["anls,none"],
        ["inst_level_strict_acc,none"],  # unit is the instruction, n counts prompts
    ],
)
def test_non_binomial_metrics_are_excluded(result_keys):
    task = _task("t", 7.67, result_keys)
    results = {"t": {result_keys[0]: 0.0604}}
    _, _, _, check, evidence = mod._grade_one(
        task, results, "t", FULL_REF, sample_counts={"t": 1000}
    )
    assert check == ReportCheckTypes.FAIL
    assert evidence["accuracy_rule"] == mod.RULE_FAIL
    assert evidence["noise_n"] is None


def test_mean_of_two_metrics_on_one_entry_is_excluded():
    keys = [IFEVAL_KEY, "prompt_level_loose_acc,none"]
    task = _task("leaderboard_ifeval", 9.24, keys, score_task_keys_mean)
    results = {"leaderboard_ifeval": {k: 0.0832 for k in keys}}
    _, _, _, check, evidence = mod._grade_one(
        task,
        results,
        "leaderboard_ifeval",
        FULL_REF,
        sample_counts={"leaderboard_ifeval": IFEVAL_N},
    )
    assert check == ReportCheckTypes.FAIL
    assert evidence["noise_n"] is None


def test_score_above_reference_passes_by_ratio():
    _, _, _, check, evidence = _ifeval(9.24, 9.43)
    assert check == ReportCheckTypes.PASS
    assert evidence["accuracy_rule"] == mod.RULE_RATIO
    assert evidence["noise_se"] is not None  # still recorded for the audit


def test_no_reference_is_na():
    task = _task("t", None, ["acc,none"])
    _, _, _, check, evidence = mod._grade_one(
        task, {"t": {"acc,none": 0.1}}, "t", FULL_REF, sample_counts={"t": 100}
    )
    assert check == ReportCheckTypes.NA
    assert evidence["accuracy_rule"] == mod.RULE_NA


@pytest.mark.parametrize("reference", [100.0, 150.0])
def test_degenerate_or_out_of_range_reference_gets_no_noise_band(reference):
    passed, rule, se = mod.noise_aware_check(False, reference - 0.5, reference, 1000)
    assert (passed, rule) == (False, mod.RULE_FAIL)
    assert se == (0.0 if reference == 100.0 else None)


def test_zero_score_is_a_real_failure():
    # A model that answers nothing is not "within noise" of a 5% reference.
    passed, rule, _ = mod.noise_aware_check(False, 0.0, 5.0, 541)
    assert (passed, rule) == (False, mod.RULE_FAIL)


def test_gpu_reference_path_uses_the_same_rule():
    # gpu_reference_score drives the check when present (accept_eval_score).
    task = _task("leaderboard_ifeval", None, [IFEVAL_KEY])
    ref = {"reference_score": 9.24, "tolerance": 0.05, "is_subset_reference": False}
    results = {"leaderboard_ifeval": {IFEVAL_KEY: 0.0832}}
    _, _, _, check, evidence = mod._grade_one(
        task,
        results,
        "leaderboard_ifeval",
        ref,
        sample_counts={"leaderboard_ifeval": IFEVAL_N},
    )
    assert check == ReportCheckTypes.PASS
    assert evidence["accuracy_rule"] == mod.RULE_WITHIN_NOISE


def test_score_one_keeps_its_four_tuple_and_uses_n_total():
    task = _task("leaderboard_ifeval", 9.24, [IFEVAL_KEY])
    results = {"leaderboard_ifeval": {IFEVAL_KEY: 0.0832}}
    out = mod._score_one(task, results, "leaderboard_ifeval", FULL_REF, n_total=541)
    assert len(out) == 4
    assert out[3] == ReportCheckTypes.PASS


# --- evidence reaches the report block ---------------------------------------


def _ctx():
    ctx = MagicMock()
    ctx.runtime_config.limit_samples_mode = None
    return ctx


def test_block_records_rule_se_and_n():
    task = _task("leaderboard_ifeval", 9.24, [IFEVAL_KEY])
    results = {"leaderboard_ifeval": {IFEVAL_KEY: 0.0832}}
    with patch(f"{_MOD}.block_id", return_value=""):
        (block,) = mod.blocks_for_task(
            _ctx(), task, results, sample_counts={"leaderboard_ifeval": IFEVAL_N}
        )
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS
    assert block.data["accuracy_rule"] == "within_noise"
    assert block.data["noise_n"] == IFEVAL_N
    assert block.data["noise_se"] == pytest.approx(1.2453, abs=1e-3)
    assert block.data["noise_z"] == 1.96


# --- sample counts from lm-eval output ---------------------------------------


def test_loader_reads_subtask_and_group_counts(tmp_path):
    names = list(MATH_HARD_SIZES)
    payload = {
        "results": {
            "leaderboard_math_hard": {"exact_match,none": 0.01, "alias": "math"},
            **{n: {"exact_match,none": 0.01, "alias": n} for n in names},
        },
        "group_subtasks": {"leaderboard_math_hard": names, **{n: [] for n in names}},
        "configs": {n: {"task": n, "dataset_path": "math"} for n in names},
        "n-samples": {
            n: {"original": size, "effective": size}
            for n, size in MATH_HARD_SIZES.items()
        },
    }
    path = tmp_path / "results_2026-10-07T00-00-00.json"
    path.write_text(json.dumps(payload))
    _, counts = mod.load_eval_results([str(path)])
    assert counts["leaderboard_math_hard"] == 1324
    for name, size in MATH_HARD_SIZES.items():
        assert counts[name] == size


def test_group_count_is_omitted_when_a_leaf_count_is_missing(tmp_path):
    payload = {
        "results": {
            "mmlu_pro": {"exact_match,custom-extract": 0.1, "alias": "mmlu_pro"},
            "mmlu_pro_law": {"exact_match,custom-extract": 0.1},
            "mmlu_pro_math": {"exact_match,custom-extract": 0.1},
        },
        "group_subtasks": {"mmlu_pro": ["mmlu_pro_law", "mmlu_pro_math"]},
        "configs": {
            n: {"task": n, "dataset_path": "mmlu_pro"}
            for n in ["mmlu_pro_law", "mmlu_pro_math"]
        },
        "n-samples": {"mmlu_pro_law": {"original": 1101, "effective": 1101}},
    }
    path = tmp_path / "results_x.json"
    path.write_text(json.dumps(payload))
    _, counts = mod.load_eval_results([str(path)])
    assert "mmlu_pro" not in counts
    assert counts["mmlu_pro_law"] == 1101
