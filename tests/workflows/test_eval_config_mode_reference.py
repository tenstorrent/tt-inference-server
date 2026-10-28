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

from llm_module.eval_configs import _apply_profile_request_body_overrides
from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_utils import score_task_single_key
from test_module._test_common import ReportCheckTypes
from test_module.llm_tests import llm_eval_tests
from reference_config.evals.eval_config import (
    EvalTaskScore,
    ModeReferenceScore,
    _eval_config_map,
    accept_eval_score,
    resolve_eval_reference,
)
from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path
from workflows.workflow_types import DeviceTypes, EvalLimitMode, WorkflowVenvType


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


def test_mode_reference_min_correct_holds_when_scored_count_shrinks():
    # 20% of 5 is 1 solved, but with 1 errored trial n = 4 and floor(0.8) = 0.
    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(20.0, tolerance=0.0)
        }
    )
    floor_only = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
    assert accept_eval_score(floor_only, 0.0, n_total=4) is True

    score = _make_score(
        mode_reference_scores={
            EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                20.0, tolerance=0.0, min_correct=1
            )
        }
    )
    ci = resolve_eval_reference(score, EvalLimitMode.CI_NIGHTLY)
    assert ci["min_correct"] == 1
    assert accept_eval_score(ci, 0.0, n_total=4) is False  # 0 of 4 scored
    assert accept_eval_score(ci, 25.0, n_total=4) is True  # 1 of 4 scored
    assert accept_eval_score(ci, 0.0, n_total=5) is False
    assert accept_eval_score(ci, 20.0, n_total=5) is True


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


# --- gemma-4 release entries: the model-card presets --------------------------


def _gemma4_task(model, task_name):
    return next(t for t in _eval_config_map[model].tasks if t.task_name == task_name)


GEMMA4_RELEASE_MODELS = (
    "google/gemma-4-31B-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-12B-it",
)


def test_gemma4_gpqa_runs_the_full_set_in_every_preset():
    """Both presets run all 198 questions, graded against the published score and
    the full-set GPU reference; no preset carries a subset reference."""
    for model in GEMMA4_RELEASE_MODELS:
        task = _gemma4_task(model, "r1_gpqa_diamond")
        assert EvalLimitMode.CI_NIGHTLY not in task.limit_samples_map
        assert EvalLimitMode.CI_LONG not in task.limit_samples_map
        assert not task.score.mode_reference_scores
        assert task.score.gpu_reference_score is not None
        ref = resolve_eval_reference(task.score, EvalLimitMode.CI_NIGHTLY)
        assert ref["is_subset_reference"] is False
        # The reference is a thinking-mode run; the task pins the mode per
        # request rather than relying on the serving default.
        # The 31B / 12B tasks pin thinking on themselves; the 26B task is shared by
        # two implementations, so its pin lives in each profile's override.
        assert task.thinking is (None if model == "google/gemma-4-26B-A4B-it" else True)


def test_gemma4_release_models_run_the_model_card_evals_in_both_presets():
    """The entries carry exactly the model-card evals, every one a blocking gate
    (priority must, graded against the same-protocol GPU reference where one exists),
    and ci-nightly and ci-long run the same set at the same sizes: GPQA Diamond in
    full, one 24-task Terminal-Bench 2.0 wave, the whole LiveCodeBench v6 window and
    865 MMMU-Pro questions."""
    for model in GEMMA4_RELEASE_MODELS:
        tasks = _eval_config_map[model].tasks
        assert [t.task_name for t in tasks] == [
            "r1_gpqa_diamond",
            "terminal_bench_2",
            "livecodebench",
            "mmmu_pro_standard_cot",
            # The 12B also serves audio: the model card's FLEURS (WER) joins its set.
            *(["fleurs_en"] if model == "google/gemma-4-12B-it" else []),
        ]
        assert all((t.priority or "must") == "must" for t in tasks)
        terminal = _gemma4_task(model, "terminal_bench_2").agentic_eval_config
        names = terminal.task_names_map
        assert names[EvalLimitMode.CI_NIGHTLY] == names[EvalLimitMode.CI_LONG]
        assert len(set(names[EvalLimitMode.CI_LONG])) == 24
        assert terminal.n_concurrent_trials == 24
        assert terminal.agent_kwargs["llm_kwargs"]["extra_body"][
            "chat_template_kwargs"
        ] == {"enable_thinking": True}
        lcb = _gemma4_task(model, "livecodebench")
        assert EvalLimitMode.CI_NIGHTLY not in lcb.limit_samples_map
        assert EvalLimitMode.CI_LONG not in lcb.limit_samples_map
        assert lcb.thinking is True
        mmmu = _gemma4_task(model, "mmmu_pro_standard_cot")
        # Stock lmms-eval client; thinking off per request (the protocol of the
        # GPU reference) through the request-overrides wrapper.
        assert mmmu.eval_class == "openai_compatible"
        assert mmmu.workflow_venv_type == WorkflowVenvType.EVALS_VISION
        assert mmmu.thinking is False and not mmmu.request_body
        assert mmmu.batch_size == 32
        assert mmmu.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == 0.5
        assert mmmu.limit_samples_map[EvalLimitMode.CI_LONG] == 0.5
        refs = mmmu.score.mode_reference_scores
        # Same-protocol GPU reference measured on the 865-doc subset; both
        # presets share it.
        assert set(refs) == {EvalLimitMode.CI_NIGHTLY, EvalLimitMode.CI_LONG}
        assert refs[EvalLimitMode.CI_NIGHTLY].score == refs[EvalLimitMode.CI_LONG].score
        ref = resolve_eval_reference(mmmu.score, EvalLimitMode.CI_LONG)
        assert ref["is_subset_reference"] is True


def test_gemma4_12b_fleurs_is_the_audio_gate_scored_as_100_minus_wer():
    """FLEURS en_us runs on the stock lmms-eval client from the vision venv (the
    wrapper encodes the clips), thinking off per request, the full 647 utterances
    in both presets, and is graded as 100 - WER (the model card's 0.069 WER)."""
    fleurs = _gemma4_task("google/gemma-4-12B-it", "fleurs_en")
    assert fleurs.eval_class == "openai_compatible"
    assert fleurs.workflow_venv_type == WorkflowVenvType.EVALS_VISION
    assert fleurs.modalities == ["audio"]
    assert fleurs.thinking is False and not fleurs.request_body
    assert EvalLimitMode.CI_NIGHTLY not in fleurs.limit_samples_map
    assert EvalLimitMode.CI_LONG not in fleurs.limit_samples_map
    assert not fleurs.score.mode_reference_scores
    assert fleurs.score.published_score == 100 - 6.9
    # The gate inverts a "WER"-unit task itself (100 - WER), so the task returns the
    # raw key: graded through the harness, a 6.35 WER scores 93.65 and passes.
    assert fleurs.score.score_func is score_task_single_key
    assert fleurs.score.score_func_kwargs["unit"] == "WER"
    score, ratio_to_published, _, check, _ = llm_eval_tests._grade_one(
        fleurs,
        {"fleurs_en": {"wer,none": 6.35}},
        "fleurs_en",
        {"reference_score": None, "tolerance": 0.05},
        n_total=647,
    )
    assert score == 100 - 6.35
    assert ratio_to_published > 1.0 and check == ReportCheckTypes.PASS
    # Other release models do not serve audio and carry no audio task.
    for model in ("google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it"):
        assert "fleurs_en" not in {t.task_name for t in _eval_config_map[model].tasks}


def _gemma4_multimodal_release_specs():
    """The six tt-transformers Blackhole QB2 / Galaxy blocks of the release entries."""
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
    )
    return [
        spec
        for template in templates
        if template.weights[0] in GEMMA4_RELEASE_MODELS
        for spec in template.expand_to_specs()
        if spec.impl.impl_id == "tt_transformers"
        and spec.device_type in (DeviceTypes.P300X2, DeviceTypes.BLACKHOLE_GALAXY)
    ]


def test_gemma4_evals_pin_thinking_per_request_over_the_serving_default(
    monkeypatch, tmp_path
):
    """The multimodal serving blocks keep thinking on by default; the mode an
    eval is graded in is set by its own request: the text evals thinking on
    (their references), MMMU-Pro thinking off (its GPU reference's protocol),
    rendered through the request-overrides wrapper (--harness lmms-eval for
    the lmms-eval task)."""
    monkeypatch.setenv("OPENAI_API_BASE", "restore-me")
    specs = _gemma4_multimodal_release_specs()
    assert len(specs) == 6
    for spec in specs:
        assert (
            spec.device_model_spec.vllm_args["default-chat-template-kwargs"]
            == '{"enable_thinking": true}'
        ), spec.model_id
        tasks = {
            t.task_name: t
            for t in _apply_profile_request_body_overrides(
                _eval_config_map[spec.hf_model_repo].tasks, spec
            )
        }
        mmmu = build_eval_command(
            tasks["mmmu_pro_standard_cot"], spec, spec.device_type, tmp_path, 8000
        )
        sep = mmmu.index("--")
        assert mmmu[1].endswith("llm_module/lm_eval_request_overrides.py")
        assert mmmu[2:sep] == [
            "--request-body",
            '{"chat_template_kwargs": {"enable_thinking": false}}',
            "--harness",
            "lmms-eval",
        ]
        assert mmmu[sep + 1 : sep + 5] == [
            "--tasks",
            "mmmu_pro_standard_cot",
            "--model",
            "openai_compatible",
        ]
        for name in ("r1_gpqa_diamond", "livecodebench"):
            cmd = build_eval_command(
                tasks[name], spec, spec.device_type, tmp_path, 8000
            )
            assert cmd[2 : cmd.index("--")] == [
                "--request-body",
                '{"chat_template_kwargs": {"enable_thinking": true}}',
            ], name
        tb = tasks["terminal_bench_2"].agentic_eval_config.agent_kwargs
        assert tb["llm_kwargs"]["extra_body"]["chat_template_kwargs"] == {
            "enable_thinking": True
        }
