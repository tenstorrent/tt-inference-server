# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the TTS WER-quality eval runner.

Focuses on the pure decision logic (WER -> score, WER -> accuracy_check,
dependency preflight) and the shape of the emitted eval Block. The heavy
transcription pipeline (``TTSQualityTest``) is exercised on hardware, not
here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import ReportCheckTypes
from test_module.eval_tests import tts_eval_tests as tts
from test_module.eval_tests.tts_eval_tests import (
    DEFAULT_WER_THRESHOLD,
    _intelligibility_score,
    _missing_quality_deps,
    _tts_eval_block,
    _wer_accuracy_check,
)


def _task():
    score = SimpleNamespace(
        tolerance=0.05,
        published_score=14.0,
        published_score_ref="ref-url",
    )
    return SimpleNamespace(task_name="tts_generation", score=score)


def _ctx():
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="speecht5_tts"),
        device=SimpleNamespace(name="N150"),
    )


class TestIntelligibilityScore:
    def test_perfect_transcription_scores_100(self):
        assert _intelligibility_score(0.0) == 100.0

    def test_ten_percent_wer_scores_90(self):
        assert _intelligibility_score(0.1) == pytest.approx(90.0)

    def test_total_error_scores_zero(self):
        assert _intelligibility_score(1.0) == 0.0

    def test_wer_above_one_is_clamped_to_zero(self):
        assert _intelligibility_score(1.5) == 0.0


class TestWerAccuracyCheck:
    def test_at_or_below_threshold_passes(self):
        assert (
            _wer_accuracy_check(0.1, DEFAULT_WER_THRESHOLD, 5) is ReportCheckTypes.PASS
        )
        assert (
            _wer_accuracy_check(DEFAULT_WER_THRESHOLD, DEFAULT_WER_THRESHOLD, 5)
            is ReportCheckTypes.PASS
        )

    def test_above_threshold_fails(self):
        assert (
            _wer_accuracy_check(0.5, DEFAULT_WER_THRESHOLD, 5) is ReportCheckTypes.FAIL
        )

    def test_no_valid_samples_is_na_not_fail(self):
        # avg_wer defaults to 1.0 when nothing transcribed; must not be a FAIL.
        assert _wer_accuracy_check(1.0, DEFAULT_WER_THRESHOLD, 0) is ReportCheckTypes.NA

    def test_missing_avg_wer_is_na(self):
        assert (
            _wer_accuracy_check(None, DEFAULT_WER_THRESHOLD, 5) is ReportCheckTypes.NA
        )


class TestMissingQualityDeps:
    def test_importable_deps_report_nothing_missing(self, monkeypatch):
        monkeypatch.setattr(tts, "TTS_QUALITY_DEPS", ("sys", "json"))
        assert _missing_quality_deps() == []

    def test_unimportable_dep_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            tts, "TTS_QUALITY_DEPS", ("sys", "definitely_not_a_real_module_xyz")
        )
        assert _missing_quality_deps() == ["definitely_not_a_real_module_xyz"]


class TestTtsEvalBlockShape:
    def test_carries_canonical_accuracy_fields_only(self):
        block = _tts_eval_block(
            _ctx(),
            _task(),
            score=90.0,
            wer=0.1,
            accuracy_check=ReportCheckTypes.PASS,
        )
        assert block.kind == "evals"
        assert block.task_type == "text_to_speech"
        assert block.data["accuracy_check"] is ReportCheckTypes.PASS
        assert block.data["score"] == 90.0
        assert block.data["wer"] == 0.1
        assert block.data["task_name"] == "tts_generation"
        # Latency/target-check fields belong to the benchmark, never the eval.
        for perf_key in (
            "target_checks",
            "performance_check",
            "rtr",
            "p90_ttft",
            "p95_ttft",
        ):
            assert perf_key not in block.data

    def test_na_block_records_error_reason(self):
        block = _tts_eval_block(
            _ctx(),
            _task(),
            score=None,
            wer=None,
            accuracy_check=ReportCheckTypes.NA,
            error="deps unavailable: torch",
        )
        assert block.data["accuracy_check"] is ReportCheckTypes.NA
        assert block.data["score"] is None
        assert block.data["error"] == "deps unavailable: torch"
