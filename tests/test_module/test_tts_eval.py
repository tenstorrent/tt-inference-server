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


# --------------------------------------------------------------- tt-tts correctness


def _tt_tts_ctx(tmp_path, workflow_args=None, impl_name="tt-tts"):
    return SimpleNamespace(
        model_spec=SimpleNamespace(
            model_name="tts-2", impl=SimpleNamespace(impl_name=impl_name)
        ),
        device=SimpleNamespace(name="QUAD_GALAXY"),
        all_params=SimpleNamespace(tasks=[_task()]),
        runtime_config=SimpleNamespace(workflow_args=workflow_args),
        base_url="http://quad-driver.example:8000",
        server_port=8000,
        output_path=str(tmp_path),
    )


def _graded(tag, i, ok=True, wer=0.0):
    return {"tag": tag, "i": i, "pass": ok, "wer": wer}


class TestCorrectnessParams:
    def test_defaults(self, tmp_path):
        assert tts._correctness_params(_tt_tts_ctx(tmp_path)) == {
            "refs": 5,
            "requests": 100,
            "concurrency": 100,
        }

    def test_eval_keys_override_and_benchmark_keys_are_ignored(self, tmp_path):
        ctx = _tt_tts_ctx(
            tmp_path,
            "concurrency=600,700,800 isl=128 eval_requests=20 eval_concurrency=10 "
            "eval_refs=0 eval_other=1",
        )
        assert tts._correctness_params(ctx) == {
            "refs": 0,
            "requests": 20,
            "concurrency": 10,
        }

    @pytest.mark.parametrize(
        "arg", ["eval_requests=0", "eval_concurrency=x", "eval_refs=-1"]
    )
    def test_invalid_values_are_rejected(self, tmp_path, arg):
        with pytest.raises(ValueError):
            tts._correctness_params(_tt_tts_ctx(tmp_path, arg))


class TestTtTtsRouting:
    def test_tt_tts_runs_the_correctness_eval(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts, "require_health", lambda ctx, req=None: None)
        monkeypatch.setattr(
            tts, "_run_tts_correctness_eval", lambda ctx, task: "correctness"
        )
        assert tts.run_tts_eval(_tt_tts_ctx(tmp_path)) == "correctness"

    def test_other_tts_models_keep_the_quality_eval(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts, "require_health", lambda ctx, req=None: None)
        monkeypatch.setattr(
            tts, "_run_tts_correctness_eval", lambda ctx, task: pytest.fail("routed")
        )
        monkeypatch.setattr(tts, "_missing_quality_deps", lambda: ["torch"])
        block = tts.run_tts_eval(_tt_tts_ctx(tmp_path, impl_name="speecht5-tts"))
        assert block.data["accuracy_check"] is ReportCheckTypes.NA

    def test_missing_faster_whisper_is_na(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts, "_can_import", lambda name: False)
        block = tts._run_tts_correctness_eval(_tt_tts_ctx(tmp_path), _task())
        assert block.data["accuracy_check"] is ReportCheckTypes.NA
        assert "faster-whisper" in block.data["error"]


class TestCorrectnessVerdict:
    def _run(self, monkeypatch, tmp_path, records, load_failures, ref_failures):
        from test_module.benchmark_tests import tts_correctness

        seen = {}

        def fake_run_check(target, refs, requests, concurrency, args):
            seen.update(target=target, refs=refs, requests=requests, args=args)
            return {
                "records": records,
                "summary": {
                    "load_requests": requests,
                    "load_failures": load_failures,
                    "reference_failures": ref_failures,
                },
            }

        monkeypatch.setattr(tts, "_can_import", lambda name: True)
        monkeypatch.setattr(tts_correctness, "run_check", fake_run_check)
        monkeypatch.setattr(tts_correctness, "report", lambda summary, recs: "")
        ctx = _tt_tts_ctx(tmp_path, "eval_requests=3 eval_concurrency=3 eval_refs=1")
        return tts._run_tts_correctness_eval(ctx, _task()), seen

    def test_all_correct_passes(self, monkeypatch, tmp_path):
        records = [_graded("ref", 0)] + [_graded("load", i) for i in range(3)]
        block, seen = self._run(monkeypatch, tmp_path, records, 0, 0)

        assert seen["target"].host == "quad-driver.example"
        assert (seen["refs"], seen["requests"]) == (1, 3)
        assert seen["args"].max_wer == tts.CORRECTNESS_MAX_WER
        assert seen["args"].outdir == str(tmp_path / "tts_correctness")
        assert block.data["accuracy_check"] is ReportCheckTypes.PASS
        assert block.data["score"] == 100.0
        assert block.data["failed_responses"] is None

    def test_any_malformed_response_fails(self, monkeypatch, tmp_path):
        records = [_graded("ref", 0)] + [
            _graded("load", 0),
            _graded("load", 1, ok=False, wer=0.9),
            _graded("load", 2),
        ]
        block, _ = self._run(monkeypatch, tmp_path, records, 1, 0)

        assert block.data["accuracy_check"] is ReportCheckTypes.FAIL
        assert block.data["score"] == 75.0
        assert block.data["load_failures"] == 1
        assert block.data["failed_responses"] == "load_001"
