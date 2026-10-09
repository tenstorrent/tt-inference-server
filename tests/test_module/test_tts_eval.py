# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the TTS eval runners.

Focuses on the pure decision logic (WER -> score, WER -> accuracy_check,
dependency preflight, the SECS baselines and margin gate), the shape of the
emitted eval Block and the EvalConfig -> eval-target wiring. The heavy
scoring models (Whisper, the speaker encoder, UTMOS) and live servers are
exercised on hardware, not here.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from test_module._test_common import ReportCheckTypes
from test_module._test_common import TestConfig as _TestConfig
from test_module.eval_tests import tts_eval_tests, tts_naturalness_test
from test_module.eval_tests import tts_speaker_similarity_test as secs_mod
from test_module.eval_tests.tts_client import SynthesizedAudio
from test_module.eval_tests.tts_eval_tests import (
    DEFAULT_WER_THRESHOLD,
    _intelligibility_score,
    _missing_deps,
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


class TestMissingDeps:
    def test_importable_deps_report_nothing_missing(self):
        assert _missing_deps(("sys", "json")) == []

    def test_unimportable_dep_is_reported(self):
        assert _missing_deps(("sys", "definitely_not_a_real_module_xyz")) == [
            "definitely_not_a_real_module_xyz"
        ]


class TestTtsEvalBlockShape:
    def test_carries_canonical_accuracy_fields_only(self):
        block = _tts_eval_block(
            _ctx(),
            _task(),
            score=90.0,
            accuracy_check=ReportCheckTypes.PASS,
            metrics={"wer": 0.1},
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
            accuracy_check=ReportCheckTypes.NA,
            error="deps unavailable: torch",
            metrics={"wer": None},
        )
        assert block.data["accuracy_check"] is ReportCheckTypes.NA
        assert block.data["score"] is None
        assert block.data["error"] == "deps unavailable: torch"


# ---------------------------------------------------------------- SECS logic


class TestCrossSpeakerPartner:
    @pytest.mark.parametrize("count", [3, 4, 8])
    def test_never_self_and_never_the_text_owner(self, count):
        for i in range(count):
            partner = secs_mod.cross_speaker_partner(i, count)
            assert partner != i
            assert partner != (i + 1) % count  # the speaker whose text is spoken

    def test_needs_two_speakers(self):
        with pytest.raises(ValueError):
            secs_mod.cross_speaker_partner(0, 1)


class TestSummarizeSecs:
    def test_means_over_complete_pairs_only(self):
        results = [
            {"secs": 0.9, "baseline_secs": 0.6, "margin": 0.3},
            {"secs": 0.7, "baseline_secs": 0.6, "margin": 0.1},
            {"secs": 0.8, "baseline_secs": None, "margin": None},
        ]
        summary = secs_mod.summarize_secs(
            results, secs_floor=0.65, margin_threshold=0.15
        )
        assert summary["valid_samples"] == 2 and summary["sample_count"] == 3
        assert summary["mean_secs"] == pytest.approx(0.8)
        assert summary["mean_baseline_secs"] == pytest.approx(0.6)
        assert summary["mean_margin"] == pytest.approx(0.2)
        assert summary["success"] is True

    def test_margin_below_threshold_fails(self):
        results = [{"secs": 0.9, "baseline_secs": 0.88, "margin": 0.02}]
        assert not secs_mod.summarize_secs(results, 0.65, 0.05)["success"]

    def test_no_complete_pair_fails(self):
        summary = secs_mod.summarize_secs([{"secs": None, "margin": None}], 0.65, 0.05)
        assert summary["mean_secs"] is None and not summary["success"]


class _FakeClient:
    """Voice ids travel as one-sample "audio"; a broken cloner ignores the clip."""

    def __init__(self, *, has_default_voice, follows_clip=True):
        self.can_clone = True
        self.has_default_voice = has_default_voice
        self.follows_clip = follows_clip
        self.calls = []

    def describe(self):
        return "fake"

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        self.calls.append((text, voice))
        speaker = int(voice.wav_bytes) if voice and self.follows_clip else 99
        return SynthesizedAudio(samples=np.array([speaker]), sample_rate=16000)


def _secs_test(monkeypatch, client, targets=None):
    clips = [(f"spk{i}", str(i).encode(), f"text {i}") for i in range(4)]
    monkeypatch.setattr(secs_mod, "iter_speaker_clips", lambda *a, **k: iter(clips))
    monkeypatch.setattr(secs_mod, "make_tts_client", lambda targets: client)
    monkeypatch.setattr(
        secs_mod,
        "decode_wav_file",
        lambda b: SynthesizedAudio(samples=np.array([int(b)]), sample_rate=16000),
    )
    test = secs_mod.TTSSpeakerSimilarityTest(_config(), targets or {})
    test.base_url = "http://fake"
    monkeypatch.setattr(test, "_load_encoder", lambda: None)

    import torch

    def embed(audio):
        # Same speaker -> identical vector; different speakers -> cosine 0.5.
        vec = torch.full((8,), 0.5)
        vec[int(audio.samples[0]) % 8] += 1.0
        return vec

    monkeypatch.setattr(test, "_embed", embed)
    return test


def _config():
    return _TestConfig({"timeout": 60, "retry_attempts": 1, "retry_delay": 0})


class TestSecsBaselines:
    def test_cross_speaker_baseline_for_a_server_without_default_voice(
        self, monkeypatch
    ):
        client = _FakeClient(has_default_voice=False)
        test = _secs_test(monkeypatch, client, {"margin_threshold": 0.1})
        result = asyncio.run(test._run_specific_test_async())

        assert result["baseline"] == "cross_speaker"
        # Every request clones, and speaker i's baseline is the previous
        # speaker's clone of the same text.
        assert all(voice is not None for _, voice in client.calls)
        for i, row in enumerate(result["per_sample"]):
            assert row["baseline_speaker"] == f"spk{(i - 1) % 4}"
            assert row["secs"] == pytest.approx(1.0)
            assert row["margin"] > 0.1
        assert result["success"] is True

    def test_cross_speaker_margin_collapses_when_cloning_ignores_the_clip(
        self, monkeypatch
    ):
        client = _FakeClient(has_default_voice=False, follows_clip=False)
        test = _secs_test(monkeypatch, client, {"margin_threshold": 0.05})
        result = asyncio.run(test._run_specific_test_async())

        assert result["mean_margin"] == pytest.approx(0.0, abs=1e-6)
        assert result["success"] is False

    def test_default_voice_baseline_is_kept_for_servers_with_one(self, monkeypatch):
        client = _FakeClient(has_default_voice=True)
        test = _secs_test(monkeypatch, client)
        result = asyncio.run(test._run_specific_test_async())

        assert result["baseline"] == "default_voice"
        # Alternating clone / default-voice requests, same text per pair.
        voices = [voice for _, voice in client.calls]
        assert voices[1::2] == [None] * 4 and all(v is not None for v in voices[::2])

    def test_default_voice_baseline_is_refused_without_a_default_voice(
        self, monkeypatch
    ):
        client = _FakeClient(has_default_voice=False)
        test = _secs_test(monkeypatch, client, {"secs_baseline": "default_voice"})
        with pytest.raises(ValueError, match="no default voice"):
            asyncio.run(test._run_specific_test_async())


class TestNaturalnessBattery:
    CLIPS = [("a", b"A", "text a"), ("b", b"B", "text b"), ("c", b"C", "text c")]

    def test_default_voice_reads_each_text_once(self):
        battery = tts_naturalness_test.naturalness_battery(self.CLIPS, cloned=False)
        assert battery == [("text a", None), ("text b", None), ("text c", None)]

    def test_clones_read_the_next_speakers_text(self):
        battery = tts_naturalness_test.naturalness_battery(self.CLIPS, cloned=True)
        assert [text for text, _ in battery] == ["text b", "text c", "text a"]
        assert [voice.wav_bytes for _, voice in battery] == [b"A", b"B", b"C"]
        assert battery[0][1].transcript == "text a"


# ------------------------------------------------------- harness wiring
class TestTaskTargets:
    def test_model_kwargs_become_targets_minus_lm_eval_keys(self):
        task = SimpleNamespace(
            model_kwargs={"tts_api": "qwen3_cpp", "add_bos_token": "True"}
        )
        assert tts_eval_tests._task_targets(task) == {"tts_api": "qwen3_cpp"}

    def test_no_model_kwargs_means_module_defaults(self):
        assert tts_eval_tests._task_targets(SimpleNamespace(model_kwargs=None)) == {}

    def test_secs_runner_passes_the_task_targets(self, monkeypatch):
        seen = {}

        class _Fake:
            def __init__(self, config, targets, ctx):
                seen["targets"] = targets

            def run_tests(self):
                return SimpleNamespace(data={"success": True})

        monkeypatch.setattr(secs_mod, "TTSSpeakerSimilarityTest", _Fake)
        task = SimpleNamespace(model_kwargs={"tts_api": "qwen3_cpp", "secs_floor": 0.7})
        ctx = SimpleNamespace(base_url="http://fake")
        tts_eval_tests._run_tts_secs_eval(ctx, task)
        assert seen["targets"] == {"tts_api": "qwen3_cpp", "secs_floor": 0.7}
