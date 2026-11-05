# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the TTS eval runners."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from test_module._test_common import ReportCheckTypes
from test_module._test_common import TestConfig as _TestConfig
from test_module.eval_tests import tts_eval_tests, tts_naturalness_test
from test_module.eval_tests import tts_isolation_test as iso_mod
from test_module.eval_tests import tts_quality_test as wer_mod
from test_module.eval_tests import tts_speaker_similarity_test as secs_mod
from test_module.eval_tests.tts_client import (
    SynthesizedAudio,
    TTSClient,
    VoiceClip,
    make_tts_client,
    synthesize_all,
)
from test_module.eval_tests.tts_eval_tests import (
    DEFAULT_WER_THRESHOLD,
    _intelligibility_score,
    _missing_deps,
    _tts_eval_block,
    _wer_accuracy_check,
)


SEED_TTS_TASKS = (
    "seed_tts_eval",
    "seed_tts_content_consistency",
    "seed_tts_speaker_similarity",
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
            results, secs_floor=0.65, margin_threshold=0.15, min_valid_fraction=0.6
        )
        assert summary["valid_samples"] == 2 and summary["sample_count"] == 3
        assert summary["failed_samples"] == 1
        assert summary["mean_secs"] == pytest.approx(0.8)
        assert summary["mean_baseline_secs"] == pytest.approx(0.6)
        assert summary["mean_margin"] == pytest.approx(0.2)
        assert summary["success"] is True

    def test_margin_below_threshold_fails(self):
        results = [{"secs": 0.9, "baseline_secs": 0.88, "margin": 0.02}]
        assert not secs_mod.summarize_secs(results, 0.65, 0.05)["success"]

    def test_one_failed_pair_fails_by_default(self):
        results = [{"secs": 0.95, "baseline_secs": 0.6, "margin": 0.35}] * 7 + [
            {"secs": None, "baseline_secs": None, "margin": None}
        ]
        summary = secs_mod.summarize_secs(results, 0.65, 0.05)
        assert summary["mean_secs"] == pytest.approx(0.95)
        assert summary["failed_samples"] == 1 and not summary["success"]

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

    def embed(audio):
        # Same speaker -> identical vector; different speakers -> cosine 0.5.
        vec = np.full(8, 0.5)
        vec[int(audio.samples[0]) % 8] += 1.0
        return vec

    def secs(ref_embedding, audio):
        out = embed(audio)
        return float(
            ref_embedding @ out / (np.linalg.norm(ref_embedding) * np.linalg.norm(out))
        )

    monkeypatch.setattr(test, "_embed", embed)
    monkeypatch.setattr(test, "_secs", secs)
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
        # The clones, then default-voice requests for the same texts.
        texts = [text for text, _ in client.calls]
        voices = [voice for _, voice in client.calls]
        assert voices[4:] == [None] * 4 and all(v is not None for v in voices[:4])
        assert texts[4:] == texts[:4]

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


class _FailingClient:
    """Synthesizes only the first ``good`` requests; the rest fail like a 400 would."""

    can_clone = True
    has_default_voice = True

    def __init__(self, good):
        self.good = good
        self.calls = 0

    def describe(self):
        return "failing"

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        self.calls += 1
        if self.calls > self.good:
            return None
        return SynthesizedAudio(samples=np.zeros(24000), sample_rate=24000)


def _naturalness_result(monkeypatch, good, targets=None):
    clips = [(f"spk{i}", b"x", f"text {i}") for i in range(8)]
    mod = tts_naturalness_test
    monkeypatch.setattr(mod, "iter_speaker_clips", lambda *a, **k: iter(clips))
    monkeypatch.setattr(mod, "make_tts_client", lambda t: _FailingClient(good))
    test = mod.TTSNaturalnessTest(_config(), targets or {"utmos_threshold": 3.0})
    test.base_url = "http://fake"
    monkeypatch.setattr(test, "_load_predictor", lambda: None)
    monkeypatch.setattr(test, "_score", lambda audio: 4.5)
    return asyncio.run(test._run_specific_test_async())


class TestNaturalnessFailedSamples:
    def test_one_good_sample_out_of_eight_does_not_pass(self, monkeypatch):
        result = _naturalness_result(monkeypatch, good=1)
        assert result["mean_utmos"] == pytest.approx(4.5)
        assert result["valid_samples"] == 1 and result["failed_samples"] == 7
        assert result["success"] is False

    def test_all_samples_valid_passes(self, monkeypatch):
        result = _naturalness_result(monkeypatch, good=8)
        assert result["failed_samples"] == 0 and result["success"] is True

    def test_a_lower_valid_fraction_tolerates_failures(self, monkeypatch):
        targets = {"utmos_threshold": 3.0, "min_valid_fraction": 0.75}
        assert _naturalness_result(monkeypatch, good=6, targets=targets)["success"]
        assert not _naturalness_result(monkeypatch, good=5, targets=targets)["success"]


class _CloneOnlyClient:
    """Records each request; the second one fails."""

    can_clone = True
    has_default_voice = False

    def __init__(self):
        self.requests = []

    def describe(self):
        return "clone-only"

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        self.requests.append((text, voice))
        if len(self.requests) == 2:
            return None
        return SynthesizedAudio(samples=np.zeros(24000), sample_rate=24000)


class TestWerThroughAClient:
    CLIPS = [("a", b"A", "text a"), ("b", b"B", "text b"), ("c", b"C", "text c")]

    def _run(self, monkeypatch, targets):
        client = _CloneOnlyClient()
        monkeypatch.setattr(
            wer_mod, "iter_speaker_clips", lambda *a, **k: iter(self.CLIPS)
        )
        monkeypatch.setattr(wer_mod, "make_tts_client", lambda t: client)
        test = wer_mod.TTSQualityTest(_config(), targets)
        test.base_url = "http://fake"
        monkeypatch.setattr(test, "_load_whisper_model", lambda: None)
        monkeypatch.setattr(
            test, "_download_samples", lambda **k: pytest.fail("media-server path")
        )
        heard = iter(["text b", "wrong"])
        monkeypatch.setattr(test, "_transcribe_samples", lambda a, sr: next(heard))
        return client, asyncio.run(test._run_specific_test_async())

    def test_clones_read_the_next_speakers_text_and_wer_is_scored(self, monkeypatch):
        targets = {"tts_api": "vllm_omni", "sample_count": 3, "cleanup": False}
        client, result = self._run(monkeypatch, targets)
        assert [text for text, _ in client.requests] == ["text b", "text c", "text a"]
        assert [voice.wav_bytes for _, voice in client.requests] == [b"A", b"B", b"C"]
        assert result["valid_samples"] == 2 and result["total_samples"] == 3
        assert result["avg_wer"] == pytest.approx(0.5)
        assert not result["success"]

    def test_the_task_threshold_is_the_gate(self, monkeypatch):
        targets = {"tts_api": "vllm_omni", "cleanup": False, "wer_threshold": 0.6}
        _, result = self._run(monkeypatch, targets)
        assert result["wer_threshold"] == 0.6 and result["success"]

    def test_the_runner_passes_the_task_targets(self, monkeypatch):
        seen = {}

        class _Fake:
            def __init__(self, config, targets, ctx):
                seen["targets"] = targets

            def run_tests(self):
                return SimpleNamespace(data={"success": True})

        monkeypatch.setattr(wer_mod, "TTSQualityTest", _Fake)
        monkeypatch.setattr(tts_eval_tests, "get_num_calls", lambda ctx: 2)
        task = SimpleNamespace(model_kwargs={"tts_api": "vllm_omni"})
        tts_eval_tests._run_tts_quality_eval(SimpleNamespace(base_url="x"), task)
        assert seen["targets"]["tts_api"] == "vllm_omni"
        assert seen["targets"]["wer_threshold"] == DEFAULT_WER_THRESHOLD


class TestEvalBlockSampleCounts:
    def test_counts_reach_the_report_and_no_audio_is_a_fail(self, monkeypatch):
        result = {
            "success": False,
            "mean_utmos": None,
            "valid_samples": 0,
            "failed_samples": 8,
            "sample_count": 8,
        }
        monkeypatch.setattr(tts_eval_tests, "_missing_deps", lambda deps: [])
        monkeypatch.setattr(
            tts_eval_tests, "_run_tts_naturalness_eval", lambda ctx, task: result
        )
        block = tts_eval_tests._run_naturalness_task(_ctx(), _task())
        assert block.data["accuracy_check"] == ReportCheckTypes.FAIL
        assert block.data["failed_samples"] == 8 and block.data["sample_count"] == 8


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


class TestQwen3EvalConfig:
    def _tasks(self, repo):
        # EVAL_CONFIGS holds only the active catalog; Qwen3-TTS is in dev.
        from reference_config.evals.eval_config import _eval_config_map

        return {t.task_name: t for t in _eval_config_map[repo].tasks}

    @pytest.mark.parametrize(
        "size, wer",
        [("1.7B", {"tts_generation", *SEED_TTS_TASKS}), ("0.6B", set())],
    )
    def test_base_registers_secs_and_naturalness_on_clones(self, size, wer):
        tasks = self._tasks(f"Qwen/Qwen3-TTS-12Hz-{size}-Base")
        assert set(tasks) == {"tts_speaker_similarity", "tts_naturalness"} | wer
        # The seed_tts tasks' client is EvalScope's, in its own venv.
        for task in (t for n, t in tasks.items() if n not in SEED_TTS_TASKS):
            targets = tts_eval_tests._task_targets(task)
            client = make_tts_client(targets)
            assert client.can_clone and not client.has_default_voice
            assert task.task_name in tts_eval_tests._TASK_RUNNERS
        secs = tts_eval_tests._task_targets(tasks["tts_speaker_similarity"])
        assert secs["secs_baseline"] == "cross_speaker"
        assert 0 < secs["margin_threshold"] < secs["secs_floor"] < 1

    @pytest.mark.parametrize("size", ["1.7B", "0.6B"])
    def test_custom_voice_registers_naturalness_with_a_fixed_speaker(self, size):
        tasks = self._tasks(f"Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice")
        assert set(tasks) == {"tts_naturalness"}
        targets = tts_eval_tests._task_targets(tasks["tts_naturalness"])
        client = make_tts_client(targets)
        assert client.has_default_voice and client.speaker
        assert 1.0 < targets["utmos_threshold"] < 5.0

    def test_voice_design_registers_naturalness_with_a_fixed_description(self):
        tasks = self._tasks("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        assert set(tasks) == {"tts_naturalness"}
        targets = tts_eval_tests._task_targets(tasks["tts_naturalness"])
        client = make_tts_client(targets)
        assert client.has_default_voice and not client.can_clone
        assert 1.0 < targets["utmos_threshold"] < 5.0
        request = client.build_request("text a", None)
        assert request == {
            "json": {
                "text": "text a",
                "language": "English",
                "description": targets["description"],
            }
        }
        clips = [("s1", b"A", "text a"), ("s2", b"B", "text b")]
        battery = tts_naturalness_test.naturalness_battery(
            clips, cloned=not client.has_default_voice
        )
        assert battery == [("text a", None), ("text b", None)]

    @pytest.mark.parametrize("release", ["Base", "CustomVoice"])
    def test_0b6_never_sends_a_description(self, release):
        small = self._tasks(f"Qwen/Qwen3-TTS-12Hz-0.6B-{release}")
        large = self._tasks(f"Qwen/Qwen3-TTS-12Hz-1.7B-{release}")
        # WER and Seed-TTS-Eval are calibrated only on the 1.7B Base.
        assert set(small) == set(large) - {"tts_generation", *SEED_TTS_TASKS}
        for name, task in small.items():
            targets = tts_eval_tests._task_targets(task)
            assert "description" not in targets
            client = make_tts_client(targets)
            voice = VoiceClip(b"RIFF") if client.can_clone else None
            request = client.build_request("hi", voice)
            fields = (
                request["json"]
                if "json" in request
                else {field[0]["name"] for field in request["data"]._fields}
            )
            assert "description" not in fields


class _MarkedClient:
    """Audio carries the request's index; ``swap`` exchanges two batched outputs."""

    can_clone = True
    has_default_voice = False

    def __init__(self, texts, swap=None):
        self.texts = texts
        self.swap = swap
        self.batched = 0

    def describe(self):
        return "marked"

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        index = self.texts.index(text)
        if self.swap and len(self.texts) <= self.batched:
            index = {self.swap[0]: self.swap[1], self.swap[1]: self.swap[0]}.get(
                index, index
            )
        self.batched += 1
        return SynthesizedAudio(samples=np.full(2400, index), sample_rate=24000)


class TestCrossTalk:
    # Each request reads the next speaker's text in its own voice.
    CLIPS = [(s, s.encode(), f"text of {s}") for s in ("a", "b", "c", "d")]

    def _run(self, monkeypatch, swap=None):
        texts = [self.CLIPS[(i + 1) % 4][2] for i in range(4)]
        client = _MarkedClient(texts, swap)
        monkeypatch.setattr(iso_mod, "iter_speaker_clips", lambda *a, **k: self.CLIPS)
        monkeypatch.setattr(iso_mod, "make_tts_client", lambda t: client)
        voice_of = {clip[1]: i for i, clip in enumerate(self.CLIPS)}
        monkeypatch.setattr(
            iso_mod,
            "decode_wav_file",
            lambda wav: SynthesizedAudio(np.full(10, voice_of[wav]), 16000),
        )
        test = iso_mod.TTSIsolationTest(_config(), {"tts_api": "vllm_omni"})
        test.base_url = "http://fake"
        for name in ("_load_whisper_model", "_load_encoder", "_unload_encoder"):
            monkeypatch.setattr(test, name, lambda: None)
        monkeypatch.setattr(
            test, "_transcribe_samples", lambda audio, sr: texts[int(audio[0])]
        )
        monkeypatch.setattr(test, "_embed", lambda audio: int(audio.samples[0]))
        monkeypatch.setattr(test, "_cosine", lambda a, b: 0.95 if a == b else 0.5)
        return asyncio.run(test._run_specific_test_async())

    def test_separate_requests_pass(self, monkeypatch):
        result = self._run(monkeypatch)
        assert result["success"] and result["mismatches"] == 0
        assert result["wer_gap"] == 0.0
        assert [r["own_best"] for r in result["per_request"]] == [True] * 4

    def test_swapped_outputs_fail(self, monkeypatch):
        result = self._run(monkeypatch, swap=(0, 2))
        assert not result["success"] and result["mismatches"] == 2
        assert [r["own_best"] for r in result["per_request"]] == [
            False,
            True,
            False,
            True,
        ]

    def test_a_wer_gap_beyond_the_limit_fails(self):
        rows = [
            iso_mod.isolation_row(i, "s", 0.0, wers, [0.9, 0.5], 0.9)
            if i == 0
            else iso_mod.isolation_row(i, "s", 0.0, wers[::-1], [0.5, 0.9], 0.9)
            for i, wers in enumerate([[0.3, 1.0], [0.3, 1.0]])
        ]
        assert all(r["own_best"] for r in rows)
        assert not iso_mod.summarize_isolation(rows, 0.10)["success"]
        assert iso_mod.summarize_isolation(rows, 0.35)["success"]

    def test_the_galaxy_spec_suite_runs_it(self):
        from test_module.dispatch import _instantiate_spec_test
        from test_module.test_categorization_system import TestFilter

        def cases(device):
            suites = (
                TestFilter()
                .filter_by_model("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
                .filter_by_device(device)
                .get_tests()
            )
            return [c for s in suites for c in s["test_cases"]]

        assert cases("n150") == []
        (case,) = [c for c in cases("galaxy") if c["name"] == "TTSIsolationTest"]
        test = _instantiate_spec_test(case, None)
        assert isinstance(test, iso_mod.TTSIsolationTest)
        assert test.targets["concurrency"] == 4
        assert make_tts_client(test.targets).describe().startswith("vllm_omni")
        block = test._block({"success": True})
        assert block.kind == "spec_tests"
        assert block.targets["notes"] == [iso_mod.CROSS_TALK_LEGEND]


class _CountingClient(TTSClient):
    """Records how many requests are in flight at once; audio carries the seed."""

    has_default_voice = False

    def __init__(self):
        self.in_flight = self.peak = 0

    async def synthesize(self, url, text, *, seed, voice=None, timeout_s=300):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0.01 * (5 - seed % 5))
        self.in_flight -= 1
        return SynthesizedAudio(samples=np.full(2400, seed), sample_rate=24000)

    async def timed_synthesize(self, url, text, *, voice=None, timeout_s=120):
        raise NotImplementedError

    def describe(self):
        return "counting"


class TestEvalConcurrency:
    @pytest.mark.parametrize("concurrency", [1, 4])
    def test_synthesize_all_caps_requests_in_flight_and_keeps_order(self, concurrency):
        client = _CountingClient()
        requests = [(f"t{i}", i, None) for i in range(8)]
        audios = asyncio.run(synthesize_all(client, "u", requests, concurrency))
        assert client.peak == concurrency
        assert [int(a.samples[0]) for a in audios] == list(range(8))

    def test_naturalness_sends_its_requests_at_the_task_concurrency(self, monkeypatch):
        client = _CountingClient()
        clips = [(s, s.encode(), f"text of {s}") for s in "abcdefgh"]
        mod = tts_naturalness_test
        monkeypatch.setattr(mod, "iter_speaker_clips", lambda *a, **k: clips)
        monkeypatch.setattr(mod, "make_tts_client", lambda targets: client)
        test = mod.TTSNaturalnessTest(_config(), {"concurrency": 4})
        test.base_url = "http://fake"
        monkeypatch.setattr(test, "_load_predictor", lambda: None)
        monkeypatch.setattr(test, "_score", lambda audio: 4.5)
        result = asyncio.run(test._run_specific_test_async())
        assert client.peak == 4
        assert result["concurrency"] == 4 and result["valid_samples"] == 8


class TestEvalNotes:
    @pytest.fixture(autouse=True)
    def _no_block_id(self, monkeypatch):
        monkeypatch.setattr(tts_eval_tests, "block_id", lambda ctx: "id")

    def _block(self, name, result):
        task = SimpleNamespace(task_name=name, score=_task().score)
        return _tts_eval_block(
            SimpleNamespace(),
            task,
            score=90.0,
            accuracy_check=ReportCheckTypes.PASS,
            result=result,
        )

    def test_a_task_line_states_its_sample_count_concurrency_and_gate(self):
        block = self._block(
            "tts_generation",
            {"total_samples": 10, "concurrency": 4, "wer_threshold": 0.2},
        )
        legend, line = block.targets["notes"]
        assert legend == tts_eval_tests.EVAL_LEGEND
        assert tts_eval_tests.LIBRITTS_R_URL in legend
        assert "10 samples, 4 requests at a time" in line
        assert line.endswith("Pass: WER ≤ 0.2.")

    @pytest.mark.parametrize(
        "published, ref, shown",
        [
            (14.0, "ref-url", (14.0, "ref-url")),
            (14.0, "", ("N/A", "N/A")),
            (None, "", ("N/A", "N/A")),
        ],
    )
    def test_a_published_score_shows_only_with_its_source(self, published, ref, shown):
        score = SimpleNamespace(
            tolerance=0.05, published_score=published, published_score_ref=ref
        )
        block = _tts_eval_block(
            SimpleNamespace(),
            SimpleNamespace(task_name="tts_generation", score=score),
            score=90.0,
            accuracy_check=ReportCheckTypes.PASS,
        )
        assert (
            block.data["published_score"],
            block.data["published_score_ref"],
        ) == shown

    def test_the_table_has_no_tolerance_since_nothing_is_within_it(self):
        block = self._block("tts_generation", {})
        assert "tolerance" not in block.data

    def test_a_task_that_did_not_run_has_no_counts_or_gate(self):
        _, line = self._block("tts_naturalness", {}).targets["notes"]
        assert "samples" not in line and "Pass" not in line

    def test_the_report_prints_the_legend_once_then_every_task(self):
        from report_module import renderers
        from report_module.generator import _collapse_same_heading_blocks

        blocks = [
            self._block("tts_generation", {"wer_threshold": 0.2}),
            self._block("tts_naturalness", {"utmos_threshold": 4.2}),
        ]
        (merged,) = _collapse_same_heading_blocks(blocks)
        text = renderers.render_evals(merged, {})
        assert text.count("**Legend**") == 1
        assert "WER ≤ 0.2" in text and "mean_utmos ≥ 4.2" in text
        assert renderers.EVALS_METHODOLOGY_NOTE not in text


_SEED_SUMMARY = {
    "sample_count": 1088,
    "concurrency": 4,
    "wer_evalscope": 0.010,
    "wer_official": 0.0163,
    "mean_sim": 0.689,
}


def _seed_tasks():
    from reference_config.evals.eval_config import (
        _eval_config_map,
        resolve_task_for_device,
    )

    tasks = _eval_config_map["Qwen/Qwen3-TTS-12Hz-1.7B-Base"].tasks
    return [
        resolve_task_for_device(t, "GALAXY")
        for t in tasks
        if t.task_name in SEED_TTS_TASKS
    ]


def _seed_ctx(limit_samples_mode=None):
    return SimpleNamespace(
        model_spec=SimpleNamespace(model_name="Qwen3-TTS-12Hz-1.7B-Base"),
        device=SimpleNamespace(name="GALAXY"),
        base_url="http://127.0.0.1:8000",
        runtime_config=SimpleNamespace(limit_samples_mode=limit_samples_mode),
    )


class TestSeedTts:
    @pytest.fixture(autouse=True)
    def _fresh_runs(self, monkeypatch):
        monkeypatch.setattr(tts_eval_tests, "_SEED_TTS_RUNS", {})

    def _rows(self, monkeypatch, summary, limit_samples_mode=None):
        calls = []

        def run(ctx, targets, limit=None):
            calls.append((targets, limit))
            if isinstance(summary, Exception):
                raise summary
            return dict(summary)

        monkeypatch.setattr(tts_eval_tests, "_run_seed_tts_eval", run)
        ctx = _seed_ctx(limit_samples_mode)
        blocks = [tts_eval_tests._run_seed_tts_task(ctx, t) for t in _seed_tasks()]
        return calls, {b.data["task_name"]: b.data for b in blocks}

    @pytest.mark.parametrize(
        "mode, limit", [(None, None), ("ci-nightly", 100), ("smoke-test", 0.01)]
    )
    def test_ci_mode_runs_100_clips_and_a_full_run_all(self, monkeypatch, mode, limit):
        calls, _ = self._rows(monkeypatch, _SEED_SUMMARY, mode)
        assert [lim for _, lim in calls] == [limit]

    def test_three_rows_score_one_run(self, monkeypatch):
        calls, rows = self._rows(monkeypatch, _SEED_SUMMARY)
        assert len(calls) == 1 and calls[0][0]["concurrency"] == 4
        assert rows["seed_tts_eval"]["wer"] == 0.010
        assert rows["seed_tts_content_consistency"]["wer"] == 0.0163
        assert rows["seed_tts_speaker_similarity"]["mean_sim"] == 0.689
        assert rows["seed_tts_eval"]["score"] == pytest.approx(99.0)
        assert rows["seed_tts_speaker_similarity"]["score"] == pytest.approx(68.9)
        assert {r["accuracy_check"] for r in rows.values()} == {ReportCheckTypes.PASS}

    def test_only_the_evalscope_wer_carries_the_model_card_score(self, monkeypatch):
        _, rows = self._rows(monkeypatch, _SEED_SUMMARY)
        evalscope = rows["seed_tts_eval"]
        assert evalscope["published_score"] == pytest.approx(100 * (1 - 0.0124))
        assert (
            "huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            in (evalscope["published_score_ref"])
        )
        for name in ("seed_tts_content_consistency", "seed_tts_speaker_similarity"):
            assert rows[name]["published_score"] == "N/A"

    def test_each_row_gates_on_its_own_value(self, monkeypatch):
        summary = {**_SEED_SUMMARY, "wer_official": 0.05, "mean_sim": 0.5}
        _, rows = self._rows(monkeypatch, summary)
        assert rows["seed_tts_eval"]["accuracy_check"] is ReportCheckTypes.PASS
        assert (
            rows["seed_tts_content_consistency"]["accuracy_check"]
            is ReportCheckTypes.FAIL
        )
        assert (
            rows["seed_tts_speaker_similarity"]["accuracy_check"]
            is ReportCheckTypes.FAIL
        )

    def test_a_failed_run_is_not_started_again_and_every_row_says_why(
        self, monkeypatch
    ):
        calls, rows = self._rows(monkeypatch, RuntimeError("server gone"))
        assert len(calls) == 1
        for row in rows.values():
            assert row["accuracy_check"] is ReportCheckTypes.NA
            assert "server gone" in row["error"]

    def test_the_media_server_client_is_refused_before_provisioning(self, monkeypatch):
        import workflow_module.venv_provisioner as vp

        monkeypatch.setattr(
            vp, "get_venv_provisioner", lambda: pytest.fail("provisioned")
        )
        with pytest.raises(ValueError, match="vllm_omni"):
            tts_eval_tests._run_seed_tts_eval(_seed_ctx(), {"tts_api": "qwen3_cpp"})

    def test_the_runner_gets_the_venv_paths_server_and_key(self, monkeypatch, tmp_path):
        import json
        import subprocess

        import workflow_module.venv_provisioner as vp

        class Provisioner:
            def provision(self, venv_type, model_spec):
                return True

            def venv_path(self, venv_type):
                return tmp_path

            def venv_python(self, venv_type):
                return str(tmp_path / "bin" / "python")

        seen = {}

        def run(cmd, env, check, timeout):
            seen.update(cmd=cmd, env=env)
            out = cmd[cmd.index("--out") + 1]
            with open(out, "w") as f:
                json.dump(_SEED_SUMMARY, f)

        monkeypatch.setattr(vp, "get_venv_provisioner", lambda: Provisioner())
        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setenv("OPENAI_API_KEY", "tok")
        targets = {"tts_api": "vllm_omni", "concurrency": 4}
        assert tts_eval_tests._run_seed_tts_eval(_seed_ctx(), targets) == _SEED_SUMMARY
        assert "--limit" not in seen["cmd"]
        tts_eval_tests._run_seed_tts_eval(_seed_ctx(), targets, 100)
        cmd, env = seen["cmd"], seen["env"]
        assert cmd[cmd.index("--limit") + 1] == "100"
        assert cmd[0] == str(tmp_path / "bin" / "python")
        assert cmd[cmd.index("--api-url") + 1] == "http://127.0.0.1:8000/v1"
        assert cmd[cmd.index("--concurrency") + 1] == "4"
        assert cmd[cmd.index("--sv-checkpoint") + 1].startswith(str(tmp_path))
        assert env["TORCH_HOME"].startswith(str(tmp_path))
        assert env["TTS_API_KEY"] == "tok"
