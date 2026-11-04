# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""TTS benchmark request path against fake servers (no numpy, torch or soundfile)."""

from __future__ import annotations

import asyncio
import struct
from types import SimpleNamespace

import aiohttp
import pytest

from test_module.benchmark_tests import tts_benchmark_tests as bench
from test_module.eval_tests import tts_client
from test_module.eval_tests.tts_client import (
    MediaServerTTSClient,
    Qwen3CppTTSClient,
    parse_pcm16_wav_header,
)


def _streamed_wav_header(sample_rate=24000):
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    return (
        b"RIFF\xff\xff\xff\xffWAVEfmt "
        + struct.pack("<I", 16)
        + fmt
        + b"data\xff\xff\xff\xff"
    )


class _Content:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _Response:
    def __init__(self, status=200, chunks=(), payload=None, content_type=""):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.content = _Content(list(chunks))
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return b"".join(self.content._chunks).decode("utf-8", "replace")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, response, log, kwargs):
        self._response = response
        self._log = log
        log.append({"session": kwargs})

    def post(self, url, **kwargs):
        self._log.append({"url": url, **kwargs})
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    log = []

    def install(response):
        monkeypatch.setattr(
            aiohttp, "ClientSession", lambda **kw: _Session(response, log, kw)
        )
        return log

    return install


@pytest.fixture
def clock(monkeypatch):
    def install(*ticks):
        it = iter(ticks)
        monkeypatch.setattr(
            tts_client, "time", SimpleNamespace(monotonic=lambda: next(it))
        )

    return install


def _ctx(model_kwargs=None, tasks=None):
    if tasks is None:
        tasks = [SimpleNamespace(task_name="t", model_kwargs=model_kwargs)]
    return SimpleNamespace(
        all_params=SimpleNamespace(tasks=tasks), base_url="http://srv"
    )


CUSTOM_VOICE = {
    "tts_api": "qwen3_cpp",
    "qwen3_release": "custom_voice",
    "speaker": "ryan",
    "language": "English",
}
BASE = {"tts_api": "qwen3_cpp", "qwen3_release": "base", "language": "English"}


class TestClientTargets:
    def test_python_media_server_models_keep_the_default_client(self):
        assert bench._tts_client_targets(_ctx(tasks=[])) == {}
        assert bench._tts_client_targets(_ctx({"add_bos_token": True})) == {}
        assert bench._tts_client_targets(SimpleNamespace(all_params=[])) == {}

    @pytest.mark.parametrize(
        "repo, release, speaker, described",
        [
            ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "base", None, False),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "custom_voice", "ryan", False),
            ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voice_design", None, True),
            ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "base", None, False),
            ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "custom_voice", "ryan", False),
        ],
    )
    def test_qwen3_settings_come_from_the_eval_config(
        self, repo, release, speaker, described
    ):
        from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

        targets = bench._tts_client_targets(
            SimpleNamespace(all_params=ALL_EVAL_CONFIGS[repo])
        )
        client = tts_client.make_tts_client(targets)
        assert isinstance(client, Qwen3CppTTSClient)
        assert (client.release, client.speaker, client.language) == (
            release,
            speaker,
            "English",
        )
        assert bool(client.description) == described

    def test_the_galaxy_deployment_speaks_to_vllm_omni(self):
        from reference_config.evals.eval_config import ALL_EVAL_CONFIGS

        repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ctx = SimpleNamespace(
            all_params=ALL_EVAL_CONFIGS[repo], device=SimpleNamespace(name="GALAXY")
        )
        client = tts_client.make_tts_client(bench._tts_client_targets(ctx))
        assert isinstance(client, tts_client.VllmOmniTTSClient)
        assert (client.clone_mode, client.language) == ("voice_only", "English")


class TestMediaServerBenchmark:
    def test_request_and_metrics_are_unchanged(self, server, clock):
        log = server(_Response(payload={"audio": "QUJD", "duration": 3.0}))
        clock(10.0, 10.5, 12.0)
        result = asyncio.run(
            MediaServerTTSClient().timed_synthesize("http://srv/v1/audio/speech", "hi")
        )
        assert log[0] == {"session": {}}
        request = log[1]
        assert request["json"] == {"text": "hi", "response_format": "json"}
        assert request["headers"] == {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        assert request["timeout"].total == 120
        assert result.ok and result.elapsed_s == 2.0
        assert result.ttft_ms == 500.0 and result.audio_seconds == 3.0

    def test_wav_response_is_rejected(self, server):
        server(_Response(content_type="audio/wav"))
        result = asyncio.run(MediaServerTTSClient().timed_synthesize("u", "hi"))
        assert not result.ok


class TestQwen3Benchmark:
    def test_ttft_is_the_first_audio_byte_and_duration_the_samples(self, server, clock):
        audio = b"\x00\x00" * 24000
        log = server(_Response(chunks=[_streamed_wav_header(), audio[:10], audio[10:]]))
        clock(0.0, 0.1, 4.0, 4.5, 5.0)
        client = tts_client.make_tts_client(CUSTOM_VOICE)
        result = asyncio.run(client.timed_synthesize("u", "hi"))
        assert log[1]["json"] == {
            "text": "hi",
            "language": "English",
            "speaker": "ryan",
        }
        assert result.ok
        assert result.ttft_ms == 4000.0 and result.elapsed_s == 5.0
        assert result.audio_seconds == 1.0

    def test_bearer_follows_the_environment(self, server, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "secret")
        log = server(_Response(chunks=[_streamed_wav_header() + b"\0" * 24000]))
        client = tts_client.make_tts_client(CUSTOM_VOICE)
        assert asyncio.run(client.timed_synthesize("u", "hi")).ok
        assert log[0]["session"]["headers"] == {"Authorization": "Bearer secret"}

    @pytest.mark.parametrize(
        "response",
        [
            _Response(status=401, chunks=[b'{"error": "unauthorized"}']),
            _Response(chunks=[_streamed_wav_header() + b"\0" * 100]),
            _Response(chunks=[b"not a wav"]),
        ],
    )
    def test_failed_or_truncated_streams_fail(self, server, response):
        server(response)
        client = tts_client.make_tts_client(CUSTOM_VOICE)
        assert not asyncio.run(client.timed_synthesize("u", "hi")).ok


class TestRunBenchmark:
    def test_base_clones_the_payload_clip_and_reads_the_default_text(self, server):
        log = server(_Response(chunks=[_streamed_wav_header() + b"\0" * 48000]))
        statuses = bench._run_tts_benchmark(_ctx(BASE), 2)
        assert [s.status for s in statuses] == [True, True]
        assert all(s.audio_duration == 1.0 and s.rtr > 0 for s in statuses)
        assert all(s.text == bench.DEFAULT_TTS_TEXT for s in statuses)
        assert log[1]["url"] == "http://srv/v1/audio/speech"
        fields = {f[0]["name"]: f[2] for f in log[1]["data"]._fields}
        assert fields["text"] == bench.DEFAULT_TTS_TEXT
        assert fields["language"] == "English"
        assert parse_pcm16_wav_header(fields["voice"])[:2] == (16000, 1)

    def test_voice_design_sends_its_description_and_no_clip(self, server):
        log = server(_Response(chunks=[_streamed_wav_header() + b"\0" * 24000]))
        targets = {
            "tts_api": "qwen3_cpp",
            "qwen3_release": "voice_design",
            "description": "A calm voice.",
            "language": "English",
        }
        statuses = bench._run_tts_benchmark(_ctx(targets), 1)
        assert statuses[0].status
        assert log[1]["json"] == {
            "text": bench.DEFAULT_TTS_TEXT,
            "language": "English",
            "description": "A calm voice.",
        }

    def test_a_warm_up_request_precedes_the_timed_ones(self, monkeypatch):
        calls = []

        async def fake_generate(ctx, client, text, voice=None):
            calls.append(text)
            return (len(calls) > 1, 1.0, 100.0, 1.0, 1.0)

        monkeypatch.setattr(bench, "_generate_speech", fake_generate)
        statuses = bench._run_tts_benchmark(_ctx(CUSTOM_VOICE), 2)
        assert len(calls) == 3
        assert [s.status for s in statuses] == [True, True]

    def test_failures_are_reported_not_raised(self, server):
        server(_Response(status=401))
        statuses = bench._run_tts_benchmark(_ctx(CUSTOM_VOICE), 1)
        assert statuses[0].status is False and statuses[0].ttft_ms is None

    def test_python_media_server_reads_the_task_text(self, server):
        log = server(_Response(payload={"audio": "QUJD", "duration": 1.0}))
        ctx = _ctx(tasks=[SimpleNamespace(task_name="tts_generation")])
        statuses = bench._run_tts_benchmark(ctx, 1)
        assert statuses[0].status and statuses[0].audio_duration == 1.0
        assert log[1]["json"]["text"] == "tts_generation"


def test_voice_clip_is_a_normalized_pcm16_wav():
    wav = bench._benchmark_voice_clip().wav_bytes
    assert parse_pcm16_wav_header(wav) == (16000, 1, 44)
    samples = struct.unpack(f"<{(len(wav) - 44) // 2}h", wav[44:])
    assert len(samples) == 16000 * 5.28
    assert 0.85 * 32767 < max(abs(s) for s in samples) <= 0.9 * 32767 + 1


def test_playback_waits_only_when_audio_runs_out():
    # 0.5 s of audio at 0.2 s plays until 0.7 s; the next chunk lands at 0.9 s.
    arrivals = [(0.2, 0.5), (0.9, 1.0), (1.2, 2.0)]
    total, longest = bench._playback_wait(arrivals)
    assert total == pytest.approx(0.2) and longest == pytest.approx(0.2)


class _SweepClient:
    has_default_voice = False

    def __init__(self, fail_at=None):
        self.batches = []
        self.fail_at = fail_at

    def describe(self):
        return "sweep"

    async def timed_synthesize(self, url, text, *, voice=None, timeout_s=120):
        self.batches.append(text)
        if self.fail_at is not None and len(self.batches) == self.fail_at:
            return tts_client.TimedSynthesis(False)
        return tts_client.TimedSynthesis(
            True, 2.0, 250.0, 3.0, [(0.25, 1.0), (1.0, 3.0)]
        )


class TestConcurrencySweep:
    def _run(self, monkeypatch, client, levels):
        checks = {}

        def fake_checks(ctx, ttfa_ms, rtr):
            checks.update(ttfa_ms=ttfa_ms, rtr=rtr)
            return {}, None

        monkeypatch.setattr(bench, "make_tts_client", lambda targets: client)
        monkeypatch.setattr(bench, "_tts_target_checks", fake_checks)
        monkeypatch.setattr(bench, "block_id", lambda ctx: "id")
        block = bench._run_tts_concurrency_sweep(_ctx(BASE), levels)
        return block.data["records"], checks

    def test_one_row_per_prompt_and_concurrency(self, monkeypatch):
        client = _SweepClient()
        records, checks = self._run(monkeypatch, client, [1, 2])
        assert [(r["prompt"], r["concurrency"]) for r in records] == [
            (p, c) for p in bench.SWEEP_PROMPTS for c in (1, 2)
        ]
        # One warm-up, then C requests per row.
        assert len(client.batches) == 1 + 3 * (1 + 2)
        row = records[0]
        assert row["rtr_min"] == pytest.approx(1.5)
        assert row["stream_rate_min"] == pytest.approx(3.0 / 1.75)
        assert row["audio_s_per_s"] == pytest.approx(1.5)
        assert checks == {"ttfa_ms": pytest.approx(250.0), "rtr": pytest.approx(1.5)}

    def test_the_report_prints_the_legend_and_prompts_below_the_rows(self, monkeypatch):
        from report_module.renderers import render_benchmarks

        monkeypatch.setattr(bench, "make_tts_client", lambda targets: _SweepClient())
        monkeypatch.setattr(bench, "_tts_target_checks", lambda *a: ({}, None))
        monkeypatch.setattr(bench, "block_id", lambda ctx: "id")
        block = bench._run_tts_concurrency_sweep(_ctx(BASE), [1, 2])
        text = render_benchmarks(block, {})
        legend = text.index("**Legend**")
        assert text.index("| short") < legend < text.index("**Prompts**")
        assert "- **short**: 7 words. “Good morning. Today is a beautiful day.”" in text
        assert "- **long**: 119 words. “It was the best of times" in text

    def test_a_failed_request_fails_the_gate(self, monkeypatch):
        # Call 6 (after the warm-up) is in the short prompt's C=3 row, not at C=1.
        records, checks = self._run(monkeypatch, _SweepClient(fail_at=6), [1, 2, 3])
        assert sum(r["failed_requests"] for r in records) == 1
        assert checks["rtr"] == 0.0

    def test_the_sweep_runs_when_the_task_asks_for_it(self, monkeypatch):
        targets = dict(BASE, benchmark_concurrency=[1, 4])
        called = {}
        monkeypatch.setattr(bench, "require_health", lambda ctx: None)
        monkeypatch.setattr(bench, "_tts_client_targets", lambda ctx: targets)
        monkeypatch.setattr(
            bench,
            "_run_tts_concurrency_sweep",
            lambda ctx, levels: called.setdefault("levels", levels),
        )
        ctx = _ctx(targets)
        ctx.model_spec = SimpleNamespace(model_name="m")
        ctx.device = SimpleNamespace(name="GALAXY")
        bench.run_tts_benchmark(ctx)
        assert called["levels"] == [1, 4]
