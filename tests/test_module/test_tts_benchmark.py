# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""TTS benchmark request paths and the tt-tts load sweep, against fake servers
(no numpy, torch or soundfile)."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

from report_module.status import TestStatus
from test_module._test_common import ReportCheckTypes
from test_module.benchmark_tests import tts_benchmark_tests as bench
from test_module.benchmark_tests import tts_load_harness as harness
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


# ------------------------------------------- tt-tts chunk-aware load benchmark

_HEADER = b"\0" * tts_client.STREAM_WAV_HEADER_BYTES
_CHUNK = b"\1\0" * 480  # 10 ms of 48 kHz mono PCM16


class _FakeTtTts:
    """A real HTTP server streaming one HTTP chunk per audio chunk."""

    def __init__(self, chunks=4, gap_s=0.01, status=200, header_only=False):
        self.chunks, self.gap_s = chunks, gap_s
        self.status, self.header_only = status, header_only
        self.payloads = []
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    async def _speech(self, request):
        self.payloads.append(await request.json())
        if self.status != 200:
            return web.Response(status=self.status, text="boom")
        response = web.StreamResponse()
        response.enable_chunked_encoding()
        await response.prepare(request)
        await response.write(_HEADER)
        for _ in range(0 if self.header_only else self.chunks):
            await asyncio.sleep(self.gap_s)
            await response.write(_CHUNK)
        await response.write_eof()
        return response

    def _serve(self):
        asyncio.set_event_loop(self._loop)
        app = web.Application()
        app.router.add_post(harness.SPEECH_PATH, self._speech)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self._started.set()
        self._loop.run_forever()

    def __enter__(self):
        self._thread.start()
        self._started.wait(5)
        return self

    def __exit__(self, *exc):
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)

    def target(self, text="hello", **kw):
        return harness.Target(
            url=f"http://127.0.0.1:{self.port}{harness.SPEECH_PATH}", text=text, **kw
        )


async def _one(target, **kw):
    async with harness._session() as session:
        return await harness.request(session, target, **kw)


def test_stream_speech_times_every_http_chunk():
    with _FakeTtTts(chunks=4, gap_s=0.02) as server:
        body = bytearray()
        rec = asyncio.run(_one(server.target(speech_ids=[1, 2]), body=body))
    assert rec["ok"] and rec["error"] is None
    assert rec["nchunks"] == 4
    assert len(rec["gaps"]) == 2  # TC and 4C+: the gaps after the second chunk
    assert rec["ttfb_s"] >= 0.02 and rec["sc_s"] >= 0.015
    assert rec["audio_s"] == pytest.approx(0.04)
    assert rec["chars"] == 5 and rec["voice_speech_ids"] == 2
    assert bytes(body) == _HEADER + _CHUNK * 4
    assert server.payloads == [{"text": "hello", "speech_ids": [1, 2]}]


@pytest.mark.parametrize(
    "server_kw, error",
    [({"status": 500}, "http 500: boom"), ({"header_only": True}, "empty audio")],
)
def test_stream_speech_failures_keep_a_lifetime(server_kw, error):
    with _FakeTtTts(**server_kw) as server:
        rec = asyncio.run(_one(server.target()))
    assert not rec["ok"] and rec["error"].startswith(error)
    assert rec["t_end"] >= rec["t_send"]


def test_preflight_raises_on_a_failing_server():
    with _FakeTtTts(status=503) as server:
        with pytest.raises(harness.PreflightError):
            harness.sweep(
                harness.SweepConfig(concurrency=(1,), duration=0.1, warmup=0.0),
                server.target(),
            )


def test_sweep_runs_levels_in_order_and_aggregates(tmp_path):
    out = tmp_path / "results.json"
    texts = iter(["a" * 10, "b" * 20] * 1000)
    with _FakeTtTts(chunks=3, gap_s=0.005) as server:
        base = server.target()
        doc = harness.sweep(
            harness.SweepConfig(
                concurrency=(2, 1, 2), duration=0.3, warmup=0.1, settle=0.0, out=out
            ),
            base,
            draw=lambda: base.with_prompt(next(texts), []),
        )
    assert [lvl["conc"] for lvl in doc["levels"]] == [2, 1]
    assert json.loads(out.read_text())["meta"]["finished"]
    rows, chars = harness.aggregate(doc)
    assert [row["conc"] for row in rows] == [2, 1]
    assert 10 <= chars <= 20
    for row in rows:
        assert row["n_ok"] > 0 and row["n_err"] == 0
        assert row["chunks"] == 3
        assert 0 < row["C"] <= row["conc"] + 1e-9
        assert (
            row["ttft_p50"] > 0 and row["ttf4_p50"] != row["ttf4_p50"]
        )  # 3 chunks: no 4C+


def test_aggregate_counts_window_arrivals_and_skips_capped_chars():
    def rec(t_send, ok=True, capped=False):
        return {
            "conc": 1, "t_send": t_send, "t_end": t_send + 1.0, "ok": ok,
            "capped": capped, "chars": 100, "nchunks": 4, "ttfb_s": 0.1,
            "sc_s": 0.2, "tc_gaps": [0.3, 0.4, 0.5],
        }  # fmt: skip

    doc = {
        "meta": {"chars_per_request": 100},
        "levels": [{"conc": 1, "window_start": 10.0, "window_end": 20.0}],
        "records": [rec(9.5), rec(11.0), rec(12.0, capped=True), rec(13.0, ok=False)],
    }
    (row,), _ = harness.aggregate(doc)
    assert (row["n_ok"], row["n_err"]) == (2, 1)
    assert row["rps"] == pytest.approx(0.2)
    assert row["mchar_h"] == pytest.approx(100 / 10 * 3600 / 1e6)  # capped: no credit
    assert row["ttfb_p50"] == pytest.approx(100.0)
    assert row["ttft_max"] == pytest.approx(300.0)
    assert row["ttf4_max"] == pytest.approx(500.0)
    assert row["C"] == pytest.approx(3.5 / 10)


def _load_ctx(workflow_args="", **fields):
    return SimpleNamespace(
        runtime_config=SimpleNamespace(workflow_args=workflow_args), **fields
    )


def test_load_params_defaults_and_overrides(caplog):
    assert bench._load_params(_load_ctx()) == bench.LoadParams()
    params = bench._load_params(
        _load_ctx(
            "concurrency=16,4,16 isl=128 warmup=5 duration=30 prewarm "
            "voice_ids=/v.json eval_requests=10"
        )
    )
    assert params == bench.LoadParams(
        concurrency=(16, 4),
        isl=128,
        voice_ids="/v.json",
        warmup=5,
        duration=30,
        prewarm=True,
    )
    assert "ignores" not in caplog.text  # eval_* belongs to the correctness eval
    assert bench._load_params(_load_ctx("prewarm_only=1")).prewarm_only


@pytest.mark.parametrize(
    "args", ["concurrency=", "concurrency=0", "isl=-1", "duration=x", "prewarm=maybe"]
)
def test_load_params_rejects_invalid_values(args):
    with pytest.raises(ValueError):
        bench._load_params(_load_ctx(args))


def _harness_row(conc, errors=0, **latency):
    row = {"conc": conc, "n_ok": 20, "n_err": errors, "rps": 2.0, "mchar_h": 1.5,
           "C": conc * 0.9, "chunks": 12.0}  # fmt: skip
    for key in ("ttfb", "ttfs", "ttft", "ttf4"):
        for stat in ("p50", "p95", "max"):
            row[f"{key}_{stat}"] = latency.get(f"{key}_{stat}", 50.0)
    return row


def test_report_rows_rename_metrics_and_drop_nan():
    (row,) = bench._report_rows([_harness_row(4, ttf4_max=math.nan)])
    assert row["concurrency"] == 4 and row["num_successful"] == 20
    assert row["fc_p50_ms"] == 50.0 and row["c4_p95_ms"] == 50.0
    assert row["c4_max_ms"] is None


_TTS2_CTX = SimpleNamespace(
    model_spec=SimpleNamespace(
        hf_model_repo="propritery/tts-2", cli_args={"device": "quad_galaxy"}
    )
)


def test_load_targets_are_inclusive_and_grade_the_worst_level():
    at_target = dict(
        ttfb_p50=100.0, ttfb_p95=125.0, ttfs_p50=300.0, ttfs_p95=300.0,
        ttft_p50=600.0, ttft_p95=600.0, ttf4_p50=960.0, ttf4_p95=960.0,
    )  # fmt: skip
    levels = bench.TTS_LOAD_CONCURRENCY
    rows = bench._report_rows([_harness_row(c, **at_target) for c in levels])
    _, verdict = bench._load_target_checks(_TTS2_CTX, rows, levels)
    assert verdict == ReportCheckTypes.PASS

    slow = [_harness_row(c, **{**at_target, "ttfb_p95": 126.0 if c == 16 else 1.0})
            for c in levels]  # fmt: skip
    _, verdict = bench._load_target_checks(_TTS2_CTX, bench._report_rows(slow), levels)
    assert verdict == ReportCheckTypes.FAIL


@pytest.mark.parametrize(
    "rows",
    [
        [_harness_row(c) for c in (1, 2, 4, 8)],
        [_harness_row(c, errors=int(c == 16)) for c in (1, 2, 4, 8, 16)],
        [
            _harness_row(c, ttfs_p95=math.nan if c == 16 else 1.0)
            for c in (1, 2, 4, 8, 16)
        ],
    ],
    ids=("missing-level", "request-error", "missing-latency"),
)
def test_incomplete_sweeps_fail_the_integrity_check(rows):
    checks, verdict = bench._load_target_checks(
        _TTS2_CTX, bench._report_rows(rows), bench.TTS_LOAD_CONCURRENCY
    )
    assert verdict == ReportCheckTypes.FAIL
    assert all(
        tier["benchmark_integrity_check"] == ReportCheckTypes.FAIL
        for tier in checks.values()
    )


def _ladder_sizes():
    with open(bench.TTS_TRAFFIC_DATASET) as f:
        return [v["voice_speech_ids"] for v in json.load(f)["ladder"]["voices"]]


def test_traffic_draw_sends_prefixes_of_the_real_voice():
    base = harness.Target(url="http://x/v1/audio/speech")
    voice = list(range(max(_ladder_sizes())))
    draw = bench._traffic_draw(base, voice)
    for _ in range(50):
        target = draw()
        assert target.text and target.speech_ids == voice[: len(target.speech_ids)]


def test_traffic_draw_is_seeded_and_rejects_short_voices():
    base = harness.Target(url="http://x/v1/audio/speech")
    first = [bench._traffic_draw(base)() for _ in range(2)]
    assert first[0] == first[1]  # same seed, same first prompt
    with pytest.raises(ValueError):
        bench._traffic_draw(base, [1, 2, 3])


@pytest.mark.parametrize("content", [[], [-1], [65536], ["7"]])
def test_voice_ids_file_must_hold_codec_ids(tmp_path, content):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        bench._load_voice_ids(str(path))


def _tt_tts_ctx(tmp_path, workflow_args=""):
    task = SimpleNamespace(model_kwargs={"tts_api": tts_client.TTS_API_TT_TTS})
    return _load_ctx(
        workflow_args,
        base_url="http://127.0.0.1:9",
        output_path=str(tmp_path),
        all_params=SimpleNamespace(tasks=[task]),
        model_spec=SimpleNamespace(
            model_name="tts-2",
            hf_model_repo="propritery/tts-2",
            cli_args={"device": "quad_galaxy"},
        ),
        device=SimpleNamespace(name="QUAD_GALAXY"),
    )


def test_run_tts_benchmark_routes_tt_tts_to_the_load_sweep(monkeypatch, tmp_path):
    observed = {}
    monkeypatch.setattr(bench, "require_health", lambda ctx, requirement=None: None)
    monkeypatch.setattr(bench, "block_id", lambda ctx: "tts-quad")

    def fake_sweep(cfg, target, draw=None):
        observed.update(cfg=cfg, target=target, draw=draw)
        return {"doc": True}

    monkeypatch.setattr(bench.tts_load_harness, "sweep", fake_sweep)
    monkeypatch.setattr(
        bench.tts_load_harness,
        "aggregate",
        lambda doc: ([_harness_row(1), _harness_row(16, errors=2)], 3543),
    )
    block = bench.run_tts_benchmark(_tt_tts_ctx(tmp_path, "concurrency=1,16"))

    cfg = observed["cfg"]
    assert cfg.concurrency == (1, 16) and cfg.prewarm == 0.0
    assert cfg.ramp == bench.TTS_LOAD_WARMUP_SECONDS  # replay: staggered starts
    assert cfg.out == tmp_path / bench.TTS_RAW_RESULTS_FILENAME
    assert observed["target"].url == "http://127.0.0.1:9/v1/audio/speech"
    assert callable(observed["draw"])
    assert block.title == "Text-to-Speech Load Benchmark" and block.id == "tts-quad"
    assert block.data["Benchmarks"]["num_requests"] == 42
    assert block.data["Benchmarks"]["target_check"] == ReportCheckTypes.FAIL
    assert len(block.data["Latency by Concurrency"]) == 2


def test_prewarm_only_warms_and_returns_a_skipped_block(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "require_health", lambda ctx, requirement=None: None)
    monkeypatch.setattr(bench, "block_id", lambda ctx: None)
    calls = []
    monkeypatch.setattr(
        bench.tts_load_harness,
        "prewarm",
        lambda target, conc, secs, draw=None: calls.append((conc, secs)) or 7,
    )
    block = bench.run_tts_benchmark(_tt_tts_ctx(tmp_path, "prewarm_only isl=64"))
    assert calls == [(bench.TTS_PREWARM_CONCURRENCY, bench.TTS_PREWARM_SECONDS)]
    assert block.data["status"] == TestStatus.SKIP.value
    assert block.data["Prewarm"]["requests_completed"] == 7
