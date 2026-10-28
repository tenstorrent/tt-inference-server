# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import socketserver
import threading
import time
import wave
from types import SimpleNamespace

import pytest

from test_module.benchmark_tests import tts_load_harness as harness
from test_module.benchmark_tests import tts_staged_harness as staged

CHUNK_BYTES = 1920


class _FakeTts(socketserver.ThreadingTCPServer):
    """Chunked WAV responder that records peak concurrency and request bodies."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, chunks: int = 4) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.chunks = chunks
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.content_types: list = []


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        srv = self.server
        headers = {}
        self.rfile.readline()
        while True:
            line = self.rfile.readline().decode().strip()
            if not line:
                break
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        self.rfile.read(int(headers.get("content-length", 0)))
        with srv.lock:
            srv.active += 1
            srv.peak = max(srv.peak, srv.active)
            srv.content_types.append(headers.get("content-type", ""))
        try:
            out = self.wfile
            out.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            for payload in [b"\0" * 44] + [b"\1\0" * (CHUNK_BYTES // 2)] * srv.chunks:
                out.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
                out.flush()
                time.sleep(0.01)
            out.write(b"0\r\n\r\n")
        except OSError:
            pass  # client cancelled
        finally:
            with srv.lock:
                srv.active -= 1


@pytest.fixture
def fake_server():
    srv = _FakeTts()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _row(t_s, total_s, text, ids, inflight=0, outcome="completed", chunks=3):
    return {
        "arrival_wall_us": int((1_800_000_000 + t_s) * 1e6),
        "total_us": int(total_s * 1e6),
        "text": text,
        "text_chars": len(text),
        "prompt_tokens": 10 + ids + (2 if ids else 0),
        "prompt_speech_ids": ids,
        "inflight_on_arrival": inflight,
        "outcome": outcome,
        "chunks": chunks,
        "chunk_offsets_us": [80_000 * (i + 1) for i in range(chunks)],
        "ttfc_us": 80_000,
        "audio_samples": 48000 * chunks,
        "sample_rate_hz": 48000,
    }


def _capture(tmp_path):
    rows = [
        # warmup: two rounds of two, each drained before the next
        _row(0, 1, "hi there", 0),
        _row(0.01, 1, "hi there", 0, 1),
        _row(1.5, 1, "hi there", 50),
        _row(1.51, 1, "hi there", 50, 1),
        # ladder level 1 (window 10 s), then level 2
        _row(20, 2, "alpha beta.", 100),
        _row(23, 2, "gamma.", 200),
        _row(30, 3, "alpha beta.", 200),
        _row(30.1, 3, "delta epsilon.", 100, 1),
        _row(33.2, 2, "gamma.", 100, 1),
        # probe
        _row(80, 0.2, "probe text", 100, outcome="cancelled", chunks=2),
        _row(80.3, 0.2, "probe text", 200, outcome="cancelled", chunks=2),
    ]
    path = tmp_path / "capture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _dataset(tmp_path):
    ds = staged.build_dataset(
        str(_capture(tmp_path)),
        "2027-01-15T00:00:00",
        "2027-01-16T00:00:00",
        [1, 2],
        duration_s=10,
    )
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(ds))
    return ds, path


def test_build_dataset_splits_phases_and_counts(tmp_path):
    ds, _ = _dataset(tmp_path)
    assert [len(r["requests"]) for r in ds["warmup"]] == [2, 2]
    assert [s["concurrency"] for s in ds["ladder"]["stages"]] == [1, 2]
    assert [s["captured"]["n"] for s in ds["ladder"]["stages"]] == [2, 3]
    counts = {t["text"]: t["count"] for t in ds["ladder"]["texts"]}
    assert counts == {"alpha beta.": 2, "gamma.": 2, "delta epsilon.": 1}
    assert {t["text_tokens"] for t in ds["ladder"]["texts"]} == {10}
    voices = {v["voice_speech_ids"]: v["count"] for v in ds["ladder"]["voices"]}
    assert voices == {100: 3, 200: 2}
    assert ds["probe"]["cancel_after_chunks"] == 2
    assert len(ds["probe"]["requests"]) == 2


def test_build_dataset_rejects_schedule_that_does_not_fit(tmp_path):
    with pytest.raises(ValueError, match="level 1"):
        staged.build_dataset(
            str(_capture(tmp_path)),
            "2027-01-15T00:00:00",
            "2027-01-16T00:00:00",
            [1, 1],
            duration_s=30,
        )


def test_sampler_is_seeded_and_weighted():
    ladder = {
        "texts": [
            {"text": "a", "text_tokens": 5, "count": 9},
            {"text": "b", "text_tokens": 6, "count": 1},
        ],
        "voices": [{"voice_speech_ids": 100, "count": 1}],
    }
    first = [staged.Sampler(ladder, 7).draw() for _ in range(1)]
    s1, s2 = staged.Sampler(ladder, 7), staged.Sampler(ladder, 7)
    seq1 = [s1.draw()["text"] for _ in range(200)]
    assert seq1 == [s2.draw()["text"] for _ in range(200)]
    assert 150 < seq1.count("a") < 200
    assert first[0]["voice_speech_ids"] == 100


def test_isl_counts_voice_wrapper_tokens():
    assert staged.isl_tokens(26, 0) == 26
    assert staged.isl_tokens(26, 177) == 205


def test_target_sends_voice_as_multipart():
    tgt = harness.Target(text="hello", voice_wav=b"RIFFdata")
    head, body = tgt.request.split(b"\r\n\r\n", 1)
    assert b"Content-Type: multipart/form-data; boundary=" in head
    assert b'name="text"\r\n\r\nhello\r\n' in body
    assert b'filename="voice.wav"' in body and b"RIFFdata" in body
    assert f"Content-Length: {len(body)}".encode() in head
    assert b"application/json" in harness.Target(text="hello").request


def test_run_end_to_end_against_fake_server(tmp_path, fake_server):
    _, path = _dataset(tmp_path)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=fake_server.server_address[1],
        path=harness.DEFAULT_PATH,
        key="k",
        timeout=10,
        dataset=str(path),
        levels="1,2",
        duration=0.5,
        phases="warmup,ladder,probe",
        seed=1,
        voice_dir=str(tmp_path / "voices"),
        voice_source=None,
        speech_ids_per_s=50.0,
        bootstrap_chars=100,
        out=str(tmp_path / "out.json"),
    )
    doc = staged.run(args)

    # Voice clips are cut from bootstrapped speech at the configured token rate.
    with wave.open(str(tmp_path / "voices" / "voice_200.wav")) as w:
        assert w.getframerate() == staged.VOICE_RATE_HZ
        assert w.getnframes() == 200 // 50 * staged.VOICE_RATE_HZ

    stages = [s["stage"] for s in doc["stages"]]
    assert stages == ["warmup1", "warmup2", "c1", "c2", "probe"]
    recs = doc["records"]
    assert all(r["ok"] for r in recs)
    by = {s: [r for r in recs if r["stage"] == s] for s in stages}
    assert len(by["warmup1"]) == 2 and len(by["c1"]) >= 1 and len(by["c2"]) >= 2
    assert all(r["capped"] and r["nchunks"] == 2 for r in by["probe"])
    assert {r["isl"] for r in by["probe"]} == {112, 212}
    assert fake_server.peak <= 2
    assert any(ct.startswith("multipart/form-data") for ct in fake_server.content_types)

    saved = json.loads((tmp_path / "out.json").read_text())
    table = staged.report_table(saved)
    assert "c2" in table and "capture" in table
    assert "112" in staged.probe_table(saved)


def test_charts_cover_ladder_and_probe(tmp_path, fake_server):
    pytest.importorskip("matplotlib")
    test_run_end_to_end_against_fake_server(tmp_path, fake_server)
    doc = json.loads((tmp_path / "out.json").read_text())
    paths = staged.render_charts(doc, str(tmp_path / "staged"))
    names = sorted(p.rsplit("/", 1)[1] for p in paths)
    assert names == [
        "staged_concurrency_fc.png",
        "staged_probe_fc_isl.png",
        "staged_throughput_fc.png",
    ]
    assert all((tmp_path / n).stat().st_size > 0 for n in names)
