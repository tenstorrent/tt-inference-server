# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the TTS same-prompt correctness check."""

from __future__ import annotations

import json
import socket
import threading
from types import SimpleNamespace

import pytest

from test_module.benchmark_tests import tts_correctness as correctness
from test_module.benchmark_tests import tts_load_harness as harness

TEXT = "the quick brown fox jumps over the lazy dog"


def _args(**kw):
    base = dict(
        no_asr=True,
        asr_model="tiny",
        asr_threads=1,
        max_wer=0.05,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _rec(tag, i, sha="a", audio_s=2.0, ok=True, error=None):
    return {
        "tag": tag,
        "i": i,
        "ok": ok,
        "status": 200 if ok else 429,
        "error": error,
        "ttfb_s": 0.1,
        "audio_s": audio_s,
        "sha256": sha,
        "wav": f"{tag}_{i:03d}.wav",
    }


class TestWordErrors:
    def test_identical(self):
        assert correctness.word_errors(["a", "b"], ["a", "b"]) == {
            "wer": 0.0,
            "sub": 0,
            "del": 0,
            "ins": 0,
        }

    def test_substitution_deletion_insertion_are_separated(self):
        ref = TEXT.split()
        assert correctness.word_errors(ref, ref[:-2])["del"] == 2
        assert correctness.word_errors(ref, ref + ["dog", "dog"])["ins"] == 2
        swapped = ref[:3] + ["cat"] + ref[4:]
        out = correctness.word_errors(ref, swapped)
        assert (out["sub"], out["del"], out["ins"]) == (1, 0, 0)
        assert out["wer"] == pytest.approx(1 / len(ref))

    def test_empty_hypothesis_is_all_deletions(self):
        assert correctness.word_errors(["a", "b", "c"], [])["wer"] == 1.0


def test_normalize_drops_case_and_punctuation():
    assert correctness.normalize("Hello, World! It's done.") == correctness.normalize(
        "hello world its done"
    )


class TestScore:
    def test_pcm_and_duration_are_information_only(self):
        records = [_rec("ref", i) for i in range(3)]
        records += [_rec("load", 0), _rec("load", 1, sha="b", audio_s=6.0)]
        summary = correctness.score(records, TEXT, _args())
        assert summary["deterministic"] is True
        assert all(r["pass"] for r in records)
        assert summary["load_failures"] == 0
        assert records[4]["pcm_matches_ref"] is False
        assert records[4]["duration_ratio"] == pytest.approx(3.0)

    def test_failed_request_fails_with_its_error(self):
        records = [_rec("ref", 0), _rec("load", 0, ok=False, error="http 429: full")]
        correctness.score(records, TEXT, _args())
        assert records[1]["reasons"] == ["http 429: full"]

    def test_fails_only_on_wer_against_the_text(self, monkeypatch):
        words = TEXT.split()
        transcripts = {
            "ref_000.wav": TEXT,
            "ref_001.wav": " ".join(words * 5),  # babbling reference: 400% WER
            "load_000.wav": " ".join(words[:-1]),  # 1/9 WER
            "load_001.wav": TEXT,
        }
        monkeypatch.setattr(
            correctness, "transcribe", lambda paths, model, threads: transcripts
        )
        records = [_rec("ref", 0, sha="0"), _rec("ref", 1, sha="1", audio_s=120.0)]
        records += [_rec("load", 0, sha="2"), _rec("load", 1, sha="3", audio_s=9.0)]
        summary = correctness.score(records, TEXT, _args(no_asr=False, max_wer=0.1))
        assert summary["wer_limit"] == 0.1
        assert [r["pass"] for r in records] == [True, False, False, True]
        assert records[2]["del"] == 1
        assert summary["reference_failures"] == 1
        assert summary["load_failures"] == 1

    def test_without_successful_references_load_is_still_scored(self):
        records = [_rec("ref", 0, ok=False, error="http 429: full"), _rec("load", 0)]
        summary = correctness.score(records, TEXT, _args())
        assert summary["references"] == 0
        assert summary["reference_failures"] == 1
        assert summary["reference_duration_s"] is None
        assert records[1]["pass"] is True
        assert records[1]["duration_ratio"] is None
        assert "References: 0 ok, 1 failed" in correctness.report(summary, records)


def test_report_lists_failures_with_transcripts(monkeypatch):
    transcripts = {"ref_000.wav": TEXT, "load_000.wav": "the quick brown fox"}
    monkeypatch.setattr(
        correctness, "transcribe", lambda paths, model, threads: transcripts
    )
    records = [_rec("ref", 0, sha="0"), _rec("load", 0, sha="1")]
    records += [_rec("load", 1, ok=False, error="http 429: full")]
    summary = correctness.score(records, TEXT, _args(no_asr=False, max_wer=0.1))
    out = correctness.report(summary, records)
    assert "load WER: max 55.6%" in out
    assert "Load: 0/2 pass" in out
    assert "load #000 (2.0 s) wer 55.6% > 10.0%" in out
    assert '"the quick brown fox"' in out
    assert "load #001 http 429: full" in out


def _serve(chunks: list[bytes], requests: int, seen: list[dict]):
    """One-connection-per-request fake server streaming ``chunks`` as chunked WAV."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(requests)

    def loop():
        for _ in range(requests):
            conn, _ = srv.accept()
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += conn.recv(65536)
                head, body = data.split(b"\r\n\r\n", 1)
                length = int(
                    [
                        line.split(b":")[1]
                        for line in head.split(b"\r\n")
                        if line.lower().startswith(b"content-length")
                    ][0]
                )
                while len(body) < length:
                    body += conn.recv(65536)
                seen.append(json.loads(body))
                out = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                for chunk in chunks:
                    out += b"%x\r\n%s\r\n" % (len(chunk), chunk)
                conn.sendall(out + b"0\r\n\r\n")
        srv.close()

    threading.Thread(target=loop, daemon=True).start()
    return srv.getsockname()[1]


def test_send_saves_wavs_and_sends_speech_ids(tmp_path):
    header = b"RIFF" + b"\0" * 40
    pcm = [b"\1\2" * 480, b"\3\4" * 960]
    seen: list[dict] = []
    port = _serve([header] + pcm, 3, seen)
    (tmp_path / "wav").mkdir()
    tgt = harness.Target(host="127.0.0.1", port=port, text="hi", speech_ids=[7, 8])

    records = correctness.send(tgt, "load", 3, 2, str(tmp_path))

    assert seen == [{"text": "hi", "speech_ids": [7, 8]}] * 3
    assert [r["i"] for r in records] == [0, 1, 2]
    assert all(r["ok"] for r in records)
    assert len({r["sha256"] for r in records}) == 1
    wav = (tmp_path / "wav" / "load_001.wav").read_bytes()
    assert wav == header + b"".join(pcm)
    assert records[0]["audio_s"] == pytest.approx(1440 / harness.SAMPLE_RATE)


def test_target_without_speech_ids_sends_text_only():
    body = harness.Target(text="hi").request.split(b"\r\n\r\n", 1)[1]
    assert json.loads(body) == {"text": "hi"}
