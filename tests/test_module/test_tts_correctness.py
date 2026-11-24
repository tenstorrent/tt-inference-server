# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""tt-tts same-prompt correctness eval: grading, workflow args, and a run against
a fake streaming server (ASR stubbed)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from test_module._test_common import ReportCheckTypes
from test_module._test_common import TestConfig as _TestConfig
from test_module.eval_tests import tts_correctness_test as correctness
from test_module.eval_tests import tts_eval_tests
from test_module.eval_tests.tts_client import STREAM_WAV_HEADER_BYTES


class TestWordErrors:
    def test_counts_each_edit_kind(self):
        ref = "the cat sat on the mat".split()
        assert correctness.word_errors(ref, ref)["wer"] == 0.0
        sub = correctness.word_errors(ref, "the dog sat on the mat".split())
        assert (sub["sub"], sub["del"], sub["ins"]) == (1, 0, 0)
        dels = correctness.word_errors(ref, "the cat sat".split())
        assert (dels["del"], dels["wer"]) == (3, 0.5)
        ins = correctness.word_errors(ref, ref + ["again", "again"])
        assert ins["ins"] == 2

    def test_normalize_drops_case_and_punctuation(self):
        assert correctness.normalize("It's DONE, ok?") == ["its", "done", "ok"]


def _rec(tag, i, ok=True, wer=0.0, sha="a", audio_s=2.0):
    return {"tag": tag, "i": i, "ok": ok, "wer": wer if ok else None,
            "sha256": sha, "audio_s": audio_s, "error": None if ok else "http 500"}  # fmt: skip


class TestGrade:
    def test_any_bad_response_fails_it_and_is_counted(self):
        records = [
            _rec("ref", 0),
            _rec("ref", 1, ok=False),
            _rec("load", 0, sha="b", audio_s=4.0),
            _rec("load", 1, wer=0.2),
        ]
        summary = correctness.grade(records, max_wer=0.05)
        assert [r["pass"] for r in records] == [True, False, True, False]
        assert records[3]["reasons"] == ["wer 20.0% > 5.0%"]
        assert records[2]["duration_ratio"] == 2.0
        assert records[2]["pcm_matches_ref"] is False
        assert summary["reference_failures"] == 1
        assert (summary["load_requests"], summary["load_failures"]) == (2, 1)

    def test_missing_transcript_fails(self):
        records = [_rec("load", 0, wer=None)]
        correctness.grade(records, max_wer=0.05)
        assert records[0]["reasons"] == ["no transcript"]


def _ctx(workflow_args=""):
    return SimpleNamespace(runtime_config=SimpleNamespace(workflow_args=workflow_args))


class TestWorkflowArgOverrides:
    def test_only_eval_keys_apply(self):
        ctx = _ctx("concurrency=600 isl=128 eval_requests=20 eval_refs=0 eval_x=1")
        assert tts_eval_tests._correctness_overrides(ctx) == {
            "requests": 20,
            "refs": 0,
        }

    @pytest.mark.parametrize(
        "arg", ["eval_requests=0", "eval_concurrency=x", "eval_refs=-1"]
    )
    def test_invalid_values_are_rejected(self, arg):
        with pytest.raises(ValueError):
            tts_eval_tests._correctness_overrides(_ctx(arg))


def _task():
    return SimpleNamespace(
        task_name=tts_eval_tests.TASK_CORRECTNESS,
        model_kwargs={"tts_api": "tt_tts", "requests": 3},
        score=SimpleNamespace(
            tolerance=None, published_score=100.0, published_score_ref=""
        ),
    )


class TestCorrectnessTask:
    def _block(self, monkeypatch, result):
        monkeypatch.setattr(tts_eval_tests, "_missing_deps", lambda deps: [])
        monkeypatch.setattr(
            tts_eval_tests, "_run_tts_correctness_eval", lambda ctx, task: result
        )
        monkeypatch.setattr(tts_eval_tests, "block_id", lambda ctx: None)
        return tts_eval_tests._run_correctness_task(SimpleNamespace(), _task())

    def test_all_pass(self, monkeypatch):
        block = self._block(
            monkeypatch,
            {
                "success": True,
                "pass_rate": 100.0,
                "mean_wer": 0.01,
                "failed_responses": [],
            },  # fmt: skip
        )
        assert block.data["accuracy_check"] is ReportCheckTypes.PASS
        assert block.data["score"] == 100.0
        assert block.data["failed_responses"] is None

    def test_one_failure_fails(self, monkeypatch):
        block = self._block(
            monkeypatch,
            {
                "success": False,
                "pass_rate": 75.0,
                "load_failures": 1,
                "failed_responses": ["load_001"],
            },  # fmt: skip
        )
        assert block.data["accuracy_check"] is ReportCheckTypes.FAIL
        assert block.data["failed_responses"] == "load_001"

    def test_missing_deps_is_na(self, monkeypatch):
        monkeypatch.setattr(tts_eval_tests, "_missing_deps", lambda deps: ["torch"])
        monkeypatch.setattr(tts_eval_tests, "block_id", lambda ctx: None)
        block = tts_eval_tests._run_correctness_task(SimpleNamespace(), _task())
        assert block.data["accuracy_check"] is ReportCheckTypes.NA

    def test_registered_as_a_task_runner(self):
        runner = tts_eval_tests._TASK_RUNNERS[tts_eval_tests.TASK_CORRECTNESS]
        assert runner is tts_eval_tests._run_correctness_task


def _wav_header():
    return b"RIFF\xff\xff\xff\xffWAVEfmt " + bytes.fromhex(
        "10000000" "0100" "0100" "80bb0000" "00770100" "0200" "1000"
    ) + b"data\xff\xff\xff\xff"  # fmt: skip


async def _run_against_fake_server(tmp_path, targets, transcribe):
    payloads, served = [], [0]

    async def speech(request):
        payloads.append(await request.json())
        served[0] += 1
        response = web.StreamResponse()
        response.enable_chunked_encoding()
        await response.prepare(request)
        await response.write(_wav_header())
        if served[0] != 4:  # the 4th response is header-only
            await response.write(b"\1\0" * 4800)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        test = correctness.TTSCorrectnessTest(
            _TestConfig({}), {"outdir": str(tmp_path), **targets}
        )
        test.base_url = f"http://127.0.0.1:{port}"
        if transcribe is not None:
            test._load_asr = lambda model, workers, threads: transcribe
        return await test._run_specific_test_async(), payloads
    finally:
        await runner.cleanup()


def test_correctness_run_end_to_end(tmp_path):
    def transcribe(path):
        # The reference comes back garbled; everything else is read back verbatim.
        return "something else entirely" if path.endswith("ref_000.wav") else "hi there"

    result, payloads = asyncio.run(
        _run_against_fake_server(
            tmp_path,
            {
                "text": "Hi there.",
                "refs": 1,
                "requests": 4,
                "concurrency": 2,
                "speech_ids": [5, 6],
            },  # fmt: skip
            transcribe,
        )
    )
    assert len(payloads) == 5
    assert payloads[0] == {"text": "Hi there.", "speech_ids": [5, 6]}
    assert result["success"] is False
    assert result["reference_failures"] == 1  # garbled transcript
    assert result["load_failures"] == 1  # the header-only response
    assert result["pass_rate"] == pytest.approx(60.0)
    saved = json.loads(Path(result["results_file"]).read_text())
    assert len(saved["records"]) == 5
    assert len(list((tmp_path / "wav").glob("*.wav"))) == 5
    ref_wav = (tmp_path / "wav" / "ref_000.wav").read_bytes()
    assert len(ref_wav) == STREAM_WAV_HEADER_BYTES + 9600


def test_asr_failure_fails_the_run(tmp_path):
    def transcribe(path):
        raise RuntimeError("asr broke")

    with pytest.raises(RuntimeError, match="asr broke"):
        asyncio.run(
            _run_against_fake_server(
                tmp_path, {"text": "Hi.", "refs": 1, "requests": 2}, transcribe
            )
        )


@pytest.mark.parametrize("cpus,workers", [(1, 1), (7, 1), (8, 2), (16, 4), (96, 4)])
def test_default_asr_workers(cpus, workers):
    assert correctness.default_asr_workers(cpus) == workers


def test_cpu_budget_is_positive():
    assert correctness.cpu_budget() >= 1


@pytest.mark.parametrize("cpus,workers,threads", [(1, 1, 1), (2, 1, 1), (16, 4, 3)])
def test_asr_threads_leave_a_cpu_free(cpus, workers, threads):
    assert correctness.asr_threads(cpus, workers) == threads


def test_asr_load_failure_fails_the_run(tmp_path, monkeypatch):
    def broken_load(model, workers, threads):
        raise RuntimeError("no model")

    monkeypatch.setattr(
        correctness.TTSCorrectnessTest, "_load_asr", staticmethod(broken_load)
    )

    async def run():
        # The helper stubs _load_asr on the instance; undo that so the class
        # stub (a failing model load) is what runs.
        return await _run_against_fake_server(
            tmp_path, {"text": "Hi.", "refs": 1, "requests": 2}, None
        )

    with pytest.raises(RuntimeError, match="no model"):
        asyncio.run(run())
