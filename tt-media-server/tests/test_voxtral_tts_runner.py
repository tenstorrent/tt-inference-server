# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for the Voxtral TTS runner: no device, no weights, no torch (the shared conftest
stubs torch and ttnn). The pipeline is faked with numpy waveforms; the tests check request
resolution (voice, seed), the 24 kHz WAV response, long-text chunking, and the batched path
keeping request order."""

import base64
import io
import sys
import types
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if "ttnn" not in sys.modules:  # the base metal runner imports ttnn at module scope
    sys.modules["ttnn"] = MagicMock()

from domain.text_to_speech_request import TextToSpeechRequest  # noqa: E402
from tt_model_runners import voxtral_tts_runner as vr  # noqa: E402


class FakePipeline:
    def __init__(self, seconds=1.0):
        self.seconds = seconds
        self.calls = []
        self.last_timings = {"decode_ms_per_frame": 26.0}
        self.model_dir = "/fake"
        self.warmed = {"traced": True}

    def synthesize(self, text, voice, seed=0, **kw):
        self.calls.append((text, voice, seed))
        n = int(self.seconds * vr.SAMPLE_RATE)
        return np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(1, 1, n)


class FakeBatched(FakePipeline):
    def synthesize_batch(self, jobs, **kw):
        self.calls.append(("batch", tuple(jobs)))
        out = []
        for i, (text, voice, seed) in enumerate(jobs):
            n = int((i + 1) * 0.5 * vr.SAMPLE_RATE)  # distinct lengths so order is checkable
            out.append(np.full((1, 1, n), 0.1 * (i + 1), dtype=np.float32))
        return out


def _runner(batch=1, pipeline=None):
    mock_settings = MagicMock()
    mock_settings.device_mesh_shape = (1, 1)
    mock_settings.use_dynamic_batcher = False
    mock_settings.max_batch_size = batch
    mock_settings.model_weights_path = ""
    with patch("tt_model_runners.base_device_runner.setup_runner_environment"), patch(
        "tt_model_runners.base_device_runner.get_settings", return_value=mock_settings
    ):
        r = vr.TTVoxtralTTSRunner("0")
    r.pipeline = pipeline or FakePipeline()
    r.voices = ["neutral_male", "fr_female", "casual_male"]
    return r


def _wav_info(b64):
    with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()


def test_single_request_is_24khz_wav_with_duration():
    r = _runner()
    out = r.run([TextToSpeechRequest(text="Hello from Tenstorrent.", voice="fr_female", seed=3)])
    assert len(out) == 1
    res = out[0]
    rate, ch, width, frames = _wav_info(res.audio)
    assert (rate, ch, width) == (24000, 1, 2)
    assert frames == 24000 and abs(res.duration - 1.0) < 1e-6
    assert res.sample_rate == 24000 and res.format == "wav" and res.speaker_id == "fr_female"
    assert r.pipeline.calls == [("Hello from Tenstorrent.", "fr_female", 3)]


def test_voice_falls_back_to_speaker_id_then_default_and_seed_defaults_to_zero():
    r = _runner()
    r.run([TextToSpeechRequest(text="a", speaker_id="casual_male")])
    r.run([TextToSpeechRequest(text="b")])
    assert r.pipeline.calls == [("a", "casual_male", 0), ("b", vr.DEFAULT_VOICE, 0)]


def test_unknown_voice_is_rejected_before_synthesis():
    r = _runner()
    with pytest.raises(ValueError, match="unknown voice"):
        r.run([TextToSpeechRequest(text="x", voice="nope")])
    assert r.pipeline.calls == []


def test_long_text_is_chunked_and_concatenated():
    r = _runner()
    text = " ".join(f"Sentence number {i} is here." for i in range(60))  # ~1700 chars
    out = r.run([TextToSpeechRequest(text=text)])
    assert len(r.pipeline.calls) > 1, "long text should be split into sentence chunks"
    assert all(len(c[0]) <= vr.CHUNK_CHARS for c in r.pipeline.calls)
    assert abs(out[0].duration - len(r.pipeline.calls) * 1.0) < 1e-6


def test_batched_pipeline_keeps_request_order():
    r = _runner(batch=8, pipeline=FakeBatched())
    reqs = [TextToSpeechRequest(text=f"t{i}", voice="neutral_male", seed=i) for i in range(3)]
    out = r.run(reqs)
    assert [o.duration for o in out] == pytest.approx([0.5, 1.0, 1.5])
    assert r.pipeline.calls[0][0] == "batch" and [j[0] for j in r.pipeline.calls[0][1]] == ["t0", "t1", "t2"]


def test_request_model_accepts_voice_language_seed():
    req = TextToSpeechRequest(text="hi", voice="neutral_male", language="en", seed="7")
    assert req.voice == "neutral_male" and req.language == "en" and req.seed == 7
    with pytest.raises(ValueError):
        TextToSpeechRequest(text="hi", seed=-1)


def test_device_params_come_from_the_model_module():
    fake_pm = types.SimpleNamespace(L1_SMALL_SIZE=131072, TRACE_REGION_SIZE=250 * 1024 * 1024)
    r = _runner()
    with patch.object(vr.TTVoxtralTTSRunner, "_pipeline_module", staticmethod(lambda: fake_pm)):
        assert r.get_pipeline_device_params() == {"l1_small_size": 131072, "trace_region_size": 250 * 1024 * 1024}
