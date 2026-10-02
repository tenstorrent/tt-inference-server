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

from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import TextToSpeechResponse
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
            n = int(
                (i + 1) * 0.5 * vr.SAMPLE_RATE
            )  # distinct lengths so order is checkable
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
    out = r.run(
        [TextToSpeechRequest(text="Hello from Tenstorrent.", voice="fr_female", seed=3)]
    )
    assert len(out) == 1
    res = out[0]
    rate, ch, width, frames = _wav_info(res.audio)
    assert (rate, ch, width) == (24000, 1, 2)
    assert frames == 24000 and abs(res.duration - 1.0) < 1e-6
    assert (
        res.sample_rate == 24000
        and res.format == "wav"
        and res.speaker_id == "fr_female"
    )
    assert r.pipeline.calls == [("Hello from Tenstorrent.", "fr_female", 3)]


def test_voice_falls_back_to_speaker_id_then_default_and_seed_defaults_to_zero():
    r = _runner()
    r.run([TextToSpeechRequest(text="a", speaker_id="casual_male")])
    r.run([TextToSpeechRequest(text="b")])
    assert r.pipeline.calls == [("a", "casual_male", 0), ("b", vr.DEFAULT_VOICE, 0)]


def test_unknown_voice_is_rejected_before_synthesis():
    r = _runner()
    (out,) = r.run([TextToSpeechRequest(text="x", voice="nope")])
    assert isinstance(out, ValueError) and "unknown voice" in str(out)
    assert r.pipeline.calls == []


def test_one_bad_voice_fails_alone_in_a_batch():
    """Audit B1: an unknown voice must not fail the co-batched requests (nor raise, which would count toward a
    worker restart); its slot holds the error, which the service re-raises for that client only."""
    r = _runner(batch=8, pipeline=FakeBatched())
    reqs = [
        TextToSpeechRequest(text=f"t{i}", voice=v)
        for i, v in enumerate(["neutral_male", "nope", "fr_female"])
    ]
    out = r.run(reqs)
    assert isinstance(out[1], ValueError) and "unknown voice" in str(out[1])
    assert all(isinstance(out[i], TextToSpeechResponse) for i in (0, 2))
    assert [j[0] for j in r.pipeline.calls[0][1]] == [
        "t0",
        "t2",
    ]  # the bad request never reached the model


class FlakyBatched(FakeBatched):
    """synthesize_batch raises if any job's text is 'boom' (a model-side error such as 'no room')."""

    def synthesize_batch(self, jobs, **kw):
        if any(t == "boom" for t, _, _ in jobs):
            raise RuntimeError("a 2000-token prompt leaves no room for audio")
        return super().synthesize_batch(jobs, **kw)

    def synthesize(
        self, text, voice, seed=0, **kw
    ):  # the real pipeline's synthesize goes through synthesize_batch
        if text == "boom":
            raise RuntimeError("a 2000-token prompt leaves no room for audio")
        return super().synthesize(text, voice, seed=seed, **kw)


def test_model_error_is_isolated_to_its_request():
    r = _runner(batch=8, pipeline=FlakyBatched())
    out = r.run(
        [TextToSpeechRequest(text=t, voice="neutral_male") for t in ("a", "boom", "c")]
    )
    assert isinstance(out[1], RuntimeError) and "no room" in str(out[1])
    assert isinstance(out[0], TextToSpeechResponse) and isinstance(
        out[2], TextToSpeechResponse
    )


class DeadBatched(FakeBatched):
    """Every call fails with an error that is not the request's fault (a device fault, say)."""

    def synthesize_batch(self, jobs, **kw):
        raise RuntimeError("TT_THROW: device timeout")

    def synthesize(self, text, voice, seed=0, **kw):
        raise RuntimeError("TT_THROW: device timeout")


def test_every_request_failing_still_raises():
    """If nothing in the batch can be synthesized and the errors are not the requests' own, raise (likely the
    device): that must still count toward a worker restart."""
    r = _runner(batch=8, pipeline=DeadBatched())
    with pytest.raises(RuntimeError, match="device timeout"):
        r.run([TextToSpeechRequest(text=t, voice="neutral_male") for t in "abc"])


def test_a_lone_device_error_still_raises():
    r = _runner(batch=8, pipeline=DeadBatched())
    with pytest.raises(RuntimeError, match="device timeout"):
        r.run([TextToSpeechRequest(text="a", voice="neutral_male")])


def test_a_lone_request_error_fails_its_slot_without_raising():
    """Audit r2: at light load a request runs alone; a content error (no room, [END_AUDIO] on the first frame)
    must not raise, or each such request adds to the worker's error count and six of them restart the worker."""
    r = _runner(batch=8, pipeline=FlakyBatched())
    (out,) = r.run([TextToSpeechRequest(text="boom", voice="neutral_male")])
    assert isinstance(out, RuntimeError) and "no room" in str(out)


def test_a_batch_of_request_errors_fails_each_slot_without_raising():
    r = _runner(batch=8, pipeline=FlakyBatched())
    out = r.run(
        [TextToSpeechRequest(text="boom", voice="neutral_male") for _ in range(3)]
    )
    assert all(isinstance(o, RuntimeError) and "no room" in str(o) for o in out)


@pytest.mark.parametrize(
    "err, expect",
    [
        (
            ValueError(
                "a 2100-token prompt leaves no room for audio in max_seq_len=2048"
            ),
            True,
        ),
        (
            RuntimeError(
                "row 3 emitted [END_AUDIO] on the first frame -- nothing to decode"
            ),
            True,
        ),
        (RuntimeError("TT_THROW: device timeout"), False),
    ],
)
def test_request_errors_are_told_apart_from_device_errors(err, expect):
    assert vr.is_request_error(err) is expect


def test_long_text_is_chunked_and_concatenated():
    r = _runner()
    text = " ".join(f"Sentence number {i} is here." for i in range(60))  # ~1700 chars
    out = r.run([TextToSpeechRequest(text=text)])
    assert len(r.pipeline.calls) > 1, "long text should be split into sentence chunks"
    assert all(len(c[0]) <= vr.CHUNK_CHARS for c in r.pipeline.calls)
    assert abs(out[0].duration - len(r.pipeline.calls) * 1.0) < 1e-6


def test_batched_pipeline_keeps_request_order():
    r = _runner(batch=8, pipeline=FakeBatched())
    reqs = [
        TextToSpeechRequest(text=f"t{i}", voice="neutral_male", seed=i)
        for i in range(3)
    ]
    out = r.run(reqs)
    assert [o.duration for o in out] == pytest.approx([0.5, 1.0, 1.5])
    assert r.pipeline.calls[0][0] == "batch" and [
        j[0] for j in r.pipeline.calls[0][1]
    ] == ["t0", "t1", "t2"]


def test_request_model_accepts_voice_language_seed():
    req = TextToSpeechRequest(text="hi", voice="neutral_male", language="en", seed="7")
    assert req.voice == "neutral_male" and req.language == "en" and req.seed == 7
    with pytest.raises(ValueError):
        TextToSpeechRequest(text="hi", seed=-1)


def test_device_params_come_from_the_model_module():
    fake_pm = types.SimpleNamespace(
        L1_SMALL_SIZE=131072, TRACE_REGION_SIZE=250 * 1024 * 1024
    )
    r = _runner()
    with patch.object(
        vr.TTVoxtralTTSRunner, "_pipeline_module", staticmethod(lambda: fake_pm)
    ):
        assert r.get_pipeline_device_params() == {
            "l1_small_size": 131072,
            "trace_region_size": 250 * 1024 * 1024,
        }


def test_voxtral_runs_unthrottled_even_with_the_images_baked_throttle():
    """The image bakes ENV TT_MM_THROTTLE_PERF=5 and setup_runner_environment only sets the variable when the
    level is truthy, so the exemption must be "0" (tt-metal: no throttling), not None."""
    import sys
    from types import SimpleNamespace

    from config.constants import ModelRunners

    # Other test modules (test_device_worker.py, test_scheduler.py, ...) put a Mock in sys.modules["config.settings"]
    # and never restore it; in a full `pytest tests/` run Settings would then be a Mock and this test would check
    # nothing (it failed: '5' == '0'). Import the real module for this test only.
    with patch.dict(sys.modules):
        sys.modules.pop("config.settings", None)
        from config.settings import Settings

    s = SimpleNamespace(
        model_runner=ModelRunners.TT_VOXTRAL_TTS.value, default_throttle_level="5"
    )
    Settings._set_throttling_overrides(s)
    assert s.default_throttle_level == "0"
    assert (
        s.default_throttle_level
    )  # truthy, so runner_utils writes it over the baked 5

    other = SimpleNamespace(
        model_runner=ModelRunners.TT_WAN_2_2.value, default_throttle_level="5"
    )
    Settings._set_throttling_overrides(other)
    assert (
        other.default_throttle_level is None
    )  # unchanged behaviour for the other exempted runners


@pytest.mark.parametrize("seed", [2**63, 2**64, 2**70, True, 1.7, -1])
def test_out_of_range_or_non_integer_seed_is_rejected_at_validation(seed):
    """torch.Generator().manual_seed overflows at 2**64 inside the batch (fails every co-batched request);
    bool and 1.7 used to become 1 silently."""
    with pytest.raises(ValueError):
        TextToSpeechRequest(text="Hello.", voice="neutral_male", seed=seed)


@pytest.mark.parametrize("seed", [0, 7, 2**63 - 1, "12", 3.0])
def test_valid_seeds_still_pass(seed):
    assert TextToSpeechRequest(text="Hello.", seed=seed).seed == int(seed)


class FakeCapped(FakeBatched):
    """Second request of each batch hits the frame cap (stopped_naturally False)."""

    def synthesize_batch(self, jobs, **kw):
        out = super().synthesize_batch(jobs, **kw)
        self.last_timings = {
            "decode_ms_per_frame": 30.0,
            "stopped_naturally": [k != 1 for k in range(len(jobs))],
        }
        return out


def test_chunks_cut_off_by_the_frame_cap_are_logged():
    r = _runner(batch=4, pipeline=FakeCapped())
    r.logger = MagicMock()
    reqs = [
        TextToSpeechRequest(text=f"Sentence {i}.", voice="neutral_male", seed=i)
        for i in range(3)
    ]
    out = r.run(reqs)
    assert len(out) == 3
    warned = [c.args[0] for c in r.logger.warning.call_args_list]
    assert (
        len(warned) == 1
        and "1 chunk(s) hit the frame cap" in warned[0]
        and "requests [1]" in warned[0]
    )


def test_no_warning_when_every_chunk_stops_naturally():
    r = _runner(batch=4, pipeline=FakeBatched())
    r.logger = MagicMock()
    r.run(
        [
            TextToSpeechRequest(text="A.", voice="neutral_male"),
            TextToSpeechRequest(text="B.", voice="neutral_male"),
        ]
    )
    r.logger.warning.assert_not_called()
