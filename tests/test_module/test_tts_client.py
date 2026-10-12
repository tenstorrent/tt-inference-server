# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit coverage for the TTS eval request/response adapters (no server)."""

from __future__ import annotations

import asyncio
import struct

import aiohttp
import numpy as np
import pytest

from test_module.eval_tests import tts_client
from test_module.eval_tests.tts_client import (
    MIN_AUDIO_SECONDS,
    MediaServerTTSClient,
    Qwen3CppTTSClient,
    SynthesizedAudio,
    VoiceClip,
    decode_streamed_wav,
    make_tts_client,
    parse_pcm16_wav_header,
)


def _wav_header(sample_rate=24000, channels=1, data_size=0xFFFFFFFF, extra=b""):
    byte_rate = sample_rate * channels * 2
    riff_size = 0xFFFFFFFF if data_size == 0xFFFFFFFF else 36 + len(extra) + data_size
    fmt = struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, channels * 2, 16)
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + fmt
        + extra
        + b"data"
        + struct.pack("<I", data_size)
    )


def _pcm(values):
    return np.asarray(values, dtype="<i2").tobytes()


class TestStreamedWavDecode:
    def test_streaming_header_takes_everything_after_it(self):
        body = _wav_header() + _pcm([0, 16384, -16384, 32767])
        audio = decode_streamed_wav(body)
        assert audio.sample_rate == 24000
        np.testing.assert_allclose(audio.samples, [0.0, 0.5, -0.5, 32767 / 32768])

    def test_header_is_the_cpp_servers_44_bytes(self):
        assert parse_pcm16_wav_header(_wav_header())[2] == 44

    def test_finalized_header_and_extra_chunks(self):
        extra = b"LIST" + struct.pack("<I", 3) + b"abc\x00"  # odd size is padded
        body = _wav_header(16000, 1, 4, extra=extra) + _pcm([1, 2])
        sr, channels, offset = parse_pcm16_wav_header(body)
        assert (sr, channels) == (16000, 1)
        assert offset == len(body) - 4

    def test_stereo_is_downmixed_and_partial_frames_dropped(self):
        body = _wav_header(channels=2) + _pcm([16384, 0, -16384, -16384]) + b"\x01"
        audio = decode_streamed_wav(body)
        np.testing.assert_allclose(audio.samples, [0.25, -0.5])

    def test_header_only_is_zero_samples(self):
        assert decode_streamed_wav(_wav_header()).seconds == 0.0

    @pytest.mark.parametrize(
        "body",
        [b"", b"not a wav at all", b"RIFF\xff\xff\xff\xffWAVE"],
    )
    def test_garbage_is_rejected(self, body):
        with pytest.raises(ValueError):
            decode_streamed_wav(body)

    def test_non_pcm16_is_rejected(self):
        body = bytearray(_wav_header())
        body[34:36] = struct.pack("<H", 32)  # bits per sample
        with pytest.raises(ValueError, match="PCM16"):
            decode_streamed_wav(bytes(body))


class TestMakeClient:
    def test_default_is_the_python_media_server(self):
        client = make_tts_client({})
        assert isinstance(client, MediaServerTTSClient)
        assert client.can_clone and client.has_default_voice

    def test_qwen3_base_clones_and_has_no_default_voice(self):
        client = make_tts_client({"tts_api": "qwen3_cpp", "qwen3_release": "base"})
        assert isinstance(client, Qwen3CppTTSClient)
        assert client.can_clone and not client.has_default_voice

    def test_qwen3_custom_voice_needs_a_speaker(self):
        with pytest.raises(ValueError, match="speaker"):
            make_tts_client({"tts_api": "qwen3_cpp", "qwen3_release": "custom_voice"})
        client = make_tts_client(
            {"tts_api": "qwen3_cpp", "qwen3_release": "custom_voice", "speaker": "ryan"}
        )
        assert not client.can_clone and client.has_default_voice

    def test_unknown_api_and_release_are_rejected(self):
        with pytest.raises(ValueError, match="tts_api"):
            make_tts_client({"tts_api": "grpc"})
        with pytest.raises(ValueError, match="qwen3_release"):
            make_tts_client({"tts_api": "qwen3_cpp", "qwen3_release": "tiny"})


def _form_fields(form: aiohttp.FormData) -> dict:
    return {
        field[0]["name"]: (field[0].get("filename"), field[2]) for field in form._fields
    }


class TestQwen3Requests:
    CLIP = VoiceClip(b"RIFF....WAVE", transcript="what the clip says")

    def test_custom_voice_is_json_with_speaker_and_language(self):
        client = Qwen3CppTTSClient("custom_voice", speaker="ryan", language="English")
        assert client.build_request("hi", None) == {
            "json": {"text": "hi", "language": "English", "speaker": "ryan"}
        }

    def test_custom_voice_cannot_clone(self):
        client = Qwen3CppTTSClient("custom_voice", speaker="ryan")
        with pytest.raises(ValueError, match="cannot clone"):
            client.build_request("hi", self.CLIP)

    def test_base_cannot_speak_without_a_clip(self):
        with pytest.raises(ValueError, match="without a clip"):
            Qwen3CppTTSClient("base").build_request("hi", None)

    def test_voice_only_clone_is_multipart_without_reference_text(self):
        request = Qwen3CppTTSClient("base", language="English").build_request(
            "hi", self.CLIP
        )
        fields = _form_fields(request["data"])
        assert fields["text"] == (None, "hi")
        assert fields["language"] == (None, "English")
        assert fields["voice"] == ("reference.wav", self.CLIP.wav_bytes)
        assert "reference_text" not in fields
        assert "speaker" not in fields

    def test_in_context_clone_sends_the_transcript(self):
        client = Qwen3CppTTSClient("base", clone_mode="in_context")
        fields = _form_fields(client.build_request("hi", self.CLIP)["data"])
        assert fields["reference_text"] == (None, "what the clip says")

    def test_in_context_clone_needs_a_transcript(self):
        client = Qwen3CppTTSClient("base", clone_mode="in_context")
        with pytest.raises(ValueError, match="transcript"):
            client.build_request("hi", VoiceClip(b"RIFF"))


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def read(self):
        return self._body

    async def text(self):
        return self._body.decode("utf-8", "replace")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response, sink):
        self._response = response
        self._sink = sink

    def post(self, url, **kwargs):
        self._sink.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_server(monkeypatch):
    calls = []

    def install(status, body):
        monkeypatch.setattr(
            aiohttp,
            "ClientSession",
            lambda **kwargs: _FakeSession(_FakeResponse(status, body), calls),
        )
        return calls

    return install


class TestQwen3Synthesize:
    def _run(self, client, voice=None):
        return asyncio.run(
            client.synthesize("http://x/v1/audio/speech", "hi", seed=3, voice=voice)
        )

    def test_streamed_audio_is_decoded_and_seed_is_not_sent(self, fake_server):
        n = int(24000 * (MIN_AUDIO_SECONDS + 0.1))
        calls = fake_server(200, _wav_header() + _pcm(np.zeros(n)))
        audio = self._run(Qwen3CppTTSClient("custom_voice", speaker="ryan"))
        assert audio is not None and audio.sample_rate == 24000
        assert len(audio.samples) == n
        assert "seed" not in calls[0][1]["json"]

    def test_stream_that_ends_early_is_a_failed_sample(self, fake_server):
        fake_server(200, _wav_header() + _pcm(np.zeros(100)))
        assert self._run(Qwen3CppTTSClient("custom_voice", speaker="ryan")) is None

    def test_http_400_is_a_failed_sample(self, fake_server):
        fake_server(400, b'{"error": "speaker is required"}')
        assert self._run(Qwen3CppTTSClient("custom_voice", speaker="ryan")) is None


class _JsonResponse(_FakeResponse):
    def __init__(self, payload):
        super().__init__(200, b"")
        self._payload = payload

    async def json(self):
        return self._payload


class TestMediaServerSynthesize:
    def test_payload_keeps_the_python_server_contract(self, monkeypatch):
        import base64

        response = _JsonResponse({"audio": base64.b64encode(b"wav").decode()})
        calls = []
        monkeypatch.setattr(
            aiohttp, "ClientSession", lambda **kw: _FakeSession(response, calls)
        )
        monkeypatch.setattr(
            tts_client,
            "decode_wav_file",
            lambda b: SynthesizedAudio(np.zeros(1600, dtype=np.float32), 16000),
        )

        audio = asyncio.run(
            MediaServerTTSClient().synthesize(
                "http://x", "hi", seed=5, voice=VoiceClip(b"abc")
            )
        )

        assert audio.sample_rate == 16000 and len(audio.samples) == 1600
        assert calls[0][1]["json"] == {
            "text": "hi",
            "response_format": "json",
            "seed": 5,
            "reference_audio": base64.b64encode(b"abc").decode(),
        }
