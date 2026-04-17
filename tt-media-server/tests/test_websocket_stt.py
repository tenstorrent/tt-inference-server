# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDisconnect

from open_ai_api.websocket_stt import router
from resolver.service_resolver import service_resolver


@dataclass
class _MockResult:
    text: str


def _make_pcm16_bytes(duration_s: float = 0.5) -> bytes:
    """Generate silence PCM16 mono at 16 kHz for the given duration."""
    return b"\x00" * int(duration_s * 16000 * 2)


@pytest.fixture
def mock_service():
    svc = Mock()
    svc.process_request = AsyncMock(return_value=_MockResult(text="hello world"))
    return svc


@pytest.fixture
def stt_app(mock_service):
    app = FastAPI()
    app.include_router(router)  # no prefix — matches production mounting
    app.dependency_overrides[service_resolver] = lambda: mock_service
    return app


@pytest.fixture(autouse=True)
def reset_session_lock(monkeypatch):
    """Replace module-level SESSION_LOCK with a fresh lock before each test."""
    import open_ai_api.websocket_stt as stt_module

    monkeypatch.setattr(stt_module, "SESSION_LOCK", asyncio.Lock())


class TestWebSocketSTTUnit:
    """Tier 1: unit tests using mocked service — no live server required."""

    def test_connection_accepted(self, stt_app):
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.close()

    def test_transcribe_action_returns_interim_transcript(self, stt_app):
        pcm = _make_pcm16_bytes(0.5)
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.send_bytes(pcm)
                ws.send_text(json.dumps({"action": "transcribe"}))
                msg = ws.receive_json()

        assert msg["type"] == "transcript"
        assert msg["text"] == "hello world"
        assert msg["is_final"] is False
        assert isinstance(msg["time_ms"], (int, float))
        assert abs(msg["duration"] - 0.5) < 0.01

    def test_final_action_returns_final_transcript(self, stt_app):
        pcm = _make_pcm16_bytes(0.5)
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.send_bytes(pcm)
                ws.send_text(json.dumps({"action": "final"}))
                msg = ws.receive_json()

        assert msg["type"] == "transcript"
        assert msg["text"] == "hello world"
        assert msg["is_final"] is True

    def test_stop_action_sends_status_and_closes(self, stt_app):
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.send_text(json.dumps({"action": "stop"}))
                msg = ws.receive_json()

        assert msg["type"] == "status"
        assert msg["message"] == "stopped"

    def test_short_audio_skips_inference(self, stt_app, mock_service):
        pcm = b"\x00" * 100  # ~0.003s — well below MIN_DURATION_S = 0.3
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.send_bytes(pcm)
                ws.send_text(json.dumps({"action": "transcribe"}))
                msg = ws.receive_json()

        assert msg["type"] == "transcript"
        assert msg["text"] == ""
        mock_service.process_request.assert_not_called()

    def test_invalid_json_returns_error(self, stt_app):
        with TestClient(stt_app) as client:
            with client.websocket_connect("/ws/stt-live") as ws:
                ws.send_text("not valid json{{{")
                msg = ws.receive_json()
                ws.close()

        assert msg["type"] == "error"
        assert "Invalid JSON" in msg["message"]

    def test_busy_server_rejects_with_1013(self, stt_app, monkeypatch):
        import open_ai_api.websocket_stt as stt_module

        async def _always_timeout():
            raise asyncio.TimeoutError()

        monkeypatch.setattr(stt_module.SESSION_LOCK, "acquire", _always_timeout)

        with TestClient(stt_app) as client:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                with client.websocket_connect("/ws/stt-live") as ws:
                    ws.receive_text()  # triggers receipt of the close(1013) frame

        assert exc_info.value.code == 1013


@pytest.mark.live_server
class TestWebSocketSTTLive:
    """Tier 2: integration tests that require a live inference server with Whisper loaded."""

    LIVE_SERVER_URL = "ws://localhost:8000"

    @pytest.fixture
    def librispeech_audio_pcm(self):
        wav_path = Path(__file__).parent / "fixtures" / "librispeech_6s.wav"
        if not wav_path.exists():
            pytest.skip(
                f"Fixture missing: {wav_path}. Run scripts/generate_stt_fixture.py first."
            )
        with wave.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1, "Expected mono"
            assert wf.getsampwidth() == 2, "Expected PCM16"
            assert wf.getframerate() == 16000, "Expected 16 kHz"
            return wf.readframes(wf.getnframes())

    @pytest.fixture
    def ground_truth_keywords(self):
        txt_path = Path(__file__).parent / "fixtures" / "librispeech_6s_transcript.txt"
        if not txt_path.exists():
            return []
        text = txt_path.read_text().lower()
        return [w for w in text.split() if len(w) > 4]

    def test_live_streaming_transcription(
        self, librispeech_audio_pcm, ground_truth_keywords
    ):
        import websockets.sync.client as ws_sync

        CHUNK_BYTES = 16000 * 2 * 2  # 2s × 16kHz × 2 bytes/sample
        transcript_parts = []

        with ws_sync.connect(f"{self.LIVE_SERVER_URL}/ws/stt-live") as ws:
            for offset in range(
                0, len(librispeech_audio_pcm) - CHUNK_BYTES, CHUNK_BYTES
            ):
                chunk = librispeech_audio_pcm[
                    : offset + CHUNK_BYTES
                ]  # cumulative buffer
                ws.send(chunk)
                ws.send(json.dumps({"action": "transcribe"}))
                raw = ws.recv()
                msg = json.loads(raw)
                if msg.get("type") == "transcript":
                    transcript_parts.append(msg["text"])

            ws.send(librispeech_audio_pcm)
            ws.send(json.dumps({"action": "final"}))
            final_msg = json.loads(ws.recv())

        assert final_msg["type"] == "transcript"
        assert final_msg["is_final"] is True
        assert len(final_msg["text"]) > 0, "Final transcript must not be empty"

        if ground_truth_keywords:
            combined = " ".join(transcript_parts + [final_msg["text"]]).lower()
            matched = sum(1 for kw in ground_truth_keywords if kw in combined)
            match_ratio = matched / len(ground_truth_keywords)
            assert match_ratio >= 0.3, (
                f"Only {matched}/{len(ground_truth_keywords)} keywords matched "
                f"({match_ratio:.0%}). Got: {final_msg['text']!r}"
            )
