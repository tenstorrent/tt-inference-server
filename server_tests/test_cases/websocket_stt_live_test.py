# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import logging
import wave
from pathlib import Path

import aiohttp

from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

_FIXTURE_WAV = Path(__file__).parent.parent / "datasets" / "librispeech_6s.wav"
_FIXTURE_TXT = _FIXTURE_WAV.parent / "librispeech_6s_transcript.txt"

# 2-second chunks: 2s × 16 kHz × 2 bytes/sample
_CHUNK_BYTES = 16000 * 2 * 2

# Minimum fraction of ground-truth keywords that must appear in the transcript
_MIN_KEYWORD_MATCH = 0.3


def _load_pcm() -> bytes:
    with wave.open(str(_FIXTURE_WAV), "rb") as wf:
        assert wf.getnchannels() == 1, "Expected mono"
        assert wf.getsampwidth() == 2, "Expected PCM16"
        assert wf.getframerate() == 16000, "Expected 16 kHz"
        return wf.readframes(wf.getnframes())


def _load_keywords() -> list:
    if not _FIXTURE_TXT.exists():
        return []
    text = _FIXTURE_TXT.read_text().lower()
    return [w for w in text.split() if len(w) > 4]


class WebSocketSTTLiveTest(BaseTest):
    async def _run_specific_test_async(self):
        url = f"ws://localhost:{self.service_port}/ws/stt-live"
        pcm = _load_pcm()
        keywords = _load_keywords()

        logger.info(f"WebSocketSTTLiveTest: connecting to {url}")
        transcript_parts = []

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url) as ws:
                # Send audio in 2-second cumulative chunks with interim transcriptions
                for offset in range(0, len(pcm) - _CHUNK_BYTES, _CHUNK_BYTES):
                    chunk = pcm[: offset + _CHUNK_BYTES]
                    await ws.send_bytes(chunk)
                    await ws.send_str(json.dumps({"action": "transcribe"}))
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "transcript":
                            logger.info(f"  interim: {data.get('text', '')!r}")
                            transcript_parts.append(data.get("text", ""))

                # Final frame — send full buffer
                await ws.send_bytes(pcm)
                await ws.send_str(json.dumps({"action": "final"}))
                msg = await ws.receive()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    return {
                        "success": False,
                        "error": f"Expected TEXT, got {msg.type}",
                    }
                final_msg = json.loads(msg.data)

        logger.info(f"WebSocketSTTLiveTest: final={final_msg}")

        if final_msg.get("type") != "transcript":
            return {"success": False, "error": f"Unexpected final message: {final_msg}"}

        if not final_msg.get("is_final"):
            return {"success": False, "error": "Final message missing is_final=true"}

        final_text = final_msg.get("text", "")
        if not final_text:
            return {"success": False, "error": "Final transcript is empty"}

        match_ratio = 0.0
        matched = 0
        if keywords:
            combined = " ".join(transcript_parts + [final_text]).lower()
            matched = sum(1 for kw in keywords if kw in combined)
            match_ratio = matched / len(keywords)
            logger.info(f"Keyword match: {matched}/{len(keywords)} ({match_ratio:.0%})")
            if match_ratio < _MIN_KEYWORD_MATCH:
                return {
                    "success": False,
                    "error": (
                        f"Only {matched}/{len(keywords)} keywords matched "
                        f"({match_ratio:.0%}). Got: {final_text!r}"
                    ),
                }

        return {
            "success": True,
            "final_text": final_text,
            "interim_count": len(transcript_parts),
            "keyword_match_ratio": round(match_ratio, 3),
            "time_ms": final_msg.get("time_ms"),
            "duration": final_msg.get("duration"),
        }
