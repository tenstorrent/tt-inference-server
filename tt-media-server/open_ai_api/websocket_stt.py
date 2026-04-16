# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import json
import logging
import struct
import time

from config.constants import ResponseFormat
from domain.audio_processing_request import AudioProcessingRequest
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from starlette.websockets import WebSocketState

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
MIN_DURATION_S = 0.3

SESSION_LOCK = asyncio.Lock()

router = APIRouter()
logger = logging.getLogger(__name__)


def _wrap_pcm16_as_wav(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
    n = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    header += b"data" + struct.pack("<I", n)
    return header + pcm


async def _transcribe(service: BaseService, pcm_bytes: bytes, is_final: bool) -> dict:
    wav = _wrap_pcm16_as_wav(pcm_bytes)
    req = AudioProcessingRequest(
        file=wav,
        stream=False,
        response_format=ResponseFormat.JSON.value,
        is_preprocessing_enabled=False,
        perform_diarization=False,
        return_timestamps=False,
    )
    t0 = time.perf_counter()
    result = await service.process_request(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = (getattr(result, "text", "") or "").strip()
    duration = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    return {
        "type": "transcript",
        "text": text,
        "is_final": bool(is_final),
        "time_ms": round(elapsed_ms, 2),
        "duration": round(duration, 3),
    }


@router.websocket("/ws/stt-live")
async def stt_live(
    websocket: WebSocket,
    service: BaseService = Depends(service_resolver),
):
    await websocket.accept()

    try:
        await asyncio.wait_for(SESSION_LOCK.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        logger.info("STT-live: rejecting second session (busy)")
        await websocket.close(code=1013, reason="Server busy")
        return

    try:
        logger.info("STT-live: session opened")
        audio_buffer = bytearray()
        session_active = True
        try:
            while session_active:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if message.get("bytes") is not None:
                    audio_buffer = bytearray(message["bytes"])
                    continue

                text = message.get("text")
                if text is None:
                    continue

                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid JSON"}
                    )
                    continue

                action = payload.get("action")
                duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)

                if action in ("transcribe", "final"):
                    if duration < MIN_DURATION_S:
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                "text": "",
                                "is_final": action == "final",
                                "time_ms": 0,
                                "duration": round(duration, 3),
                            }
                        )
                    else:
                        try:
                            response = await _transcribe(
                                service, bytes(audio_buffer), action == "final"
                            )
                            await websocket.send_json(response)
                        except Exception as e:
                            logger.exception("STT-live transcribe failed")
                            await websocket.send_json(
                                {"type": "error", "message": str(e)}
                            )
                    if action == "final":
                        session_active = False
                elif action == "stop":
                    await websocket.send_json({"type": "status", "message": "stopped"})
                    session_active = False
                else:
                    await websocket.send_json(
                        {"type": "error", "message": f"Unknown action: {action!r}"}
                    )
        except WebSocketDisconnect:
            logger.info("STT-live: client disconnected")
        finally:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                try:
                    await websocket.close()
                except Exception:
                    pass
            logger.info("STT-live: session closed")
    finally:
        SESSION_LOCK.release()
