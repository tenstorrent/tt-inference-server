# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""
POST /tts/v1/voice:stream -- Inworld-native streaming TTS contract (see
domain/inworld_voice_stream_request.py for the exact schema this mirrors,
and what's supported vs rejected). A drop-in for code written against
Inworld's real API; contrast with /v1/audio/speech's OpenAI-compatible
raw-bytes contract (open_ai_api/text_to_speech.py), which is the same
underlying synthesis wrapped completely differently on the wire.
"""

import base64
import json

from domain.inworld_voice_stream_request import AudioConfig, InworldVoiceStreamRequest
from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from utils.ffmpeg_utils import encode_wav_bytes

router = APIRouter()

_VOICE_NOT_FOUND_MARKER = "not found. Available voices:"
_TEXT_TOO_LONG_MARKER = "exceeding the fixed ISL="

# DECODER_SAMPLE_RATE in tt_model_runners/inworld_tts_runner.py -- the native
# rate every chunk arrives at before any requested resample.
_NATIVE_SAMPLE_RATE = 48000

_ENCODING_TO_INTERNAL_FORMAT = {
    "WAV": "wav",
    "LINEAR16": "pcm",
    "PCM": "pcm",
    "MP3": "mp3",
    "OGG_OPUS": "ogg",
}


def _reencode_chunk(wav_b64: str, audio_config: AudioConfig) -> bytes:
    """Re-encodes ONE independent per-chunk WAV (already a complete, valid
    small audio file) into the requested audioConfig. Each streamed message
    is independently decodable, so a one-shot per-chunk convert is correct
    and simplest here -- contrast with /v1/audio/speech, which needs ONE
    continuous byte stream across ALL chunks and uses
    utils/streaming_audio_encoder's live-ffmpeg pipe instead.
    """
    raw_wav = base64.b64decode(wav_b64)
    fmt = _ENCODING_TO_INTERNAL_FORMAT[audio_config.audioEncoding]
    sample_rate = (
        audio_config.sampleRateHertz if audio_config.sampleRateHertz != _NATIVE_SAMPLE_RATE else None
    )
    bit_rate = audio_config.bitRate if fmt == "mp3" else None
    return encode_wav_bytes(raw_wav, fmt, sample_rate=sample_rate, bit_rate=bit_rate)


def _error_response_body(message: str, code: int) -> dict:
    return {"error": {"code": code, "message": message, "details": []}}


@router.post("/voice:stream")
async def voice_stream(
    req: InworldVoiceStreamRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    internal_request = TextToSpeechRequest(
        text=req.text,
        voice_id=req.voiceId,
        stream=True,
        temperature=req.temperature,
    )

    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail=_error_response_body("Model is not ready", 405))

    stream_iter = service.process_streaming_request(internal_request)
    try:
        # Prime the first chunk before creating the StreamingResponse --
        # Starlette commits the response status as soon as StreamingResponse
        # starts, before the generator yields anything, so an exception on
        # the first chunk (unknown voiceId, input too long) would otherwise
        # silently surface as 200 with an empty body. See the identical
        # pattern (and its rationale) in open_ai_api/text_to_speech.py.
        first_chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except Exception as e:
        detail = str(e)
        if _VOICE_NOT_FOUND_MARKER in detail:
            raise HTTPException(status_code=400, detail=_error_response_body(detail, 400))
        if _TEXT_TOO_LONG_MARKER in detail:
            raise HTTPException(status_code=422, detail=_error_response_body(detail, 422))
        raise HTTPException(status_code=500, detail=_error_response_body(detail, 500))

    async def result_stream():
        chars_sent = len(req.text)

        async def chunks():
            if first_chunk is not None:
                yield first_chunk
            async for c in stream_iter:
                yield c

        async for chunk in chunks():
            audio_bytes = _reencode_chunk(chunk.audio_base64, req.audioConfig)
            yield (
                json.dumps(
                    {
                        "result": {
                            "audioContent": base64.b64encode(audio_bytes).decode("ascii"),
                            "usage": {
                                "processedCharactersCount": chars_sent,
                                "modelId": req.modelId,
                            },
                        }
                    }
                )
                + "\n"
            )

    return StreamingResponse(result_stream(), media_type="application/json")
