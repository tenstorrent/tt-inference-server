# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


import time

from config.constants import AUDIO_RESPONSE_FORMATS
from config.settings import settings
from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import APIRouter, Depends, HTTPException, Response, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from telemetry.audio_metrics import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    char_count,
    record_tts_request,
    tts_voice_label,
)

router = APIRouter()

TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}


async def handle_tts_request(tts_request, service):
    """
    Runner returns base64; post_process converts to requested format.
    Here we return result.output_bytes (WAV/MP3/OGG) or JSON with base64.
    """
    start = time.perf_counter()
    fmt = tts_request.response_format.lower()
    status = STATUS_SUCCESS
    result = None
    try:
        result = await service.process_request(tts_request)
        if fmt in AUDIO_RESPONSE_FORMATS:
            content = getattr(result, "output_bytes", None)
            if not content:
                raise HTTPException(
                    status_code=500,
                    detail="Binary audio not available in response",
                )
            media_type = TTS_MEDIA_TYPES.get(result.format, "audio/wav")
            suggested_name = f"speech.{result.format}"
            headers = {"Content-Disposition": f"attachment; filename={suggested_name}"}
            return Response(
                content=content,
                media_type=media_type,
                headers=headers,
            )
        return get_dict_response(result)
    except HTTPException:
        status = STATUS_ERROR
        raise
    except Exception as e:
        status = STATUS_ERROR
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        record_tts_request(
            model_type=settings.model_runner,
            # The REQUESTED format, deliberately: the label tracks the client
            # mix, and json/verbose_json must stay distinguishable — the
            # delivered result.format is "wav" for every JSON response, so
            # labelling with it would collapse the mix. The mp3/ogg→wav
            # fallback is visible in the post_process warning logs instead.
            response_format=fmt,
            status=status,
            duration_seconds=time.perf_counter() - start,
            voice=tts_voice_label(tts_request, result),
            characters=char_count(getattr(tts_request, "text", None)),
            audio_seconds=getattr(result, "duration", None),
        )


@router.post("/speech")
async def text_to_speech(
    tts_request: TextToSpeechRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Convert text to speech using the provided request.

    response_format controls the response type:
        - "wav": binary WAV (Content-Type: audio/wav).
        - "mp3": binary MP3 (Content-Type: audio/mpeg); requires ffmpeg on the server.
        - "ogg": binary OGG (Content-Type: audio/ogg); requires ffmpeg on the server.
        - "json" or "verbose_json": JSON body with base64-encoded audio and metadata.

    Returns:
        FastAPI Response: either binary audio bytes (for wav/mp3/ogg) or
        JSON with keys such as audio, duration, sample_rate, format (for json/verbose_json). Default WAV.

    Raises:
        HTTPException: If text-to-speech fails or binary format requested but
        output not available (e.g. ffmpeg missing for mp3/ogg).
    """
    return await handle_tts_request(tts_request, service)


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()
