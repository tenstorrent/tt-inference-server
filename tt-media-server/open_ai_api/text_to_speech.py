# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import APIRouter, Depends, HTTPException, Response, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


TTS_BINARY_FORMATS = ("audio", "wav", "mp3", "ogg")
TTS_MEDIA_TYPES = {
    "audio": "audio/wav",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}


async def handle_tts_request(tts_request, service):
    """
    Runner returns base64; post_process converts to requested format.
    Here we return result.output_bytes (WAV/MP3/OGG) or JSON with base64.
    """
    try:
        result = await service.process_request(tts_request)
        fmt = tts_request.response_format.lower()
        if fmt in TTS_BINARY_FORMATS:
            content = getattr(result, "output_bytes", None)
            if not content:
                raise HTTPException(
                    status_code=500,
                    detail="Binary audio not available in response",
                )
            return Response(
                content=content,
                media_type=TTS_MEDIA_TYPES.get(fmt, "audio/wav"),
            )
        return get_dict_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speech")
async def text_to_speech(
    tts_request: TextToSpeechRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Convert text to speech using the provided request.

    response_format controls the response type:
        - "audio" or "wav": binary WAV (Content-Type: audio/wav).
        - "mp3": binary MP3 (Content-Type: audio/mpeg); requires ffmpeg on the server.
        - "ogg": binary OGG (Content-Type: audio/ogg); requires ffmpeg on the server.
        - "json" or "verbose_json": JSON body with base64-encoded audio and metadata.

    Returns:
        FastAPI Response: either binary audio bytes (for audio/wav/mp3/ogg) or
        JSON with keys such as audio, duration, sample_rate, format (for json/verbose_json).

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
