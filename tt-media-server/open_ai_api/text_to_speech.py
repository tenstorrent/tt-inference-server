# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
from typing import Optional

from config.constants import AUDIO_RESPONSE_FORMATS
from domain.text_to_speech_request import TextToSpeechRequest
from domain.voice_encode_request import VoiceEncodeRequest
from domain.voice_list_request import VoiceListRequest
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    Security,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

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

    Supports streaming (Inworld TTS runner only, currently) based on the
    ``stream`` field on ``tts_request``, mirroring audio.py's
    ``handle_audio_request`` NDJSON pattern for the STT endpoints -- each
    line is one base64-WAV-carrying chunk (or the final metadata-only
    marker), letting a client start playback before the whole utterance
    finishes generating instead of waiting for one complete response.
    Streaming chunks are always WAV -- response_format's MP3/OGG
    post-processing only applies to the non-streaming path.
    """
    try:
        if not getattr(tts_request, "stream", False):
            result = await service.process_request(tts_request)
            fmt = tts_request.response_format.lower()
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

        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

        async def result_stream():
            async for partial in service.process_streaming_request(tts_request):
                yield json.dumps(get_dict_response(partial)) + "\n"

        return StreamingResponse(result_stream(), media_type="application/x-ndjson")
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


async def parse_voice_encode_request(
    request: Request,
    file: Optional[UploadFile] = File(None),
    voice_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
) -> VoiceEncodeRequest:
    content_type = request.headers.get("content-type", "").lower()

    if file is not None:
        file_content = await file.read()
        return VoiceEncodeRequest(
            reference_audio=file_content,
            voice_id=voice_id,
            language=language,
            description=description,
        )
    if "application/json" in content_type:
        json_body = await request.json()
        return VoiceEncodeRequest(**json_body)
    raise HTTPException(
        status_code=400,
        detail="Use either multipart/form-data with file upload or application/json",
    )


@router.post("/voices")
async def register_voice(
    voice_encode_request: VoiceEncodeRequest = Depends(parse_voice_encode_request),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Register a reference audio clip as a reusable voice-clone prompt.

    Accepts either multipart/form-data (file upload, optional voice_id form
    field) or application/json (reference_audio + optional voice_id).

    Returns:
        JSON body with the assigned/echoed voice_id and the number of VQ
        codes the reference audio was encoded into. Pass voice_id back on a
        subsequent POST /v1/audio/speech request to synthesize in that voice.

    Raises:
        HTTPException: If voice registration fails.
    """
    try:
        result = await service.process_request(voice_encode_request)
        return get_dict_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices")
async def list_voices(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    List all registered voice-clone voices with their metadata.

    Returns:
        JSON body with a ``voices`` array, one entry per cached voice, each
        with ``voice_id``, ``language``, ``description`` and ``num_codes``.
        Voices registered before language/description support naturally report
        ``language``/``description`` as null.

    Raises:
        HTTPException: If listing the voices fails.
    """
    try:
        result = await service.process_request(VoiceListRequest())
        return get_dict_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()
