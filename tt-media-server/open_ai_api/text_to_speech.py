# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import os
from typing import Optional

from config.constants import AUDIO_RESPONSE_FORMATS, ModelRunners
from config.settings import settings
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


_VOICE_NOT_FOUND_MARKER = "not found. Available voices:"
_TEXT_TOO_LONG_MARKER = "exceeding the fixed ISL="

# Inworld TTS-2 only: a known, pre-existing bug (unrelated to any one
# session's changes -- confirmed via git-stash revert-and-reproduce, see
# models/demos/inworld_tts/FINAL_RUN.md) permanently corrupts a worker's
# subsequent NON-streaming output to near-silence once that worker has
# served any streaming request. Until that's root-caused, reject
# non-streaming requests outright instead of silently returning corrupted
# audio. Gated by an env var (default: disabled) so it's a one-line revert
# once the underlying bug is fixed -- not a hardcoded permanent behavior
# change.
_INWORLD_NON_STREAMING_DISABLED_MSG = (
    "Non-streaming text-to-speech requests are disabled on this deployment: a known bug "
    "permanently corrupts a worker's subsequent non-streaming output once it has served "
    "any streaming request. Set \"stream\": true and use the NDJSON streaming response instead."
)


def _non_streaming_disabled_for_inworld_tts() -> bool:
    return settings.model_runner == ModelRunners.TT_INWORLD_TTS.value and os.getenv(
        "INWORLD_TTS_DISABLE_NON_STREAMING", "1"
    ) != "0"


async def resolve_voice(tts_request, service) -> None:
    """Resolve the OpenAI SDK-compatible ``voice`` field against the
    registered voice list, in favor of ``voice_id`` -- does not alter
    ``voice_id``'s own matching/error behavior in any way.

    - ``voice_id`` present: wins outright, ``voice`` is ignored entirely
      (Console always sends a default ``voice`` like "alloy" alongside a
      real ``voice_id``; the explicit ID must take precedence).
    - Only ``voice`` present: matched case-insensitively against registered
      voice_ids. A match sets ``tts_request.voice_id`` so the rest of the
      pipeline (which only ever looks at ``voice_id``) needs no changes.
    - ``voice`` present but unmatched (e.g. an OpenAI default like "alloy"):
      silently ignored -- falls through to the default (TVD) voice, never
      an error, so existing OpenAI-SDK-shaped callers never break.
    """
    if tts_request.voice_id:
        return
    voice = getattr(tts_request, "voice", None)
    if not voice:
        return
    try:
        voice_list_result = await service.process_request(VoiceListRequest())
        registered = voice_list_result.to_dict().get("voices", [])
    except Exception:
        return  # Be permissive: listing failure just means no match found.
    match = next(
        (v["voice_id"] for v in registered if v.get("voice_id", "").lower() == voice.lower()),
        None,
    )
    if match:
        tts_request.voice_id = match
    # else: no match -- leave voice_id unset, default voice, no error.


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
            if _non_streaming_disabled_for_inworld_tts():
                raise HTTPException(status_code=400, detail=_INWORLD_NON_STREAMING_DISABLED_MSG)
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
        # An unknown voice_id is a client input error, not a server failure --
        # give it a 400 instead of a 500. The runner's own voice_id matching/
        # error message is unchanged; this only translates the status code by
        # recognizing that message's marker text, since the underlying
        # exception type doesn't survive the worker-process IPC boundary.
        # (Non-streaming path only: a streaming request's voice_id is only
        # looked up once generation is underway, inside an already-returned
        # StreamingResponse, where the status code can no longer change.)
        if _VOICE_NOT_FOUND_MARKER in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        # Input text too long for the fixed prefill length (tokenizer-exact
        # check, raised host-side in tt_modeling._prefill_with_perf before any
        # device call -- already fast, just needs the right status code): a
        # client input error, not a server failure.
        if _TEXT_TOO_LONG_MARKER in str(e):
            raise HTTPException(status_code=422, detail=str(e))
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
    await resolve_voice(tts_request, service)
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
