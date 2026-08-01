# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import json
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
from fastapi.exception_handlers import (
    http_exception_handler as _default_http_exception_handler,
)
from fastapi.exception_handlers import (
    request_validation_exception_handler as _default_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from starlette.exceptions import HTTPException as StarletteHTTPException
from utils.streaming_audio_encoder import build_streaming_wav_header, encode_pcm_stream, parse_wav_chunk

router = APIRouter()

TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}

# Paths these error-envelope handlers apply to -- registered globally on the
# app (exception handlers can't be scoped to a router), but this server is
# shared across many model runners whose clients already expect FastAPI's
# default {"detail": ...} shape. Restricting by exact path keeps every other
# endpoint's error format unchanged; anything else falls through to
# FastAPI's own default handler.
_OPENAI_ENVELOPE_PATHS = {"/v1/audio/speech", "/audio/speech", "/v1/audio/voices", "/audio/voices"}
_INWORLD_ENVELOPE_PATHS = {"/tts/v1/voice:stream"}


def _openai_error_body(message: str, status_code: int) -> dict:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error" if status_code < 500 else "server_error",
            "code": str(status_code),
        }
    }


def _inworld_error_body(message: str, status_code: int) -> dict:
    return {"error": {"code": status_code, "message": message, "details": []}}


async def openai_style_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Rewraps 422s for the TTS endpoints into whichever error envelope that
    exact path's real API contract expects (OpenAI's {"error": {message,
    type, code}} vs Inworld's {"error": {code, message, details}}).
    FastAPI's default shape ({"detail": [...]}) is what every other model
    runner's clients already expect -- only rewrap for these paths. Only the
    first validation error's message is used (both real envelopes carry a
    single error, not a list).
    """
    errors = exc.errors()
    message = errors[0]["msg"] if errors else "Invalid request"
    if request.url.path in _OPENAI_ENVELOPE_PATHS:
        return JSONResponse(status_code=422, content=_openai_error_body(message, 422))
    if request.url.path in _INWORLD_ENVELOPE_PATHS:
        return JSONResponse(status_code=422, content=_inworld_error_body(message, 422))
    return await _default_validation_exception_handler(request, exc)


async def openai_style_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Same rewrap as above, for HTTPExceptions raised inside the handlers
    themselves (400/405/422/500) -- everything else keeps FastAPI's default
    {"detail": ...}. If a handler already built the Inworld-shaped body
    itself (exc.detail is a dict, not a string), pass it through unchanged
    rather than double-wrapping.
    """
    if request.url.path in _OPENAI_ENVELOPE_PATHS:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content=_openai_error_body(detail, exc.status_code))
    if request.url.path in _INWORLD_ENVELOPE_PATHS:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code, content=_inworld_error_body(str(exc.detail), exc.status_code)
        )
    return await _default_http_exception_handler(request, exc)


_VOICE_NOT_FOUND_MARKER = "not found. Available voices:"
_TEXT_TOO_LONG_MARKER = "exceeding the fixed ISL="
# Distinct from _TEXT_TOO_LONG_MARKER above (the hard PREFILL_ISL=1024
# ceiling): this is tt_modeling.MAX_TEXT_TOKENS=250, a tighter practical
# limit for reliable full-length synthesis -- see that constant's comment.
_TEXT_EXCEEDS_RECOMMENDED_LENGTH_MARKER = "exceeding the recommended maximum of"

# response_format values /v1/audio/speech's raw-bytes streaming path can
# actually produce for the inworld-tts runner, and their Content-Type.
_STREAMING_MEDIA_TYPE_BY_FORMAT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "pcm": "audio/pcm",
}


def _is_inworld_tts_runner() -> bool:
    return settings.model_runner == ModelRunners.TT_INWORLD_TTS.value


def _apply_inworld_response_format_default(tts_request) -> None:
    """Defaults response_format to "mp3" (OpenAI's real /v1/audio/speech
    default) for the inworld-tts runner specifically, when the client
    didn't set it -- every other runner keeps the shared
    TextToSpeechRequest.response_format field's "wav" default unchanged.
    """
    if not _is_inworld_tts_runner():
        return
    if "response_format" not in tts_request.model_fields_set:
        tts_request.response_format = "mp3"


async def _stream_openai_compatible_audio_bytes(tts_request, service):
    """POST /v1/audio/speech's OpenAI-compatible contract for the
    inworld-tts runner: ALWAYS uses the safe streaming-decoder pipeline
    internally (never the non-streaming decoder path, which has a known
    pre-existing bug that permanently corrupts a worker's subsequent
    non-streaming output once it has served a streaming request -- see
    FINAL_RUN.md), regardless of the client's "stream" value. Returns ONE
    continuous chunked HTTP response of raw audio bytes -- no JSON, no
    base64 -- exactly what the OpenAI SDK and Console integration expect
    (contrast with /tts/v1/voice:stream's NDJSON-with-base64 contract,
    open_ai_api/inworld_voice_stream.py).
    """
    # device_workers/device_worker.py's dispatch checks request.stream
    # directly (independent of which service method the API layer calls)
    # to decide streaming vs non-streaming on the worker/runner side -- this
    # must be True or the request silently runs through the dangerous
    # non-streaming decoder despite calling process_streaming_request below.
    tts_request.stream = True

    fmt = (tts_request.response_format or "mp3").lower()
    if fmt not in _STREAMING_MEDIA_TYPE_BY_FORMAT:
        raise HTTPException(
            status_code=422,
            detail=f"response_format={fmt!r} not supported -- must be one of "
            f"{sorted(_STREAMING_MEDIA_TYPE_BY_FORMAT)}",
        )

    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    # Prime the first chunk before creating the StreamingResponse -- see the
    # identical rationale in the non-inworld path below: Starlette commits
    # the response status as soon as StreamingResponse starts, before the
    # generator yields anything, so a validation failure (unknown voice_id,
    # input too long) must surface here or it silently becomes 200+empty.
    stream_iter = service.process_streaming_request(tts_request)
    try:
        first_chunk = await stream_iter.__anext__()
    except StopAsyncIteration:
        first_chunk = None
    except Exception as e:
        detail = str(e)
        if _VOICE_NOT_FOUND_MARKER in detail:
            raise HTTPException(status_code=400, detail=detail)
        if _TEXT_TOO_LONG_MARKER in detail or _TEXT_EXCEEDS_RECOMMENDED_LENGTH_MARKER in detail:
            raise HTTPException(status_code=422, detail=detail)
        raise HTTPException(status_code=500, detail=detail)

    media_type = _STREAMING_MEDIA_TYPE_BY_FORMAT[fmt]
    if first_chunk is None:

        async def _empty_body():
            return
            yield  # pragma: no cover -- makes this an async generator function

        return StreamingResponse(_empty_body(), media_type=media_type)

    raw_pcm, framerate, sampwidth, channels = parse_wav_chunk(base64.b64decode(first_chunk.audio_base64))

    async def pcm_chunks():
        yield raw_pcm
        async for chunk in stream_iter:
            frames, _fr, _sw, _ch = parse_wav_chunk(base64.b64decode(chunk.audio_base64))
            yield frames

    if fmt == "wav":

        async def body():
            yield build_streaming_wav_header(framerate, sampwidth, channels)
            async for pcm in pcm_chunks():
                yield pcm

        return StreamingResponse(body(), media_type=media_type)

    if fmt == "pcm":
        return StreamingResponse(pcm_chunks(), media_type=media_type)

    # mp3/ogg: continuous live-ffmpeg encode across all chunks.
    async def body():
        async for b in encode_pcm_stream(pcm_chunks(), framerate, channels, fmt):
            yield b

    return StreamingResponse(body(), media_type=media_type)


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

    For the inworld-tts runner, delegates entirely to
    _stream_openai_compatible_audio_bytes (see its docstring): that runner's
    non-streaming decoder path has a known pre-existing corruption bug, so
    this endpoint always uses the safe streaming pipeline internally and
    returns a continuous raw-bytes chunked response, regardless of the
    client's "stream" value.

    For every other TTS runner, unchanged: honors "stream" (NDJSON,
    base64-WAV-per-chunk, mirroring audio.py's STT streaming pattern) vs a
    single complete JSON/binary response.
    """
    if _is_inworld_tts_runner():
        try:
            return await _stream_openai_compatible_audio_bytes(tts_request, service)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

        # Prime the FIRST chunk here, before constructing the StreamingResponse
        # below -- Starlette commits the response status as soon as
        # StreamingResponse starts (before the generator produces anything),
        # so an exception raised inside the generator (e.g. input too long --
        # see _TEXT_TOO_LONG_MARKER below) can no longer change the status
        # code once that object exists: it would silently surface as 200 with
        # an empty body. Prefill/tokenization failures (the cases this guards
        # against) happen before any decode step, so priming costs nothing
        # extra on the failure path; on success it costs one chunk's worth of
        # generation added to time-to-first-byte, in exchange for a real
        # status code instead of a silent empty response.
        stream_iter = service.process_streaming_request(tts_request)
        try:
            first_chunk = await stream_iter.__anext__()
        except StopAsyncIteration:
            first_chunk = None

        async def result_stream():
            if first_chunk is not None:
                yield json.dumps(get_dict_response(first_chunk)) + "\n"
            async for partial in stream_iter:
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
        # Covers the streaming path too now that the first chunk is primed
        # above, before any response object is created.
        if _VOICE_NOT_FOUND_MARKER in str(e):
            raise HTTPException(status_code=400, detail=str(e))
        # Input text too long for the fixed prefill length (tokenizer-exact
        # check, raised host-side in tt_modeling._prefill_with_perf before any
        # device call -- already fast, just needs the right status code): a
        # client input error, not a server failure. Covers the streaming path
        # too, same reason as above.
        if _TEXT_TOO_LONG_MARKER in str(e) or _TEXT_EXCEEDS_RECOMMENDED_LENGTH_MARKER in str(e):
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
    _apply_inworld_response_format_default(tts_request)
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
