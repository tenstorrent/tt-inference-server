# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import time
from typing import Optional

from config.constants import AudioTasks, ResponseFormat
from config.settings import settings
from domain.audio_processing_request import AudioProcessingRequest
from domain.audio_text_response import AudioTextResponse
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
from telemetry.audio_metrics import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    char_count,
    record_stt_request,
)

# One task per deployment: settings.audio_task decides which router is live.
STT_TASK = (
    "translation"
    if settings.audio_task.lower() == AudioTasks.TRANSLATE.value
    else "transcription"
)


async def parse_audio_request(
    request: Request,
    file: Optional[UploadFile] = File(None),
    stream: Optional[bool] = Form(False),
    response_format: Optional[str] = Form(ResponseFormat.VERBOSE_JSON.value),
    is_preprocessing_enabled: Optional[bool] = Form(True),
    perform_diarization: Optional[bool] = Form(False),
    temperatures: Optional[str] = Form(None),
    compression_ratio_threshold: Optional[float] = Form(None),
    logprob_threshold: Optional[float] = Form(None),
    no_speech_threshold: Optional[float] = Form(None),
    return_timestamps: Optional[bool] = Form(False),
    prompt: Optional[str] = Form(None),
) -> AudioProcessingRequest:
    content_type = request.headers.get("content-type", "").lower()

    if file is not None:
        file_content = await file.read()

        return AudioProcessingRequest(
            file=file_content,
            stream=stream or False,
            response_format=response_format or ResponseFormat.VERBOSE_JSON.value,
            is_preprocessing_enabled=is_preprocessing_enabled
            if is_preprocessing_enabled is not None
            else True,
            perform_diarization=perform_diarization or False,
            temperatures=temperatures,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            return_timestamps=return_timestamps or False,
            prompt=prompt,
        )
    if "application/json" in content_type:
        json_body = await request.json()
        return AudioProcessingRequest(**json_body)
    raise HTTPException(
        status_code=400,
        detail="Use either multipart/form-data with file upload or application/json",
    )


transcriptions_router = APIRouter()


@transcriptions_router.post("/transcriptions")
async def transcribe_audio(
    audio_transcription_request: AudioProcessingRequest = Depends(parse_audio_request),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Transcribe audio using the provided request.
    Supports both streaming and non-streaming based on the 'stream' field in the request.

    Returns:
        The transcription result or StreamingResponse based on request.stream field.

    Raises:
        HTTPException: If transcription fails.
    """
    return await handle_audio_request(audio_transcription_request, service)


translations_router = APIRouter()


@translations_router.post("/translations")
async def translate_audio(
    audio_translation_request: AudioProcessingRequest = Depends(parse_audio_request),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Translate audio using the provided request.
    Supports both streaming and non-streaming based on the 'stream' field in the request.

    Returns:
        The translation result or StreamingResponse based on request.stream field.

    Raises:
        HTTPException: If translation fails.
    """
    return await handle_audio_request(audio_translation_request, service)


async def handle_audio_request(audio_request, service):
    start = time.perf_counter()
    is_text_format = audio_request.response_format.lower() == ResponseFormat.TEXT.value

    if not audio_request.stream:
        status = STATUS_SUCCESS
        result = None
        try:
            result = await service.process_request(audio_request)
            if is_text_format:
                return Response(content=result.text, media_type="text/plain")
            return get_dict_response(result)
        except HTTPException:
            status = STATUS_ERROR
            raise
        except Exception as e:
            status = STATUS_ERROR
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            record_stt_request(
                model_type=settings.model_runner,
                task=STT_TASK,
                streaming=False,
                status=status,
                duration_seconds=time.perf_counter() - start,
                audio_seconds=getattr(result, "duration", None),
                characters=char_count(getattr(result, "text", None)),
            )

    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        record_stt_request(
            model_type=settings.model_runner,
            task=STT_TASK,
            streaming=True,
            status=STATUS_ERROR,
            duration_seconds=time.perf_counter() - start,
        )
        raise HTTPException(status_code=405, detail="Model is not ready")

    async def result_stream():
        # Partial chunks carry incremental text; the final AudioTextResponse
        # (yielded for non-text formats) carries the full transcript and the
        # input audio duration, so it supersedes the accumulated count.
        streamed_characters = 0
        final_result = None
        status = STATUS_ERROR
        try:
            async for partial in service.process_streaming_request(audio_request):
                if isinstance(partial, AudioTextResponse):
                    final_result = partial
                else:
                    streamed_characters += char_count(partial.text) or 0
                if is_text_format:
                    yield partial.text + "\n"
                else:
                    yield json.dumps(get_dict_response(partial)) + "\n"
            status = STATUS_SUCCESS
        finally:
            if final_result is not None:
                audio_seconds = final_result.duration
                characters = char_count(final_result.text)
            else:
                # text format never yields the final result; preprocessing
                # stored the input duration on the request.
                audio_seconds = getattr(audio_request, "_duration", None)
                characters = streamed_characters
            record_stt_request(
                model_type=settings.model_runner,
                task=STT_TASK,
                streaming=True,
                status=status,
                duration_seconds=time.perf_counter() - start,
                audio_seconds=audio_seconds,
                characters=characters,
            )

    media_type = "text/plain" if is_text_format else "application/x-ndjson"
    return StreamingResponse(result_stream(), media_type=media_type)


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()


router = APIRouter()
if settings.audio_task.lower() == AudioTasks.TRANSLATE.value:
    router.include_router(translations_router)
else:
    router.include_router(transcriptions_router)
