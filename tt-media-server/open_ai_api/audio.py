# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
from typing import Optional

from config.constants import AudioTasks, ResponseFormat
from config.settings import settings
from domain.audio_processing_request import AudioProcessingRequest
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
from security.api_key_cheker import get_api_key


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
    try:
        if not audio_request.stream:
            result = await service.process_request(audio_request)
            if audio_request.response_format.lower() == ResponseFormat.TEXT.value:
                return Response(content=result.text, media_type="text/plain")
            return get_dict_response(result)
        else:
            try:
                service.scheduler.check_is_model_ready()
            except Exception:
                raise HTTPException(status_code=405, detail="Model is not ready")

            async def result_stream():
                async for partial in service.process_streaming_request(audio_request):
                    if (
                        audio_request.response_format.lower()
                        == ResponseFormat.TEXT.value
                    ):
                        yield partial.text + "\n"
                    else:
                        yield json.dumps(get_dict_response(partial)) + "\n"

            media_type = (
                "text/plain"
                if (audio_request.response_format.lower() == ResponseFormat.TEXT.value)
                else "application/x-ndjson"
            )
            return StreamingResponse(result_stream(), media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
