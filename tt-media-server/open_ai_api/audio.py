# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
from typing import Optional

from domain.audio_transcription_request import AudioTranscriptionRequest
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

router = APIRouter()


async def parse_audio_request(
    request: Request,
    file: Optional[UploadFile] = File(None),
    stream: Optional[bool] = Form(False),
    return_json_response: Optional[bool] = Form(True),
    is_preprocessing_enabled: Optional[bool] = Form(True),
    perform_diarization: Optional[bool] = Form(False),
) -> AudioTranscriptionRequest:
    content_type = request.headers.get("content-type", "").lower()

    if file is not None:
        file_content = await file.read()
        return AudioTranscriptionRequest(
            file=file_content,
            stream=stream or False,
            return_json_response=return_json_response,
            is_preprocessing_enabled=is_preprocessing_enabled
            if is_preprocessing_enabled is not None
            else True,
            perform_diarization=perform_diarization or False,
        )
    if "application/json" in content_type:
        json_body = await request.json()
        return AudioTranscriptionRequest(**json_body)
    raise HTTPException(
        status_code=400,
        detail="Use either multipart/form-data with file upload or application/json with AudioTranscriptionRequest",
    )


@router.post("/transcriptions")
async def transcribe_audio(
    audio_transcription_request: AudioTranscriptionRequest = Depends(
        parse_audio_request
    ),
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
    try:
        if not audio_transcription_request.stream:
            result = await service.process_request(audio_transcription_request)
            if not hasattr(result, "to_dict"):
                raise ValueError(
                    f"Unexpected response type: {type(result).__name__}. Expected response class with to_dict() method."
                )

            if audio_transcription_request.return_json_response:
                return result.to_dict()
            else:
                return Response(content=result.text, media_type="text/plain")
        else:
            try:
                service.scheduler.check_is_model_ready()
            except Exception:
                raise HTTPException(status_code=405, detail="Model is not ready")

            async def result_stream():
                async for partial in service.process_streaming_request(
                    audio_transcription_request
                ):
                    if not hasattr(partial, "to_dict"):
                        raise ValueError(
                            f"Unexpected response type: {type(partial).__name__}. Expected response class with to_dict() method."
                        )

                    if audio_transcription_request.return_json_response:
                        result = partial.to_dict()
                        yield json.dumps(result) + "\n"
                    else:
                        yield partial.text + "\n"

            media_type = (
                "application/x-ndjson"
                if audio_transcription_request.return_json_response
                else "text/plain"
            )
            return StreamingResponse(result_stream(), media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
