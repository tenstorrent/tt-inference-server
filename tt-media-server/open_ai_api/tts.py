# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
from typing import Optional

from config.constants import AudioResponseFormat
from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import (
    APIRouter,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    Security,
)
from fastapi.responses import StreamingResponse
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

# Content type constants
CONTENT_TYPE_MULTIPART = "multipart/form-data"
CONTENT_TYPE_JSON = "application/json"
MEDIA_TYPE_AUDIO_WAV = "audio/wav"
MEDIA_TYPE_NDJSON = "application/x-ndjson"


async def parse_tts_request(
    request: Request,
    text: Optional[str] = Form(None),
    stream: Optional[bool] = Form(False),
    response_format: Optional[str] = Form(AudioResponseFormat.VERBOSE_JSON.value),
    speaker_id: Optional[str] = Form(None),
    speaker_embedding: Optional[str] = Form(None),
) -> TextToSpeechRequest:
    """Parse TTS request from form data or JSON"""
    content_type = request.headers.get("content-type", "").lower()

    if text is not None or content_type == CONTENT_TYPE_MULTIPART:
        # Form data request
        return TextToSpeechRequest(
            text=text,
            stream=stream or False,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
        )

    if CONTENT_TYPE_JSON in content_type:
        # JSON request
        json_body = await request.json()
        return TextToSpeechRequest(**json_body)

    raise HTTPException(
        status_code=400,
        detail=f"Use either {CONTENT_TYPE_MULTIPART} with text parameter or {CONTENT_TYPE_JSON}",
    )


router = APIRouter()


@router.post("/tts")
async def text_to_speech(
    tts_request: TextToSpeechRequest = Depends(parse_tts_request),
    service=Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Convert text to speech using the provided request.
    Supports both streaming and non-streaming based on the 'stream' field in the request.

    Returns:
        The audio response or StreamingResponse based on request.stream field.

    Raises:
        HTTPException: If text-to-speech fails.
    """
    try:
        if not tts_request.stream:
            result = await service.process_request(tts_request)
            if tts_request.response_format.lower() == AudioResponseFormat.TEXT.value:
                return Response(content=result.audio, media_type=MEDIA_TYPE_AUDIO_WAV)
            return get_dict_response(result)

        # Streaming path
        try:
            service.scheduler.check_is_model_ready()
        except Exception:
            raise HTTPException(status_code=405, detail="Model is not ready")

            async def result_stream():
                async for partial in service.process_streaming_request(tts_request):
                    if (
                        tts_request.response_format.lower()
                        == AudioResponseFormat.TEXT.value
                    ):
                        yield partial.audio + "\n"
                    else:
                        yield json.dumps(get_dict_response(partial)) + "\n"

            media_type = (
                MEDIA_TYPE_AUDIO_WAV
                if (
                    tts_request.response_format.lower()
                    == AudioResponseFormat.TEXT.value
                )
                else MEDIA_TYPE_NDJSON
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
