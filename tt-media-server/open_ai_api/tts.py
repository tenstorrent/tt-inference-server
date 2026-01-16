# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
import base64
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

            # Check if result is valid
            if result is None:
                raise HTTPException(status_code=500, detail="No result returned from TTS model")

            # Check if result is an error string
            if isinstance(result, str):
                raise HTTPException(status_code=500, detail=f"TTS processing error: {result}")

            # Return raw audio file (WAV binary) for TEXT/WAV format
            if tts_request.response_format.lower() in [AudioResponseFormat.TEXT.value, AudioResponseFormat.WAV.value]:
                # Decode base64 to binary audio
                audio_binary = base64.b64decode(result.audio)
                return Response(content=audio_binary, media_type=MEDIA_TYPE_AUDIO_WAV)
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {type(e).__name__}: {str(e)}")


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()
