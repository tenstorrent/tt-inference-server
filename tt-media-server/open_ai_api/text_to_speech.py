# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import APIRouter, Depends, HTTPException, Response, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

router = APIRouter()


async def handle_tts_request(tts_request, service):
    """
    Handle TTS request with different response formats.

    Returns:
        If response_format is "audio" or "wav", returns WAV bytes directly.
        Otherwise, returns JSON with base64-encoded audio.
    """
    try:
        result = await service.process_request(tts_request)
        if tts_request.response_format.lower() in ("audio", "wav"):
            # Return WAV bytes directly
            if hasattr(result, "wav_bytes") and result.wav_bytes:
                return Response(content=result.wav_bytes, media_type="audio/wav")
            else:
                raise HTTPException(
                    status_code=500,
                    detail="WAV bytes not available in response",
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

    Returns:
        If response_format is "audio" or "wav", returns WAV bytes directly.
        Otherwise, returns JSON with base64-encoded audio.

    Raises:
        HTTPException: If text-to-speech fails.
    """
    return await handle_tts_request(tts_request, service)


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()
