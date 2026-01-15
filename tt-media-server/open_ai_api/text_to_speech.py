# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Security,
)
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post("/speech")
async def text_to_speech(
    tts_request: TextToSpeechRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Convert text to speech using the provided request.

    Returns:
        Generated audio base64 encoded.

    Raises:
        HTTPException: If text-to-speech fails.
    """
    try:
        result = await service.process_request(tts_request)
        return get_dict_response(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_dict_response(obj):
    if not hasattr(obj, "to_dict"):
        raise ValueError(
            f"Unexpected response type: {type(obj).__name__}. Expected response class with to_dict() method."
        )
    return obj.to_dict()
