# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import logging
import traceback

from domain.text_to_speech_request import TextToSpeechRequest
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    Security,
)
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

logger = logging.getLogger(__name__)

MEDIA_TYPE_AUDIO_WAV = "audio/wav"

router = APIRouter()


@router.post("/speech")
async def text_to_speech(
    tts_request: TextToSpeechRequest,
    service=Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Convert text to speech using SpeechT5.

    Args:
        tts_request: Request containing text and optional speaker settings

    Returns:
        WAV audio file

    Raises:
        HTTPException: If text-to-speech fails.
    """
    try:
        logger.info(f"TTS request received: text='{tts_request.text[:50]}...'")
        result = await service.process_request(tts_request)

        # Check if result is valid
        if result is None:
            raise HTTPException(
                status_code=500, detail="No result returned from TTS model"
            )

        # Check if result is an error string
        if isinstance(result, str):
            raise HTTPException(
                status_code=500, detail=f"TTS processing error: {result}"
            )

        logger.info(f"TTS result type: {type(result)}")
        logger.info(f"TTS result has audio attr: {hasattr(result, 'audio')}")

        if hasattr(result, 'audio'):
            logger.info(f"Audio base64 length: {len(result.audio)}")
            audio_binary = base64.b64decode(result.audio)
            logger.info(f"Audio binary length: {len(audio_binary)}")
            logger.info(f"Audio header (first 12 bytes): {audio_binary[:12]}")
            return Response(content=audio_binary, media_type=MEDIA_TYPE_AUDIO_WAV)
        else:
            raise HTTPException(
                status_code=500, detail=f"Result has no audio attribute: {result}"
            )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"TTS error: {type(e).__name__}: {str(e)}"
        )
