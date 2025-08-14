# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from fastapi import APIRouter, Depends, Security, HTTPException
from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/transcriptions')
async def transcribe_audio(
    audio_transcription_request: AudioTranscriptionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Transcribe audio using the provided request.

    Returns:
        The transcription result, typically as a JSON-compatible dict or string.

    Raises:
        HTTPException: If transcription fails.
    """
    try:
        result = await service.process_request(audio_transcription_request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))