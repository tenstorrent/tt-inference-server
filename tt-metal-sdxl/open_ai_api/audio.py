# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import StreamingResponse
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


# Streaming endpoint for Whisper transcription
@router.post('/transcriptions/stream')
async def transcribe_audio_stream(
    audio_transcription_request: AudioTranscriptionRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Stream transcription results as they are generated.

    Returns:
        StreamingResponse yielding partial transcription results.
        
    Raises:
        HTTPException: If transcription fails.
    """
    try:
        # The service must support a streaming mode (e.g., pass stream=True)
        # This assumes your service.process_request supports a 'stream' kwarg
        async def result_stream():
            # If your service/process_request is not async generator, adapt as needed
            async for partial in service.process_request(audio_transcription_request, stream=True):
                # You can yield plain text, or JSON lines, etc.
                yield partial + "\n"
        return StreamingResponse(result_stream(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))