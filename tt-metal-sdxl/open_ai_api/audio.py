# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import json
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
        async def result_stream():
            # Get the async generator from the service  
            async_generator = await service.process_request(audio_transcription_request, stream=True)
            
            # Stream results as JSON lines for easier parsing
            async for partial in async_generator:
                # Normalize all results to dict format, then convert to JSON
                if isinstance(partial, dict):
                    result = partial
                elif isinstance(partial, str):
                    result = {"type": "partial_text", "text": partial}
                else:
                    result = {"type": "data", "content": str(partial)}
                
                yield json.dumps(result) + "\n"
        return StreamingResponse(result_stream(), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
