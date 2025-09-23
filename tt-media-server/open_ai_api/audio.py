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
    Supports both streaming and non-streaming based on the 'stream' field in the request.

    Returns:
        The transcription result or StreamingResponse based on request.stream field.

    Raises:
        HTTPException: If transcription fails.
    """
    try:
        if not audio_transcription_request.stream:
            result = await service.process_request(audio_transcription_request)
            if not hasattr(result, 'to_dict'):
                raise ValueError(
                    f"Unexpected response type: {type(result).__name__}. Expected response class with to_dict() method."
                )
            
            return result.to_dict()
        else:
            try:
                service.scheduler.check_is_model_ready()
            except Exception as e:
                raise HTTPException(status_code=405, detail="Model is not ready")
            
            async def result_stream():
                async_generator = await service.process_request(audio_transcription_request)
                
                # Stream results as JSON lines for easier parsing
                async for partial in async_generator:
                    if not hasattr(partial, 'to_dict'):
                        raise ValueError(
                            f"Unexpected response type: {type(partial).__name__}. Expected response class with to_dict() method."
                        )
                    
                    result = partial.to_dict()
                    yield json.dumps(result) + "\n"
            
            return StreamingResponse(result_stream(), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
