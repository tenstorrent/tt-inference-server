# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any
from fastapi import APIRouter, Depends, Response, Security, HTTPException
from domain.image_generate_request import ImageGenerateRequest
from domain.audio_transcription_request import AudioTranscriptionRequest
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/generations')
async def generate_image(
    image_generate_request: ImageGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
) -> Response:
    """
    Generate an image based on the provided request.

    Returns:
        Response: The generated image as a PNG.

    Raises:
        HTTPException: If image generation fails.
    """
    try:
        result = await service.process_request(image_generate_request)
        return Response(content=result, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@router.get('/tt-liveness')
def liveness(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Check service liveness and model readiness.

    Returns:
        dict: Dictionary containing service status and model information.

    Raises:
        HTTPException: If service is unavailable or model check fails.
    """
    try:
        return {'status': 'alive', **service.check_is_model_ready()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liveness check failed: {e}")

@router.post('/tt-deep-reset')
async def deep_reset(service: BaseService = Depends(service_resolver)) -> dict[str, Any]:
    """
    Schedules a deep reset of the service and its model.

    Returns:
        dict: Status message indicating the result of the reset operation.

    Raises:
        HTTPException: If reset fails.
    """
    try:
        await service.deep_reset()
        return {'status': 'reset successful'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")