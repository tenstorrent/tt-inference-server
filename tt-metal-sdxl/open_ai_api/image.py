# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any
from fastapi import APIRouter, Depends, Response, Security
from domain.image_generate_request import ImageGenerateRequest
from model_services.base_model import BaseModel
from resolver.model_resolver import model_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/generations')
async def generateImage(imageGenerateRequest: ImageGenerateRequest, service: BaseModel = Depends(model_resolver), api_key: str = Security(get_api_key)):
    try:
        result = await service.processImage(imageGenerateRequest)
        return Response(content=result, media_type="image/png")
    except Exception as e:
        return Response(status_code=500, content=str(e))


@router.get('/tt-liveness')
def liveness(service: BaseModel = Depends(model_resolver)) -> dict[str, Any]:
    """
    Check service liveness and model readiness.
    
    Returns:
        Dict containing service status and model information
        
    Raises:
        HTTPException: If service is unavailable or model check fails
    """
    return {'status': 'alive', **service.checkIsModelReady()}

@router.get('/tt-deep-reset')
async def deepReset(service: BaseModel = Depends(model_resolver)) -> dict[str, Any]:
    """
    Schedules a deep reset of the service and its model.
    
    Returns:
        OK if successful, error message if failed

    Raises:
        HTTPException: If reset fails
    """
    try:
        await service.deep_reset()
        return {'status': 'reset successful'}
    except Exception as e:
        return {'status': 'reset failed', 'error': str(e)}