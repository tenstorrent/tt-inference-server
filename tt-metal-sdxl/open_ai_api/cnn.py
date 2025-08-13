# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any
from fastapi import APIRouter, Depends, Response, Security
from domain.image_search_request import ImageSearchRequest
from model_services.base_model import BaseModel
from resolver.model_resolver import model_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/search-image')
async def searchImage(image_search_request: ImageSearchRequest, service: BaseModel = Depends(model_resolver), api_key: str = Security(get_api_key)):
    try:
        result = await service.processImage(image_search_request)
        return {"image_data": result, "status": "success"}
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