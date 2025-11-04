# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()

@router.post('/generations')
async def generate_video(
    video_generation_request: VideoGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Generate a video based on the provided request.

    Returns:
        JSONResponse: The generated video as a base64 string.

    Raises:
        HTTPException: If video generation fails.
    """
    try:
        result = await service.process_request(video_generation_request)
        return JSONResponse(content={"videos": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
