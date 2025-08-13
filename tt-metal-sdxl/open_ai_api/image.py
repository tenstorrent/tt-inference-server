# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any
from fastapi import APIRouter, Depends, Response, Security, HTTPException
from domain.image_generate_request import ImageGenerateRequest
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