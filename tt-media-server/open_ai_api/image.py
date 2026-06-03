# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from domain.image_generate_request import ImageGenerateRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

generate_image_router = APIRouter()


@generate_image_router.post("/generations")
async def generate_image(
    image_generate_request: ImageGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Generate an image based on the provided request.

    Returns:
        JSONResponse: The generated images as a list of base64 strings.

    Raises:
        HTTPException: If image generation fails.
    """
    try:
        import time as _time

        _t0 = _time.time()
        result = await service.process_request(image_generate_request)
        _elapsed = round(_time.time() - _t0, 2)
        return JSONResponse(content={"images": result, "generation_time": _elapsed})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

router = APIRouter()
router.include_router(generate_image_router)
