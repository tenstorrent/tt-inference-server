# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.image_edit_request import ImageEditRequest
from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post('/edits')
async def edit_image(
    image_edit_request: ImageEditRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Edit an image based on the provided request.
    Returns:
        JSONResponse: The edited images as a list of base64 strings.
    Raises:
        HTTPException: If image editing fails.
    """
    try:
        result = await service.process_request(image_edit_request)
        return JSONResponse(content={"images": result})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
