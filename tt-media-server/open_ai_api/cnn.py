# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.image_search_request import ImageSearchRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post("/search-image")
async def searchImage(
    image_search_request: ImageSearchRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Process image search request using CNN model.

    Returns:
        The image search result with success status.

    Raises:
        HTTPException: If image search fails.
    """
    try:
        result = await service.process_request(image_search_request)
        return {"image_data": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
