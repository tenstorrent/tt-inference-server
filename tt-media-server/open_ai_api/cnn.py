# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


import logging
from typing import Optional

from config.constants import ResponseFormat
from domain.image_search_request import ImageSearchRequest
from domain.image_search_response import ImageSearchResponse
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()

# Constants
CONTENT_TYPE_JSON = "application/json"

logger = logging.getLogger(__name__)


async def _parse_image_search_request(
    request: Request,
    file: Optional[UploadFile] = File(None),
    response_format: Optional[str] = Form(ResponseFormat.JSON.value),
    top_k: Optional[int] = Form(3),
    min_confidence: Optional[float] = Form(70.0),
) -> ImageSearchRequest:
    """Parse the image search request from the request body."""
    content_type = request.headers.get("content-type", "").lower()

    if CONTENT_TYPE_JSON in content_type:
        json_body = await request.json()
        return ImageSearchRequest(**json_body)

    if file is not None:
        file_content = await file.read()

        return ImageSearchRequest(
            prompt=file_content,
            response_format=response_format,
            top_k=top_k,
            min_confidence=min_confidence,
        )

    raise HTTPException(
        status_code=400,
        detail="Use either multipart/form-data with file upload or application/json",
    )


@router.post("/search-image", response_model=ImageSearchResponse)
async def searchImage(
    image_search_request: ImageSearchRequest = Depends(_parse_image_search_request),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
) -> ImageSearchResponse:
    """
    Process image search request using CNN model.

    Supports two input methods:
        - multipart/form-data: Upload image file directly
        - application/json: Send base64-encoded image in 'prompt' field

    Returns:
        The image search result with success status.

    Raises:
        HTTPException: If image search fails.
    """
    try:
        result = await service.process_request(image_search_request)
        logger.info("✅ Successfully finished image search result, result: %s", result)
        # Wrap single result in list to match ImageSearchResponse schema
        image_data = [result] if not isinstance(result, list) else result
        return ImageSearchResponse(image_data=image_data)
    except Exception as e:
        logger.error("❌ Error processing image search request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
