# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from typing import Optional

from config.constants import ImageSearchResponseFormat
from domain.image_search_request import ImageSearchRequest
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


async def _parse_image_search_request(
    request: Request,
    file: Optional[UploadFile] = File(None),
    response_format: Optional[str] = Form(ImageSearchResponseFormat.JSON.value),
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
            response_format=response_format
            or ImageSearchResponseFormat.VERBOSE_JSON.value,
            top_k=top_k,
            min_confidence=min_confidence,
        )

    raise HTTPException(
        status_code=400,
        detail="Use either multipart/form-data with file upload or application/json",
    )


@router.post("/search-image")
async def searchImage(
    image_search_request: ImageSearchRequest = Depends(_parse_image_search_request),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
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
        return {"image_data": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
