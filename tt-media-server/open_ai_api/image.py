# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.constants import ModelRunners
from config.settings import settings
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
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
        result = await service.process_request(image_generate_request)
        return JSONResponse(content={"images": result})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


image_to_image_router = APIRouter()


@image_to_image_router.post("/image-to-image")
async def image_to_image(
    image_to_image_request: ImageToImageRequest,
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
        result = await service.process_request(image_to_image_request)
        return JSONResponse(content={"images": result})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


edit_image_router = APIRouter()


@edit_image_router.post("/edits")
async def edit_image(
    image_edit_request: ImageEditRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
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


router = APIRouter()

if settings.model_runner == ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value:
    router.include_router(image_to_image_router)
elif settings.model_runner == ModelRunners.TT_SDXL_EDIT.value:
    router.include_router(edit_image_router)
else:
    router.include_router(generate_image_router)
