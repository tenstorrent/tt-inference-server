# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import io
from config.settings import settings
from config.constants import ModelRunners, ModelServices
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
from fastapi import APIRouter, Depends, Security, HTTPException
from fastapi.responses import JSONResponse, Response
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key


generate_image_router = APIRouter()

@generate_image_router.post('/generations', response_class=Response, responses={
    200: {
        "content": {"image/png": {}},
        "description": "Generated image as PNG file",
    }
})
async def generate_image(
    image_generate_request: ImageGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Generate an image based on the provided request.

    Returns:
        Response: The generated image as a downloadable PNG file.

    Raises:
        HTTPException: If image generation fails.
    """
    try:
        result = await service.process_request(image_generate_request)
        
        # Handle different result types
        if isinstance(result, list) and len(result) > 0:
            # If result is a list, take the first image
            image = result[0]
        else:
            # Single image
            image = result
            
        # Check if it's a PIL Image
        if hasattr(image, 'save'):
            # Convert PIL Image to binary PNG data
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return Response(
                content=img_buffer.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=generated_image.png"
                }
            )
        else:
            # Fallback if it's not a PIL Image
            raise HTTPException(status_code=500, detail="Invalid image format received")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


image_to_image_router = APIRouter()

@image_to_image_router.post('/image-to-image', response_class=Response, responses={
    200: {
        "content": {"image/png": {}},
        "description": "Generated image as PNG file",
    }
})
async def image_to_image(
    image_to_image_request: ImageToImageRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Generate an image based on the provided request.
    Returns:
        Response: The generated image as a downloadable PNG file.
    Raises:
        HTTPException: If image generation fails.
    """
    try:
        result = await service.process_request(image_to_image_request)
        
        # Handle different result types
        if isinstance(result, list) and len(result) > 0:
            # If result is a list, take the first image
            image = result[0]
        else:
            # Single image
            image = result
            
        # Check if it's a PIL Image
        if hasattr(image, 'save'):
            # Convert PIL Image to binary PNG data
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return Response(
                content=img_buffer.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=image_to_image.png"
                }
            )
        else:
            # Fallback if it's not a PIL Image
            raise HTTPException(status_code=500, detail="Invalid image format received")
            
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


edit_image_router = APIRouter()

@edit_image_router.post('/edits', response_class=Response, responses={
    200: {
        "content": {"image/png": {}},
        "description": "Edited image as PNG file",
    }
})
async def edit_image(
    image_edit_request: ImageEditRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key)
):
    """
    Edit an image based on the provided request.
    Returns:
        Response: The edited image as a downloadable PNG file.
    Raises:
        HTTPException: If image editing fails.
    """
    try:
        result = await service.process_request(image_edit_request)
        
        # Handle different result types
        if isinstance(result, list) and len(result) > 0:
            # If result is a list, take the first image
            image = result[0]
        else:
            # Single image
            image = result
            
        # Check if it's a PIL Image
        if hasattr(image, 'save'):
            # Convert PIL Image to binary PNG data
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            return Response(
                content=img_buffer.getvalue(),
                media_type="image/png",
                headers={
                    "Content-Disposition": "inline; filename=edited_image.png"
                }
            )
        else:
            # Fallback if it's not a PIL Image
            raise HTTPException(status_code=500, detail="Invalid image format received")
            
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
