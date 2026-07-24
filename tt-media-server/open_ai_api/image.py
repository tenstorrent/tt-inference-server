# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from config.constants import ModelRunners
from config.settings import settings
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
import json
import os
import shutil

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

_LORA_DIR = os.environ.get("LORA_DIR", "/loras")
_LORA_STATE = os.path.join(_LORA_DIR, "active_lora.json")

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


# ----------------------------- FLUX.1-Kontext-dev -----------------------------
# Kontext reuses the shared image endpoints — /generations (text->image) and
# /edits (instruction edit; mask is optional and ignored by Kontext) — so it does
# not redefine its own copies. The only Kontext-specific surface is LoRA
# management (/lora/*): upload a FLUX.1 LoRA, record it as active, and rebuild the
# worker with the LoRA fused in. (Optional JP->EN prompt translation ships as a
# web-console add-on in a separate repo, not here.)
lora_router = APIRouter()


@lora_router.post("/lora/apply")
async def lora_apply(
    file: UploadFile = File(...),
    scale: float = Form(1.0),
    name: str = Form(None),
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Upload a FLUX.1 LoRA and activate it. Saves the file, records it as the
    active LoRA, and restarts the device worker so the pipeline is rebuilt with
    the LoRA fused into the transformer. Returns immediately with status
    'rebuilding'; poll /v1/images/lora/status until model_ready is true again (~2-3 min)."""
    try:
        up = os.path.join(_LORA_DIR, "uploaded")
        os.makedirs(up, exist_ok=True)
        # The client-supplied name is kept only as a display label — it never
        # enters a filesystem path. The upload is written to a fixed,
        # server-controlled filename so no user-controlled data is used in a path
        # expression (avoids path traversal; satisfies CodeQL).
        display_name = os.path.basename(name or file.filename or "lora.safetensors")
        dest = os.path.join(up, "active_lora.safetensors")
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with open(_LORA_STATE, "w") as f:
            json.dump({"path": dest, "scale": float(scale), "name": display_name}, f)
        await service.deep_reset()  # restart worker -> rebuild pipeline with LoRA
        return JSONResponse(
            content={
                "status": "rebuilding",
                "active": {"name": display_name, "scale": float(scale)},
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@lora_router.post("/lora/clear")
async def lora_clear(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Deactivate any LoRA and rebuild the base pipeline (~2-3 min)."""
    try:
        if os.path.isfile(_LORA_STATE):
            os.remove(_LORA_STATE)
        await service.deep_reset()
        return JSONResponse(content={"status": "rebuilding"})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@lora_router.get("/lora/status")
async def lora_status(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Current LoRA + whether the worker has finished (re)building."""
    active = None
    if os.path.isfile(_LORA_STATE):
        try:
            d = json.load(open(_LORA_STATE))
            active = {"name": d.get("name"), "scale": d.get("scale")}
        except Exception:
            active = None
    try:
        ready = bool(service.check_is_model_ready().get("model_ready"))
    except Exception:
        ready = False
    return JSONResponse(content={"active": active, "model_ready": ready})


router = APIRouter()

if settings.model_runner == ModelRunners.TT_SDXL_IMAGE_TO_IMAGE.value:
    router.include_router(image_to_image_router)
elif settings.model_runner == ModelRunners.TT_SDXL_EDIT.value:
    router.include_router(edit_image_router)
elif settings.model_runner == ModelRunners.TT_FLUX_1_KONTEXT_DEV.value:
    # Reuse the shared endpoints for text->image (/generations) and instruction
    # edit (/edits, mask optional); add only the Kontext-specific LoRA routes.
    router.include_router(generate_image_router)
    router.include_router(edit_image_router)
    router.include_router(lora_router)
else:
    router.include_router(generate_image_router)
