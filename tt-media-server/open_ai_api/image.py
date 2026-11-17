# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from config.constants import ModelRunners
from config.settings import settings
from domain.flux3_image_klein_request import (
    FLUX3_IMAGE_KLEIN,
    FLUX3_KLEIN_SIZES,
    Flux3ImageKleinRequest,
)
from domain.flux3_image_klein_sync_request import (
    KleinImageEditRequest,
    KleinImageGenerateRequest,
    KleinImageToImageRequest,
)
from domain.image_edit_request import ImageEditRequest
from domain.image_generate_request import BaseImageRequest, ImageGenerateRequest
from domain.image_to_image_request import ImageToImageRequest
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

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
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from model_services.base_service import BaseService
from model_services.generation_service import GenerationService, read_manifest
from pydantic import ValidationError
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key

_LORA_DIR = os.environ.get("LORA_DIR", "/loras")
_LORA_STATE = os.path.join(_LORA_DIR, "active_lora.json")
# FLUX.1 LoRAs run ~20-300 MB; the cap stops a stray upload from filling the
# container disk. Raise LORA_MAX_BYTES for unusually large adapters.
_LORA_MAX_BYTES = int(os.environ.get("LORA_MAX_BYTES", 512 * 1024 * 1024))
_LORA_CHUNK_BYTES = 1024 * 1024
# Upper bound on the LoRA fusion weight, matching ImageGenerateRequest.lora_scale.
_LORA_MAX_SCALE = 2.0
# safetensors caps its JSON header at 100 MB; anything larger is not a real file.
_SAFETENSORS_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _check_safetensors(path: str) -> None:
    """Confirm the upload really is a safetensors file. The format is an 8-byte
    little-endian header length followed by that many bytes of JSON metadata, so
    this reads only the header — never the tensor payload."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            raise ValueError("file is too small to be a safetensors file")
        header_len = int.from_bytes(raw_len, "little")
        if header_len == 0 or header_len > _SAFETENSORS_MAX_HEADER_BYTES:
            raise ValueError("safetensors header length is out of range")
        if 8 + header_len > size:
            raise ValueError("safetensors header length exceeds the file size")
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError("safetensors header is not valid JSON") from e
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")


async def _save_capped(upload: UploadFile, dest: str) -> int:
    """Stream the upload to dest, aborting once it exceeds _LORA_MAX_BYTES.
    Streaming (rather than upload.read() in one shot) keeps an oversized file
    from being buffered in full before the limit is noticed."""
    written = 0
    with open(dest, "wb") as f:
        while chunk := await upload.read(_LORA_CHUNK_BYTES):
            written += len(chunk)
            if written > _LORA_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"LoRA exceeds the {_LORA_MAX_BYTES} byte limit",
                )
            f.write(chunk)
    return written


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
    # The client-supplied name is kept only as a display label — it never enters
    # a filesystem path. The upload is written to a fixed, server-controlled
    # filename so no user-controlled data is used in a path expression (avoids
    # path traversal; satisfies CodeQL).
    display_name = os.path.basename(name or file.filename or "lora.safetensors")

    # Reject bad metadata before touching the disk.
    if not display_name.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="LoRA must be a .safetensors file")
    if not 0.0 <= scale <= _LORA_MAX_SCALE:
        raise HTTPException(
            status_code=422,
            detail=f"scale must be between 0.0 and {_LORA_MAX_SCALE}, got {scale}",
        )

    up = os.path.join(_LORA_DIR, "uploaded")
    dest = os.path.join(up, "active_lora.safetensors")
    # Stage under .part and swap only once the file validates, so a rejected
    # upload leaves the currently active LoRA in place.
    staged = dest + ".part"
    try:
        os.makedirs(up, exist_ok=True)
        await _save_capped(file, staged)
        try:
            _check_safetensors(staged)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid safetensors file: {e}"
            )
        os.replace(staged, dest)
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
    finally:
        if os.path.isfile(staged):
            os.remove(staged)


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


# ----------------------------- FLUX.3 Image klein -----------------------------
# Synchronous /generations, /image-to-image and /edits (without a mask) for
# clients that wait for the image (Console). Each request is one ordinary Klein job, queued with the
# /v1/generations jobs and rendered one at a time, waited for here and returned
# inline, then deleted. A client that leaves, or a wait that times out, only
# cancels a job no worker has claimed: a render is never stopped.

# Well under Console's 120 s request timeout.
KLEIN_WAIT_SECONDS = 90.0
_KLEIN_POLL_SECONDS = 0.05
# The image API's field each Klein field is filled from, for the errors.
_KLEIN_ERROR_FIELDS = {
    "images": "image",
    "aspect_ratio": "width",
    "size": "width",
    "format": "image_return_format",
}


def _klein_errors(error: ValidationError) -> RequestValidationError:
    errors = error.errors(include_url=False, include_context=False, include_input=False)
    return RequestValidationError(
        [
            {
                **e,
                # A model-level error is the output size's.
                "loc": ("body", _KLEIN_ERROR_FIELDS.get(e["loc"][0], e["loc"][0]))
                if e["loc"]
                else ("body", "width"),
            }
            for e in errors
        ]
    )


def _klein_request(
    request: BaseImageRequest, image: Optional[str]
) -> Flux3ImageKleinRequest:
    """The Klein job for an image API request; 422 for what Klein can't make."""
    image_format = (request.image_return_format or "JPEG").lower()
    fields = {"prompt": request.prompt, "format": image_format, "seed": request.seed}
    if request.width is not None:
        shape = (request.width, request.height)
        keys = [key for key, size in FLUX3_KLEIN_SIZES.items() if size == shape]
        if not keys:
            served = ", ".join(
                f"{w}x{h}" for w, h in Flux3ImageKleinRequest.served_sizes().values()
            )
            raise RequestValidationError(
                [
                    {
                        "loc": ("body", "width"),
                        "msg": f"{shape[0]}x{shape[1]} is not served; this "
                        f"deployment serves {served}",
                        "type": "value_error",
                    }
                ]
            )
        fields["aspect_ratio"], fields["size"] = keys[0]
    else:
        # Console doesn't send a size: the largest square this deployment serves.
        served = Flux3ImageKleinRequest.served_sizes()
        fields["aspect_ratio"], fields["size"] = (
            ("1:1", "4K") if ("1:1", "4K") in served else ("1:1", "2K")
        )
    if image is not None and not image.startswith(("https://", "http://")):
        # As ImageManager.base64_to_pil_image takes it for Kontext and SDXL:
        # bare base64 or a data URL, its padding restored. Klein takes a data
        # URI; the decoder reads the format from the bytes.
        prefix, _, data = image.rpartition(",")
        image = (
            (prefix or "data:image/png;base64") + "," + data + "=" * (-len(data) % 4)
        )
    if image is not None:
        fields["images"] = [image]
    try:
        klein = Flux3ImageKleinRequest(**fields)
    except ValidationError as e:
        raise _klein_errors(e)
    if image_format == "jpeg":
        klein._image_quality = request.image_quality
    return klein


async def _wait_for_job(
    service: GenerationService,
    job_id: str,
    disconnected: Callable[[], Awaitable[bool]],
) -> dict:
    """The job once it ends; 504 after KLEIN_WAIT_SECONDS or when the client
    leaves, cancelling it only if no worker has claimed it."""
    deadline = time.monotonic() + KLEIN_WAIT_SECONDS
    while True:
        metadata = service.get_job_metadata(job_id)
        if metadata is None:
            raise HTTPException(status_code=500, detail=f"Job {job_id} is gone")
        if metadata["status"] in ("completed", "failed", "cancelled"):
            return metadata
        gone = await disconnected()
        if gone or time.monotonic() > deadline:
            cancelled = service.cancel_waiting_job(job_id)
            reason = (
                "client disconnected"
                if gone
                else (f"not done within {KLEIN_WAIT_SECONDS:g} s")
            )
            outcome = (
                "cancelled before it started"
                if cancelled
                else f"it is still rendering: GET /v1/jobs/{job_id}"
            )
            raise HTTPException(
                status_code=504, detail=f"Job {job_id} {reason}; {outcome}"
            )
        await asyncio.sleep(_KLEIN_POLL_SECONDS)


def _read_image(result_path: str) -> str:
    (output,) = read_manifest(result_path)
    return base64.b64encode((Path(result_path) / output["file"]).read_bytes()).decode()


async def _klein_render(
    service: GenerationService, http_request: Request, request: Flux3ImageKleinRequest
) -> JSONResponse:
    started = time.monotonic()
    try:
        job = await service.submit(request, FLUX3_IMAGE_KLEIN.id)
    except RequestValidationError as e:
        # A reference refused at submit is the image API's `image`.
        raise RequestValidationError(
            [
                {**error, "loc": ("body", "image")}
                if tuple(error["loc"][:2]) == ("body", "images")
                else error
                for error in e.errors()
            ]
        )
    metadata = await _wait_for_job(service, job["id"], http_request.is_disconnected)
    if metadata["status"] == "failed":
        raise HTTPException(status_code=500, detail=metadata["error"]["message"])
    if metadata["status"] == "cancelled":
        raise HTTPException(status_code=409, detail=f"Job {job['id']} was cancelled")
    try:
        image = await asyncio.to_thread(
            _read_image, service.get_job_result_path(job["id"])
        )
    finally:
        await service.delete_job(job["id"])
    return JSONResponse(
        content={
            "images": [image],
            "generation_time": round(time.monotonic() - started, 2),
        }
    )


class _KleinRoute(APIRoute):
    """Answers a 422 without each error's `input`, which FastAPI echoes: for an
    image or a mask, megabytes back to the client."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def route_handler(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as e:
                errors = [
                    {key: value for key, value in error.items() if key != "input"}
                    for error in e.errors()
                ]
                return JSONResponse(
                    status_code=422, content={"detail": jsonable_encoder(errors)}
                )

        return route_handler


klein_router = APIRouter(route_class=_KleinRoute)


@klein_router.post("/generations")
async def klein_generate_image(
    http_request: Request,
    image_generate_request: KleinImageGenerateRequest,
    service: GenerationService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Text to image with FLUX.3 Image klein; waits for the image."""
    try:
        request = _klein_request(image_generate_request, image=None)
        return await _klein_render(service, http_request, request)
    except (HTTPException, RequestValidationError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@klein_router.post("/image-to-image")
async def klein_image_to_image(
    http_request: Request,
    image_to_image_request: KleinImageToImageRequest,
    service: GenerationService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """`image` as FLUX.3 Image klein's reference; waits for the image."""
    try:
        request = _klein_request(
            image_to_image_request, image=image_to_image_request.image
        )
        return await _klein_render(service, http_request, request)
    except (HTTPException, RequestValidationError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@klein_router.post("/edits")
async def klein_edit_image(
    http_request: Request,
    image_edit_request: KleinImageEditRequest,
    service: GenerationService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """As /image-to-image; a mask is refused (422)."""
    try:
        request = _klein_request(image_edit_request, image=image_edit_request.image)
        return await _klein_render(service, http_request, request)
    except (HTTPException, RequestValidationError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mounted beside the generations API: only Klein's deployment serves images.
generation_router = APIRouter()
if settings.model_runner == ModelRunners.TT_FLUX_3_IMAGE_KLEIN.value:
    generation_router.include_router(klein_router)

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
