# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import os
import tempfile
import time as _time
from typing import Annotated, Optional

from config.constants import JobTypes
from config.settings import settings
from domain.video_generate_request import VideoGenerateRequest
from domain.video_i2v_generate_request import (
    ImagePromptEntry,
    VideoI2VGenerateRequest,
)
from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from model_services.base_job_service import BaseJobService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.video_manager import VideoManager

router = APIRouter()


# Smallest valid PNG (1x1 transparent) so OpenAPI "Try it out" actually
# round-trips through ImagePromptEntry's base64 validator instead of failing
# with 422 on a non-decodable placeholder string.
_OPENAPI_IMAGE_PLACEHOLDER = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Multipart safety knobs — same shape as Stability/Runway/OpenAI image edits.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_UPLOAD_READ_CHUNK = 64 * 1024
_ALLOWED_IMAGE_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


def _validate_image_content_type(upload: UploadFile) -> None:
    """Reject non-image uploads at the boundary with 415 before reading bytes."""
    if upload.content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image content_type {upload.content_type!r}; "
                f"allowed: {sorted(_ALLOWED_IMAGE_CONTENT_TYPES)}"
            ),
        )


async def _read_capped_upload(upload: UploadFile) -> bytes:
    """Stream-read upload bytes with a hard cap to prevent RAM exhaustion.

    A naive ``await upload.read()`` would happily slurp a 4 GB body. Reading
    in chunks lets us reject early with 413 before the whole payload lands
    in Python memory and is base64-expanded by ~33%.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_UPLOAD_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {_MAX_UPLOAD_BYTES}-byte upload cap",
            )
        chunks.append(chunk)
    return b"".join(chunks)


_T2V_EXAMPLES = {
    "basic": {
        "summary": "Text-to-video",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "negative_prompt": "blurry, low quality",
            "num_inference_steps": 20,
            "seed": 42,
        },
    },
}

_I2V_EXAMPLES = {
    "single_image": {
        "summary": "I2V with one conditioning image at frame 0",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "num_inference_steps": 12,
            "seed": 42,
            "image_prompts": [
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": 0},
            ],
        },
    },
    "start_end_frame": {
        "summary": "I2V with two conditioning images (start + end)",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "num_inference_steps": 12,
            "seed": 42,
            "image_prompts": [
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": 0},
                {"image": _OPENAPI_IMAGE_PLACEHOLDER, "frame_pos": 80},
            ],
        },
    },
}


async def _submit_video_request(
    request: VideoGenerateRequest,
    service: BaseJobService,
):
    """Shared submit logic for T2V and I2V generation endpoints.

    Both endpoints behave identically once the request is parsed: the only
    difference is the request schema (presence of ``image_prompts`` for I2V).
    Keeping the body in one place avoids drift between the two code paths.
    """
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    try:
        # Synchronous mode: process and return video directly
        if not settings.use_async_video:
            _t0 = _time.time()
            video_file_path = await service.process_request(request)
            _elapsed = round(_time.time() - _t0, 2)

            # Verify the video file exists and is valid
            if not video_file_path or not isinstance(video_file_path, str):
                raise HTTPException(
                    status_code=500,
                    detail="Video generation failed: invalid file path returned",
                )

            if not os.path.exists(video_file_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Video generation failed: file not found at {video_file_path}",
                )

            file_size = os.path.getsize(video_file_path)
            if file_size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Video generation failed: empty file generated",
                )

            return FileResponse(
                video_file_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4",
                    "X-Generation-Time": str(_elapsed),
                },
            )

        # Async mode: create job and return job metadata
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generations")
async def submit_generate_video_request(
    request: Annotated[VideoGenerateRequest, Body(openapi_examples=_T2V_EXAMPLES)],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new text-to-video generation job.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata (async mode)
        FileResponse: Video file directly (sync mode when use_async_video=False)

    Raises:
        HTTPException: If video generation job submission fails.
    """
    return await _submit_video_request(request, service)


@router.post("/generations/i2v")
async def submit_generate_video_i2v_request(
    request: Annotated[VideoI2VGenerateRequest, Body(openapi_examples=_I2V_EXAMPLES)],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new image-to-video generation job (Wan2.2 I2V).

    The request must carry at least one ``image_prompts`` entry.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata (async mode)
        FileResponse: Video file directly (sync mode when use_async_video=False)

    Raises:
        HTTPException: If video generation job submission fails.
    """
    return await _submit_video_request(request, service)


@router.post("/generations/i2v/upload")
async def submit_generate_video_i2v_upload(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    frame_pos: int = Form(0),
    num_inference_steps: Optional[int] = Form(12),
    seed: Optional[int] = Form(None),
    negative_prompt: Optional[str] = Form(None),
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """Generate I2V video from a multipart-uploaded image file.

    Convenience over ``/generations/i2v`` for clients that have an image as a
    file rather than as a base64 string. The uploaded file is read,
    base64-encoded, and wrapped in a single-entry ``image_prompts`` list at
    the requested ``frame_pos``.

    Hard limits:
      * content_type must be ``image/png``, ``image/jpeg``, or ``image/webp``
      * upload body capped at 10 MB (rejected with 413 before RAM allocation)
    """
    _validate_image_content_type(image)
    image_bytes = await _read_capped_upload(image)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    request = VideoI2VGenerateRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image_prompts=[ImagePromptEntry(image=image_b64, frame_pos=frame_pos)],
    )
    return await _submit_video_request(request, service)


@router.get("/generations/{job_id}")
def get_video_metadata(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Fetch the latest metadata for a generated video.

    Returns:
        JSONResponse: Video job object with current status and metadata.

    Raises:
        HTTPException: If video job not found.
    """
    job_data = service.get_job_metadata(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(content=job_data)


@router.get("/jobs")
def get_jobs_metadata(
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Get all jobs metadata

    Returns:
        JSONResponse: Array of video job objects with current status and metadata.
    """
    job_data = service.get_all_jobs_metadata()
    if job_data is None:
        raise HTTPException(status_code=404, detail="Job metadata not found")

    return JSONResponse(content=job_data)


@log_execution_time("Downloading video content", TelemetryEvent.DOWNLOAD_RESULT, None)
@router.get("/generations/{job_id}/download")
def download_video_content(
    job_id: str,
    request: Request,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Download the generated video file as an attachment.

    Returns:
        FileResponse: Streams the full video file (MP4)

    Raises:
        HTTPException: If video not found, not completed, or failed.
    """
    file_path = service.get_job_result_path(job_id)
    if (
        file_path is None
        or not isinstance(file_path, str)
        or not os.path.exists(file_path)
    ):
        raise HTTPException(status_code=404, detail="Video content not available")

    # Create a faststart temp file before serving
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        faststart_path = tmp.name
    try:
        VideoManager.ensure_faststart(file_path, faststart_path)
        serve_path = faststart_path
    except Exception:
        serve_path = file_path

    return FileResponse(
        serve_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path),
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
        },
    )


@router.post("/generations/{job_id}/cancel")
def cancel_video_job(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Permanently cancel a video job and its stored assets.

    Returns:
        JSONResponse: Cancelled video job metadata.

    Raises:
        HTTPException: If video not found.
    """
    status = service.cancel_job(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(content=status)
