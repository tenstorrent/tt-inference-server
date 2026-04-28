# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import base64
import os
import tempfile
import time
from typing import Annotated, Optional

from config.constants import JobTypes
from config.settings import settings
from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from model_services.base_job_service import BaseJobService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.logger import TTLogger
from utils.video_manager import VideoManager

logger = TTLogger()

router = APIRouter()


_VIDEO_EXAMPLES = {
    "image_url": {
        "summary": "Image-to-video via URL",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "image": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80",
            "num_inference_steps": 12,
            "seed": 42,
        },
    },
    "image_frames": {
        "summary": "Image-to-video via image_frames",
        "value": {
            "prompt": "A serene mountain landscape with flowing water",
            "image_frames": [
                {
                    "image": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80",
                    "frame_pos": 0,
                }
            ],
            "num_inference_steps": 12,
            "seed": 42,
        },
    },
}


@router.post("/generations")
async def submit_generate_video_request(
    request: Annotated[VideoGenerateRequest, Body(openapi_examples=_VIDEO_EXAMPLES)],
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new video generation job based on the provided request.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata (async mode)
        FileResponse: Video file directly (sync mode when use_async_video=False)

    Raises:
        HTTPException: If video generation job submission fails.
    """
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    logger.info(f"[video] mode={'sync' if not settings.use_async_video else 'async'}  task={request._task_id}")

    try:
        # Synchronous mode: process and return video directly
        if not settings.use_async_video:
            t0 = time.perf_counter()
            video_file_path = await service.process_request(request)
            t1 = time.perf_counter()

            if not video_file_path or not isinstance(video_file_path, str):
                raise HTTPException(status_code=500, detail="Video generation failed: invalid file path returned")
            if not os.path.exists(video_file_path):
                raise HTTPException(status_code=500, detail=f"Video generation failed: file not found at {video_file_path}")
            file_size = os.path.getsize(video_file_path)
            if file_size == 0:
                raise HTTPException(status_code=500, detail="Video generation failed: empty file generated")

            logger.info(
                f"[video] process_request={t1-t0:.3f}s  file_size={file_size/1024:.1f}KB  task={request._task_id}"
            )

            return FileResponse(
                video_file_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={"Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4"},
            )

        # Async mode: create job and return job metadata
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generations/upload")
async def submit_generate_video_upload(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    num_inference_steps: Optional[int] = Form(12),
    seed: Optional[int] = Form(None),
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Generate a video from an uploaded image file.
    Accepts multipart/form-data with an image file upload.
    """
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode()

    request = VideoGenerateRequest(
        prompt=prompt,
        image=image_b64,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    try:
        if not settings.use_async_video:
            video_file_path = await service.process_request(request)
            if not video_file_path or not isinstance(video_file_path, str):
                raise HTTPException(status_code=500, detail="Video generation failed: invalid file path returned")
            if not os.path.exists(video_file_path):
                raise HTTPException(status_code=500, detail=f"Video generation failed: file not found at {video_file_path}")
            if os.path.getsize(video_file_path) == 0:
                raise HTTPException(status_code=500, detail="Video generation failed: empty file generated")
            return FileResponse(
                video_file_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={"Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4"},
            )
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generations/start-end-frame")
async def generate_video_start_end_frame(
    prompt: str = Form(...),
    start_frame: UploadFile = File(..., description="Image for the first frame (frame_pos=0)"),
    end_frame: UploadFile = File(..., description="Image for the last frame (frame_pos=-1)"),
    num_inference_steps: Optional[int] = Form(12),
    seed: Optional[int] = Form(None),
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Generate a video conditioned on start and end frame images.
    Accepts multipart/form-data with two image file uploads.
    The end frame is placed at the last frame position (num_frames - 1).
    """
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")

    start_bytes = await start_frame.read()
    end_bytes = await end_frame.read()

    from domain.video_generate_request import ImageFrame

    request = VideoGenerateRequest(
        prompt=prompt,
        image_frames=[
            ImageFrame(image=base64.b64encode(start_bytes).decode(), frame_pos=0),
            ImageFrame(image=base64.b64encode(end_bytes).decode(), frame_pos=-1),
        ],
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    try:
        if not settings.use_async_video:
            video_file_path = await service.process_request(request)
            if not video_file_path or not isinstance(video_file_path, str):
                raise HTTPException(status_code=500, detail="Video generation failed: invalid file path returned")
            if not os.path.exists(video_file_path):
                raise HTTPException(status_code=500, detail=f"Video generation failed: file not found at {video_file_path}")
            if os.path.getsize(video_file_path) == 0:
                raise HTTPException(status_code=500, detail="Video generation failed: empty file generated")
            return FileResponse(
                video_file_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={"Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4"},
            )
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
