# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import os
import tempfile

from config.constants import JobTypes
from config.settings import settings
from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse
from model_services.base_job_service import BaseJobService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.video_manager import VideoManager

router = APIRouter()


@router.post("/generations")
async def submit_generate_video_request(
    request: VideoGenerateRequest,
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

    try:
        # Synchronous mode: process and return video directly
        if not settings.use_async_video:
            video_file_path = await service.process_request(request)

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

            # Verify it's a valid file with size > 0
            file_size = os.path.getsize(video_file_path)
            if file_size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Video generation failed: empty file generated",
                )

            # Create a faststart temp file before serving for better streaming
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                faststart_path = tmp.name

            try:
                VideoManager.ensure_faststart(video_file_path, faststart_path)
                serve_path = faststart_path
            except Exception as e:
                # If faststart fails, serve the original file
                service.logger.warning(
                    f"Failed to create faststart video, serving original: {e}"
                )
                serve_path = video_file_path

            return FileResponse(
                serve_path,
                media_type="video/mp4",
                filename=f"video_{request._task_id}.mp4",
                headers={
                    "Content-Disposition": f"attachment; filename=video_{request._task_id}.mp4"
                },
            )

        # Async mode: create job and return job metadata
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
