# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import tempfile

from config.constants import JobTypes
from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.video_manager import VideoManager

router = APIRouter()


@router.post("/generations")
async def submit_generate_video_request(
    request: VideoGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new video generation job based on the provided request.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata.

    Raises:
        HTTPException: If video generation job submission fails.
    """
    try:
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")
    try:
        job_data = await service.create_job(JobTypes.VIDEO, request)
        return JSONResponse(content=job_data, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generations/{job_id}")
def get_video_metadata(
    job_id: str,
    service: BaseService = Depends(service_resolver),
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
    service: BaseService = Depends(service_resolver),
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
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Download the generated video file as an attachment.

    Returns:
        FileResponse: Streams the full video file (MP4)

    Raises:
        HTTPException: If video not found, not completed, or failed.
    """
    file_path = service.get_job_result(job_id)
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


@router.delete("/generations/{job_id}")
def cancel_video_job(
    job_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Permanently cancel a video job and its stored assets.

    Returns:
        JSONResponse: Cancelled video job metadata.

    Raises:
        HTTPException: If video not found.
    """
    success = service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Video job not found")

    return JSONResponse(
        content={
            "id": job_id,
            "object": JobTypes.VIDEO.value,
            "deleted": True,
        }
    )
