# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import tempfile

from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key
from telemetry.telemetry_client import TelemetryEvent
from utils.decorators import log_execution_time
from utils.video_manager import VideoManager

router = APIRouter()


@router.post("/generations")
async def submit_generate_video_request(
    video_generation_request: VideoGenerateRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new video generation job based on the provided request.

    Returns:
        JSONResponse: Video job object with job ID and initial metadata.

    Raises:
        HTTPException: If video generation submission fails.
    """
    try:
        job_data = await service.create_job("video", video_generation_request)
        return JSONResponse(content=job_data, status_code=202)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generations/{video_id}")
def get_video_metadata(
    video_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Fetch the latest metadata for a generated video.

    Returns:
        JSONResponse: Video job object with current status and metadata.

    Raises:
        HTTPException: If video not found.
    """
    job_data = service.get_job_metadata(video_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return JSONResponse(content=job_data)


@log_execution_time("Downloading video content", TelemetryEvent.DOWNLOAD_RESULT, None)
@router.get("/generations/{video_id}/download")
def download_video_content(
    video_id: str,
    request: Request,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Download the generated video file as an attachment.

    Returns:
        FileResponse: Streams the full video file (MP4) if no Range header is present.
        StreamingResponse: Streams a partial video file (MP4) with HTTP 206 if Range header is present.

    Raises:
        HTTPException: If video not found, not completed, or failed.
        HTTPException: If Range header is invalid (416).
    """
    file_path = service.get_job_result(video_id)
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

    file_size = os.path.getsize(serve_path)
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(
            serve_path,
            media_type="video/mp4",
            filename=os.path.basename(file_path),
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"
            },
        )

    # Parse Range header and stream partial content
    try:
        start, end = VideoManager.parse_range_header(range_header, file_size)
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    chunk_size = end - start + 1
    content_range = f"bytes {start}-{end}/{file_size}"
    headers = {
        "Content-Range": content_range,
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}",
    }
    return StreamingResponse(
        VideoManager.file_iterator(serve_path, start, end),
        status_code=206,
        media_type="video/mp4",
        headers=headers,
    )


@router.delete("/generations/{video_id}")
def delete_video_metadata(
    video_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Permanently delete a video job and its stored assets.

    Returns:
        JSONResponse: Deleted video job metadata.

    Raises:
        HTTPException: If video not found.
    """
    deletion_result = service.delete_job(video_id)
    if not deletion_result:
        raise HTTPException(status_code=404, detail="Video not found")

    return JSONResponse(
        content={
            "id": video_id,
            "object": "video",
            "deleted": True,
        }
    )
