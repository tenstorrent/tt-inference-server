# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.video_generate_request import VideoGenerateRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

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


@router.get("/generations/{video_id}/content")
def get_video_content(
    video_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Get the filename of the generated video.

    Returns:
        JSONResponse: Object containing the video filename (MP4).

    Raises:
        HTTPException: If video not found, not completed, or failed.
    """
    job_result = service.get_job_result(video_id)
    if job_result is None:
        raise HTTPException(status_code=404, detail="Video content not available")

    return JSONResponse(content={"filename": f"{job_result}.mp4"})


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
