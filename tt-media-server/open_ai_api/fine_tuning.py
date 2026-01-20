# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from config.constants import JobTypes
from domain.training_request import TrainingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from model_services.base_service import BaseService
from resolver.service_resolver import service_resolver
from security.api_key_cheker import get_api_key

router = APIRouter()


@router.post("/jobs")
async def submit_fine_tuning_request(
    request: TrainingRequest,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Create a new fine-tuning job.

    Returns:
        JSONResponse: Fine-tuning job object with job ID and metadata.

    Raises:
        HTTPException: If fine tuning job submission fails.
    """
    try:
        job_data = await service.create_job(JobTypes.TRAINING, request)
        return JSONResponse(content=job_data, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail={str(e)})


@router.get("/jobs")
async def list_fine_tuning_jobs(
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    List all fine-tuning jobs.

    Returns:
        JSONResponse: List of fine-tuning jobs.

    Raises:
        HTTPException: If listing jobs fails.
    """
    try:
        jobs = service.get_all_jobs_metadata(JobTypes.TRAINING)
        return JSONResponse(content={"object": "list", "data": jobs, "has_more": False})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_fine_tuning_job_metadata(
    job_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Retrieve details of a fine-tuning job.

    Returns:
        JSONResponse: Fine-tuning job object.

    Raises:
        HTTPException: If fine-tuning job not found.
    """
    job_data = service.get_job_metadata(job_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    return JSONResponse(content=job_data)


@router.delete("/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(
    job_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Cancel a running fine-tuning job.

    Returns:
        JSONResponse: Updated job object with cancelled status.

    Raises:
        HTTPException: If job not found or cannot be cancelled.
    """
    success = service.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400, detail="Job not found or cannot be cancelled"
        )

    return JSONResponse(
        content={
            "id": job_id,
            "object": JobTypes.TRAINING.value,
            "deleted": True,
        }
    )


@router.get("/jobs/{job_id}/checkpoints")
async def list_fine_tuning_checkpoints(
    job_id: str,
    service: BaseService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    List checkpoints for a fine-tuning job.

    Returns:
        JSONResponse: List of model checkpoints.

    Raises:
        HTTPException: If job not found.
    """
    try:
        # TODO: Implement checkpoint retrieval from database
        service.get_job_result_path(job_id)
        return JSONResponse(content={"object": "list", "data": [], "has_more": False})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get checkpoints: {str(e)}"
        )
