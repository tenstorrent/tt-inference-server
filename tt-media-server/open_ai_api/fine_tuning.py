# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from config.constants import JobTypes
from domain.training_request import TrainingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse
from model_services.base_job_service import BaseJobService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from utils.dataset_loaders.dataset_resolver import AVAILABLE_DATASET_LOADERS
from config.constants import MODEL_RUNNER_TO_MODEL_NAMES_MAP, MODEL_SERVICE_RUNNER_MAP, ModelServices

router = APIRouter()

@router.get("/datasets")
async def list_available_datasets(
    api_key: str = Security(get_api_key)
):
    """
    List all available datasets.

    Returns:
        JSONResponse: List of available datasets.
    """
    datasets = [loader.value for loader in AVAILABLE_DATASET_LOADERS.keys()]
    return JSONResponse(content={"data": datasets})

@router.get("/models")
async def list_training_models(
    api_key: str = Security(get_api_key)
):
    """
    List all available training models.

    Returns:
        JSONResponse: List of available training models.
    """
    runners = MODEL_SERVICE_RUNNER_MAP.get(ModelServices.TRAINING, set())
    models = []
    for runner in runners:
        names = MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(runner, set())
        models.extend(n.value for n in names)
    return JSONResponse(content={"data": models})

@router.post("/jobs")
async def submit_fine_tuning_request(
    request: TrainingRequest,
    service: BaseJobService = Depends(service_resolver),
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
        service.scheduler.check_is_model_ready()
    except Exception:
        raise HTTPException(status_code=405, detail="Model is not ready")
    try:
        job_data = await service.create_job(JobTypes.TRAINING, request)
        return JSONResponse(content=job_data, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail={str(e)})


@router.get("/jobs")
async def list_fine_tuning_jobs(
    service: BaseJobService = Depends(service_resolver),
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
    service: BaseJobService = Depends(service_resolver),
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


@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    after: int = 0,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    job_data = service.get_job_metadata(job_id)
    if not job_data:
        raise HTTPException(404, "Job not found")

    metrics = service.get_job_metrics(job_id, after)
    is_final = job_data.get("status") in ("completed", "failed", "cancelled")

    return JSONResponse(
        content={
            "data": metrics,
            "next_after": after + len(metrics),
            "is_final": is_final,
        }
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
):
    """
    Cancel a running fine-tuning job.

    Returns:
        JSONResponse: Updated job object with cancelled status.

    Raises:
        HTTPException: If job not found or cannot be cancelled.
    """
    status = service.cancel_job(job_id)
    if not status:
        raise HTTPException(
            status_code=400, detail="Job not found or cannot be cancelled"
        )

    return JSONResponse(content=status)
