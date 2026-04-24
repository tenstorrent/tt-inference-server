# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import io
import os
import zipfile

from config.constants import JobTypes
from config.settings import get_settings
from domain.training_request import TrainingRequest
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.responses import JSONResponse, StreamingResponse
from model_services.base_job_service import BaseJobService
from resolver.service_resolver import service_resolver
from security.api_key_checker import get_api_key
from security.org_id_checker import get_org_id
from utils.build_catalog import build_training_catalog

router = APIRouter()


@router.get("/catalog")
async def get_catalog(api_key: str = Security(get_api_key)):
    """
    List available models, datasets, trainers, optimizers, and clusters for fine-tuning.

    Returns:
        JSONResponse: Full training catalog.
    """
    settings = get_settings()
    num_workers = len(settings.device_ids.replace(" ", "").split("),("))
    catalog = build_training_catalog(
        settings.device,
        settings.device_mesh_shape,
        num_workers,
        settings.model,
    )
    return JSONResponse(content=catalog)


@router.post("/jobs")
async def submit_fine_tuning_request(
    request: TrainingRequest,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
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

    settings = get_settings()
    if request.device_type != settings.device:
        raise HTTPException(
            status_code=400,
            detail=f"Request device '{request.device_type}' does not match server device '{settings.device}'",
        )

    try:
        job_data = await service.create_job(JobTypes.TRAINING, request, org_id=org_id)
        return JSONResponse(content=job_data, status_code=201)
    except Exception as e:
        raise HTTPException(status_code=500, detail={str(e)})


@router.get("/jobs")
async def list_fine_tuning_jobs(
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    List all fine-tuning jobs.

    Returns:
        JSONResponse: List of fine-tuning jobs.

    Raises:
        HTTPException: If listing jobs fails.
    """
    try:
        jobs = service.get_all_jobs_metadata(JobTypes.TRAINING, org_id=org_id)
        return JSONResponse(content={"jobs": jobs})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_fine_tuning_job_metadata(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    Retrieve details of a fine-tuning job.

    Returns:
        JSONResponse: Fine-tuning job object.

    Raises:
        HTTPException: If fine-tuning job not found.
    """
    job_data = service.get_job_metadata(job_id, org_id=org_id)
    if job_data is None:
        raise HTTPException(status_code=404, detail="Fine-tuning job not found")

    return JSONResponse(content=job_data)


@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    Retrieve training metrics for a fine-tuning job.

    Returns:
        JSONResponse: Training metrics for the job.

    Raises:
        HTTPException: If job not found.
    """
    try:
        metrics = service.get_job_metrics(job_id, org_id=org_id)
    except ValueError:
        raise HTTPException(404, "Job not found")
    return JSONResponse(content=metrics)


@router.post("/jobs/{job_id}/cancel")
async def cancel_fine_tuning_job(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    Cancel a running fine-tuning job.

    Returns:
        JSONResponse: Updated job object with cancelled status.

    Raises:
        HTTPException: If job not found or cannot be cancelled.
    """
    status = service.cancel_job(job_id, org_id=org_id)
    if not status:
        raise HTTPException(
            status_code=400, detail="Job not found or cannot be cancelled"
        )

    return JSONResponse(content=status)


@router.get("/jobs/{job_id}/checkpoints")
async def list_fine_tuning_checkpoints(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    List available checkpoints for a fine-tuning job.

    Returns:
        JSONResponse: List of checkpoints for the job.

    Raises:
        HTTPException: If job not found.
    """
    try:
        checkpoints = service.get_job_checkpoints(job_id, org_id=org_id)
    except ValueError:
        raise HTTPException(404, "Job not found")
    return JSONResponse(content={"checkpoints": checkpoints})


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    Retrieve log entries for a fine-tuning job.

    Returns:
        JSONResponse: List of log entries.

    Raises:
        HTTPException: If job not found.
    """
    try:
        logs = service.get_job_logs(job_id, org_id=org_id)
    except ValueError:
        raise HTTPException(404, "Job not found")
    return JSONResponse(content=logs)


@router.get("/jobs/{job_id}/checkpoints/{checkpoint_id}")
async def download_checkpoint(
    job_id: str,
    checkpoint_id: str,
    service: BaseJobService = Depends(service_resolver),
    api_key: str = Security(get_api_key),
    org_id: str = Depends(get_org_id),
):
    """
    Download a checkpoint as a zip archive.

    Returns:
        StreamingResponse: Zip file containing the checkpoint adapter weights.

    Raises:
        HTTPException: If job or checkpoint not found.
    """
    try:
        checkpoint_path = service.get_checkpoint_download_path(
            job_id, checkpoint_id, org_id=org_id
        )
    except ValueError:
        raise HTTPException(404, "Job not found")
    if not checkpoint_path:
        raise HTTPException(
            404, "Checkpoint not found or no longer available for download"
        )
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(checkpoint_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, checkpoint_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=adapter_{checkpoint_id}.zip"
        },
    )
