# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from services.job_service import job_service, JobService
from domain.job_models import (
    TrainingJobRequest, 
    JobStatusResponse, 
    JobMetricsResponse
)

router = APIRouter()

# Dependency Injection helper (optional, allows for easier mocking in tests)
def get_service() -> JobService:
    return job_service


@router.post("/jobs", response_model=str, status_code=201)
async def create_training_job(
    request: TrainingJobRequest, 
    service: JobService = Depends(get_service)
):
    """
    Create a new fine-tuning job.
    """
    created_job_id = service.create_job(request)
    
    return created_job_id

@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    service: JobService = Depends(get_service)
):
    """
    List all current jobs.
    """
    # Assumption: service.list_jobs() returns a list of job objects 
    # compatible with JobStatusResponse
    return service.list_jobs()

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Get the status of a specific job.
    """
    return service.get_job_status(job_id)

@router.get("/jobs/{job_id}/events", response_model=JobMetricsResponse)
async def get_job_metrics(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Returns a history of events (steps and loss curves) to track progress.
    """
    # You will need to implement get_job_metrics(job_id) in your JobService
    metrics_data = service.get_job_metrics(job_id)
    
    if not metrics_data:
        raise HTTPException(status_code=404, detail="Job metrics not found")

    return metrics_data

@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Cancel a running or queued job.
    """
    return service.cancel_job(job_id) # we need to check what to return



