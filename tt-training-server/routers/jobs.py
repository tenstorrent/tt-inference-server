# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from services.job_service import job_service, JobService
from domain.job_dtos import (
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
    try:
        created_job_id = service.create_job(request)
        return created_job_id
    except Exception as e:
        # Fallback for unexpected database errors
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")

@router.get("/jobs", response_model=List[JobStatusResponse])
async def list_jobs(
    service: JobService = Depends(get_service)
):
    """
    List all current jobs.
    """
    return service.list_jobs()

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Get the status of a specific job.
    """
    job = service.get_job_status(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        
    return job

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
    
    if metrics_data is None:
        raise HTTPException(status_code=404, detail=f"Metrics for job '{job_id}' not found")

    return metrics_data

@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Cancel a running or queued job.
    """
    result = service.cancel_job(job_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        
    return result


