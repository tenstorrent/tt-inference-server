# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel

from schemas.fine_tuning import CreateJobRequest, JobStatusResponse
from services.job_service import job_service, JobService

router = APIRouter()

# Define the input schema for a training job
class TrainingJobRequest(BaseModel):
    model_id: str
    dataset_id: str
    hyperparameters: dict

class JobStatusResponse(BaseModel):
    id: str
    status: str
    trained_tokens: int
    metrics: Dict[str, float]
    finished_at: Optional[datetime]

# Dependency Injection helper (optional, allows for easier mocking in tests)
def get_service() -> JobService:
    return job_service


@router.post("/jobs", response_model=JobStatusResponse, status_code=201)
async def create_training_job(
    request: TrainingJobRequest, 
    background_tasks: BackgroundTasks,
    service: JobService = Depends(get_service)
):
    """
    Create a new fine-tuning job.
    """
    # 1. Delegate creation to service
    job_response = service.create_job(request)
    
    return job_response


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Get the status of a specific job.
    """
    return service.get_job_status(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(
    job_id: str,
    service: JobService = Depends(get_service)
):
    """
    Cancel a running or queued job.
    """
    return service.cancel_job(job_id)



# @router.post("/", status_code=202)
# async def create_fine_tuning_job(job: TrainingJobRequest, background_tasks: BackgroundTasks):
#     """
#     Starts a fine-tuning job. 
#     Returns 202 Accepted because training is async.
#     """
#     # Logic to validate inputs...
    
#     job_id = "ft-job-123" # Generate a real ID
    
#     # Send to the scheduler (imported from common/scheduling as discussed)
#     # background_tasks.add_task(scheduler.submit, job_id, job)
    
#     return {"id": job_id, "status": "queued"}

# @router.get("/{job_id}")
# async def get_fine_tuning_job(job_id: str):
#     """
#     Check the status of a specific job.
#     """
#     return {"id": job_id, "status": "running", "progress": "45%"}