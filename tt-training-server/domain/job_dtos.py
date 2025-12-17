from typing import List, Dict, Literal, Optional

from pydantic import BaseModel

from config.constants import JobStatus, JobType

# Define the input schema for a training job
class TrainingJobRequest(BaseModel):
    model_id: str
    dataset_id: str
    job_type: JobType = JobType.LORA
    hyperparameters: dict
    job_type_specific_parameters: Optional[dict] = None
    checkpoint_config: dict

class JobStatusResponse(BaseModel):
    id: str
    status: JobStatus
    current_metrics: Dict[str, List]

class JobMetricsResponse(BaseModel):
    job_id: str
    all_metrics: Dict[str, List]