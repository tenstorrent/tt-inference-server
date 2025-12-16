from typing import List, Dict

from pydantic import BaseModel

# Define the input schema for a training job
class TrainingJobRequest(BaseModel):
    model_id: str
    dataset_id: str
    job_type: str 
    hyperparameters: dict
    job_type_specific_parameters: dict # should this be optional?
    checkpoint_config: dict

class JobStatusResponse(BaseModel):
    id: str
    status: str
    metrics: Dict[str, float]

class JobMetricsResponse(BaseModel):
    job_id: str
    steps: List[int]
    training_loss: List[float]
    validation_loss: List[float]