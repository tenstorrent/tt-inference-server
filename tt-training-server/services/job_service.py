# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import uuid
import json
from typing import List, Optional

# Import the Database Class
from db.job_database import JobDatabase

# Import your Pydantic Models
from domain.job_dtos import (
    TrainingJobRequest, 
    JobStatusResponse, 
    JobMetricsResponse
)

def schedule_job(job_id: str):
    """
    Placeholder function to schedule the job for processing.
    In a real system, this would enqueue the job in a task queue or similar.
    """
    pass

class JobService:
    def __init__(self, db: JobDatabase):
            self.db = db

    # --- Public Methods (API) ---

    def create_job(self, request: TrainingJobRequest) -> str:
        job_id = str(uuid.uuid4())
        
        # Persist initial state
        self.db.insert_job(
            job_id=job_id,
            status="QUEUED",
            hyperparameters=request.hyperparameters,
            job_type=request.job_type,
            job_type_specific_parameters=request.job_type_specific_parameters,
            checkpoint_config=request.checkpoint_config
        )

        schedule_job(job_id)

        return job_id

    def list_jobs(self) -> List[JobStatusResponse]:
        raw_jobs = self.db.get_all_jobs()
        return [self._map_db_row_to_response(row) for row in raw_jobs]

    def get_job_status(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Returns the job status, or None if the job does not exist.
        """
        row = self.db.get_job(job_id)
        if not row:
            return None
            
        return self._map_db_row_to_response(row)

    def get_job_metrics(self, job_id: str) -> Optional[JobMetricsResponse]:
        """
        Returns job metrics, or None if the job does not exist.
        """
        row = self.db.get_job(job_id)
        if not row:
            return None

        # The DB returns JSON strings; we must deserialize them here.
        return JobMetricsResponse(
            job_id=row["id"],
            all_metrics=json.loads(row["metrics"]),
        )

    def cancel_job(self, job_id: str) -> Optional[JobStatusResponse]:
        """
        Cancels a job if it exists. Returns the updated status, 
        or None if the job does not exist.
        """
        row = self.db.get_job(job_id)
        if not row:
            return None

        current_status = row["status"]
        
        # Only cancel if active
        if current_status not in ["COMPLETED", "FAILED", "CANCELLED"]:
            self.db.update_job_status(job_id, "CANCELLING")
            
        # We recursively call get_job_status to return the fresh state
        return self.get_job_status(job_id)

    # --- Internal Helpers ---

    def _map_db_row_to_response(self, row: dict) -> JobStatusResponse:
            """Converts raw DB dictionary to Pydantic Response Model"""
            
            raw_metrics = json.loads(row["metrics"]) if row["metrics"] else {}

            current_metrics = {}

            for metric_name, metric_values in raw_metrics.items():
                if metric_values:
                    last_entry = metric_values[-1] # Get the latest tuple
                    last_step = last_entry[0]
                    last_value = last_entry[1]

                    current_metrics[metric_name] = last_value

            return JobStatusResponse(
                id=row["id"],
                status=row["status"], # Pydantic validates this against your Enum
                current_metrics=current_metrics
            )

# Singleton Instance
job_service = JobService(db=JobDatabase())