# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import uuid
import threading
import time
import random
import json
from typing import List, Optional

# Import the Database Class
from db import JobDatabase

# Import your Pydantic Models
from domain.job_models import (
    TrainingJobRequest, 
    JobStatusResponse, 
    JobMetricsResponse
)

class JobService:
    def __init__(self):
        # Initialize the Data Access Layer
        self.db = JobDatabase()

    # --- Public Methods (API) ---

    def create_job(self, request: TrainingJobRequest) -> str:
        job_id = str(uuid.uuid4())
        
        # Persist initial state
        self.db.insert_job(
            job_id=job_id,
            status="QUEUED",
            hyperparameters=request.hyperparameters
        )

        # TODO: schedule job

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
            steps=json.loads(row["steps"]),
            training_loss=json.loads(row["training_loss"]),
            validation_loss=json.loads(row["validation_loss"])
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
        steps = json.loads(row["steps"])
        t_loss = json.loads(row["training_loss"])
        v_loss = json.loads(row["validation_loss"])

        # Calculate summary metrics
        metrics = {"steps_completed": len(steps)}
        if t_loss:
            metrics["last_loss"] = t_loss[-1]
        if v_loss:
            metrics["last_val_loss"] = v_loss[-1]

        return JobStatusResponse(
            id=row["id"],
            status=row["status"],
            metrics=metrics
        )

    # --- Simulation Worker ---

    def _run_job_logic(self, job_id: str):
        """
        Background worker that interacts with the DB via self.db
        """
        try:
            # 1. Update to RUNNING
            time.sleep(2) 
            self.db.update_job_status(job_id, "RUNNING")

            # 2. Initialize Simulation
            total_steps = 20
            start_loss = 2.5
            
            # Fetch current state to ensure we have fresh lists
            row = self.db.get_job(job_id)
            if not row: return # Safety check

            steps_list = json.loads(row["steps"])
            t_loss_list = json.loads(row["training_loss"])
            v_loss_list = json.loads(row["validation_loss"])
            current_step = 0

            # 3. Training Loop
            for i in range(total_steps):
                # Check for cancellation
                fresh_row = self.db.get_job(job_id)
                if not fresh_row or fresh_row["status"] == "CANCELLING":
                    if fresh_row:
                        self.db.update_job_status(job_id, "CANCELLED")
                    return

                # Simulate Compute
                time.sleep(0.5)
                
                # Generate Data
                current_step += 1
                decay = 0.90
                loss = start_loss * (decay ** current_step) + random.uniform(-0.05, 0.05)
                val_loss = loss + 0.15

                steps_list.append(current_step)
                t_loss_list.append(round(loss, 4))
                v_loss_list.append(round(val_loss, 4))

                # Save Progress
                self.db.update_job_metrics(job_id, steps_list, t_loss_list, v_loss_list)

            # 4. Complete
            self.db.update_job_status(job_id, "COMPLETED")

        except Exception as e:
            print(f"Job {job_id} failed: {e}")
            self.db.update_job_status(job_id, "FAILED", error_message=str(e))

# Singleton Instance
job_service = JobService()