# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any, Dict, List, Optional


class JobDatabase:
    """Database interface for persistent job storage."""

    def __init__(self):
        pass

    def insert_job(
        self,
        job_id: str,
        object: str,
        model: str,
        status: str,
        created_at: int,
    ) -> None:
        """Insert a new job into the database."""
        pass

    def update_job_status(
        self,
        job_id: str,
        status: str,
        completed_at: Optional[int] = None,
        result: Optional[Any] = None,
        error: Optional[Dict[str, str]] = None,
    ) -> None:
        """Update job status and optional fields."""
        pass

    def delete_job(self, job_id: str) -> None:
        """Delete a job from the database."""
        pass

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retrieve all jobs from the database."""
        pass
