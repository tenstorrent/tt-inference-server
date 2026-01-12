# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any, Dict, List, Optional
from pathlib import Path

import sqlite3
import json


class JobDatabase:
    """Database interface for persistent job storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Ensure the directory exists
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a fresh database connection for each operation."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def _init_db(self):
        """Initializes the schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                model TEXT NOT NULL,
                request_parameters TEXT,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                completed_at INTEGER,
                error_message TEXT,
                result_path TEXT
            );
        """)
        conn.commit()
        conn.close()

    def insert_job(
        self,
        job_id: str,
        job_type: str,
        model: str,
        request_parameters: dict,
        status: str,
        created_at: int,
    ) -> None:
        """Insert a new job into the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO jobs (id, job_type, model, status, request_parameters, created_at) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, job_type, model, status, json.dumps(request_parameters), created_at),
        )
        conn.commit()
        conn.close()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        completed_at: Optional[int] = None,
        result_path: Optional[str] = None,
        error_message: Optional[dict] = None,
    ) -> None:
        """Update job status and optional fields."""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = ["status = ?"]
        params = [status]

        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at)

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(json.dumps(error_message))

        if result_path is not None:
            updates.append("result_path = ?")
            params.append(result_path)

        query = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
        params.append(job_id)

        cursor.execute(query, tuple(params))
        conn.commit()
        conn.close()

    def delete_job(self, job_id: str) -> None:
        """Delete a job from the database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        conn.commit()
        conn.close()

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:   
        """Retrieve a specific job from the database by its ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            job_dict = dict(row)

            if job_dict.get("request_parameters"):
                job_dict["request_parameters"] = json.loads(job_dict["request_parameters"])
                    
            if job_dict.get("error_message"):
                job_dict["error_message"] = json.loads(job_dict["error_message"])
                    
            return job_dict
            
        return None

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retrieve all jobs from the database."""
        conn = self._get_connection()
        # Using Row factory allows accessing columns by name
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()

        jobs = []
        for row in rows:
            # Convert row to dict and parse the JSON strings back to Python objects
            job_dict = dict(row)
            if job_dict.get("request_parameters"):
                job_dict["request_parameters"] = json.loads(job_dict["request_parameters"])
            if job_dict.get("error_message"):
                job_dict["error_message"] = json.loads(job_dict["error_message"])
            jobs.append(job_dict)

        conn.close()
        return jobs
