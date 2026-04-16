# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import Any, Dict, List, Optional
from pathlib import Path

import sqlite3
import json
from contextlib import contextmanager


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
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _get_cursor(self, commit: bool = True):
        conn = self._get_connection()
        try:
            yield conn.cursor()
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initializes the schema."""
        with self._get_cursor(commit=True) as cursor:
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
                    result_path TEXT,
                    org_id TEXT
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    job_id TEXT NOT NULL,
                    global_step INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value FLOAT NOT NULL,
                    learning_rate FLOAT,
                    timestamp REAL NOT NULL,
                    
                    PRIMARY KEY (job_id, global_step, metric_name), 
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    job_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    metrics TEXT,
                    created_at REAL NOT NULL,

                    PRIMARY KEY (job_id, checkpoint_id),
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    job_id TEXT NOT NULL,
                    log_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    type TEXT NOT NULL,
                    step INTEGER,
                    message TEXT NOT NULL,

                    PRIMARY KEY (job_id, log_index),
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
            """)

    def insert_job(
        self,
        job_id: str,
        job_type: str,
        model: str,
        request_parameters: dict,
        status: str,
        created_at: int,
        org_id: Optional[str] = None,
    ) -> None:
        """Insert a new job into the database."""
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO jobs (id, job_type, model, status, request_parameters, created_at, org_id) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_type,
                    model,
                    status,
                    json.dumps(request_parameters),
                    created_at,
                    org_id,
                ),
            )

    def update_job_status(
        self,
        job_id: str,
        status: str,
        completed_at: Optional[int] = None,
        result_path: Optional[str] = None,
        error_message: Optional[dict[str, str]] = None,
    ) -> None:
        """Update job status and optional fields."""
        with self._get_cursor(commit=True) as cursor:
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

    def update_result_path(self, job_id: str, result_path: str) -> None:
        """Update only the result_path for a job."""
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(
                "UPDATE jobs SET result_path = ? WHERE id = ?", (result_path, job_id)
            )
            print(f"Rows affected: {cursor.rowcount}")

    def delete_job(self, job_id: str) -> None:
        """Delete a job from the database."""
        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific job from the database by its ID."""
        with self._get_cursor(commit=False) as cursor:
            cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()

        if row:
            job_dict = dict(row)

            if job_dict.get("request_parameters"):
                job_dict["request_parameters"] = json.loads(
                    job_dict["request_parameters"]
                )

            if job_dict.get("error_message"):
                job_dict["error_message"] = json.loads(job_dict["error_message"])

            return job_dict

        return None

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Retrieve all jobs from the database."""
        with self._get_cursor(commit=False) as cursor:
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = cursor.fetchall()

        jobs = []
        for row in rows:
            # Convert row to dict and parse the JSON strings back to Python objects
            job_dict = dict(row)
            if job_dict.get("request_parameters"):
                job_dict["request_parameters"] = json.loads(
                    job_dict["request_parameters"]
                )
            if job_dict.get("error_message"):
                job_dict["error_message"] = json.loads(job_dict["error_message"])
            jobs.append(job_dict)

        return jobs

    # ------------- METRICS -------------
    def insert_metric(
        self,
        job_id: str,
        global_step: int,
        epoch: int,
        metric_name: str,
        value: float,
        timestamp: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Insert a new metric into the database."""
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO metrics (job_id, global_step, epoch, metric_name, value, learning_rate, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    global_step,
                    epoch,
                    metric_name,
                    value,
                    learning_rate,
                    timestamp,
                ),
            )

    def get_metrics_flat(self, job_id: str) -> List[Dict[str, Any]]:
        """Returns metrics as a flat list."""
        with self._get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT global_step, epoch, metric_name, value, learning_rate, timestamp FROM metrics WHERE job_id = ? ORDER BY metric_name ASC, global_step ASC",
                (job_id,),
            )
            return [
                {
                    "global_step": r["global_step"],
                    "epoch": r["epoch"],
                    "metric_name": r["metric_name"],
                    "value": r["value"],
                    "learning_rate": r["learning_rate"],
                    "timestamp": r["timestamp"],
                }
                for r in cursor.fetchall()
            ]

    # ------------- CHECKPOINTS -------------
    def insert_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str,
        step: int,
        epoch: int,
        metrics: dict,
        created_at: float,
    ) -> None:
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO checkpoints (job_id, checkpoint_id, step, epoch, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, checkpoint_id, step, epoch, json.dumps(metrics), created_at),
            )

    def get_checkpoints(self, job_id: str) -> List[Dict[str, Any]]:
        with self._get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT checkpoint_id, step, epoch, metrics, created_at FROM checkpoints WHERE job_id = ? ORDER BY step ASC",
                (job_id,),
            )
            return [
                {
                    "id": r["checkpoint_id"],
                    "step": r["step"],
                    "epoch": r["epoch"],
                    "metrics": json.loads(r["metrics"]) if r["metrics"] else {},
                    "created_at": r["created_at"],
                }
                for r in cursor.fetchall()
            ]

    # ------------- LOGS -------------
    def insert_log(
        self,
        job_id: str,
        log_index: int,
        timestamp: str,
        log_type: str,
        step: int,
        message: str,
    ) -> None:
        with self._get_cursor(commit=True) as cursor:
            cursor.execute(
                "INSERT INTO logs (job_id, log_index, timestamp, type, step, message) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, log_index, timestamp, log_type, step, message),
            )

    def get_logs(self, job_id: str) -> List[Dict[str, Any]]:
        with self._get_cursor(commit=False) as cursor:
            cursor.execute(
                "SELECT log_index, timestamp, type, step, message FROM logs WHERE job_id = ? ORDER BY log_index ASC",
                (job_id,),
            )
            return [
                {
                    "id": r["log_index"],
                    "timestamp": r["timestamp"],
                    "type": r["type"],
                    "step": r["step"],
                    "message": r["message"],
                }
                for r in cursor.fetchall()
            ]
