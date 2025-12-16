# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Default path, can be overridden in __init__
DEFAULT_DB_PATH = Path("/storage/jobs.db")

class JobDatabase:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        # Ensure the directory exists
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Creates a fresh database connection for each operation.
        Essential for SQLite thread safety in this architecture.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def _init_db(self):
        """Initializes the schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                job_type TEXT NOT NULL,
                job_type_specific_parameters TEXT,
                checkpoint_config TEXT,
                
                hyperparameters TEXT,                 
                metrics TEXT,
                error_message TEXT
            );
        ''')
        conn.commit()
        conn.close()

    # --- Write Operations ---

    def insert_job(self, job_id: str, status: str, hyperparameters: Dict[str, Any], job_type: str, 
                   job_type_specific_parameters: Optional[dict], checkpoint_config: dict):
        """Inserts a new job. Handles JSON serialization of hyperparameters."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Serialize initial data
        hp_json = json.dumps(hyperparameters)
        empty_metrics = json.dumps({"training_loss": [], "validation_loss": []})

        cursor.execute(
            """
            INSERT INTO jobs (id, status, hyperparameters, job_type, job_type_specific_parameters, checkpoint_config, metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, status, hp_json, job_type, json.dumps(job_type_specific_parameters), 
             json.dumps(checkpoint_config), empty_metrics)
        )
        conn.commit()
        conn.close()

    def update_job_status(self, job_id: str, status: str, error_message: Optional[str] = None):
        """Updates just the status (and optionally error message)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if error_message:
            cursor.execute("UPDATE jobs SET status = ?, error_message = ? WHERE id = ?", (status, error_message, job_id))
        else:
            cursor.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
            
        conn.commit()
        conn.close()

    def update_job_metrics(self, job_id: str, step: int, scalar_metrics: Dict[str, float]):
        """
        Appends new metric values to the history.
        
        Args:
            job_id: The ID of the job to update.
            step: The current training step (e.g., 50).
            scalar_metrics: Dict of current values, e.g., {"loss": 0.45, "accuracy": 0.88}
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT metrics FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return # Or raise an error if strictly required

        current_json = row["metrics"]
        
        metrics_store = json.loads(current_json) if current_json else {}

        for key, value in scalar_metrics.items():
            # If this metric key doesn't exist yet, create a list for it
            if key not in metrics_store:
                metrics_store[key] = []
            
            metrics_store[key].append((step, value))

        cursor.execute(
            "UPDATE jobs SET metrics = ? WHERE id = ?", 
            (json.dumps(metrics_store), job_id)
        )
        
        conn.commit()
        conn.close()

    # --- Read Operations ---

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Returns raw dictionaries of all jobs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns a raw dictionary for a specific job, or None."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None