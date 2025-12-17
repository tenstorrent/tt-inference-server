# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Default path, can be overridden in __init__
DEFAULT_DB_PATH = Path("./temp_storage/jobs.db")

class JobDatabase:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        # Ensure the directory exists
        if self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.clear_all_data()

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
                       
                dataset_id TEXT,
                base_model_id TEXT,
                tag TEXT,
                
                job_type TEXT NOT NULL,
                job_type_specific_parameters TEXT,
                checkpoint_config TEXT,
                
                hyperparameters TEXT,                 
                error_message TEXT
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                job_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                value FLOAT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                PRIMARY KEY (job_id, step, metric_name), 
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );
        ''')
        conn.commit()
        conn.close()

    def clear_all_data(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM jobs;') 
        # Note: metrics are deleted automatically due to CASCADE
        conn.commit()
        conn.close()

    # --- Write Operations ---

    def insert_job(self, job_id: str, status: str, hyperparameters: Dict[str, Any], job_type: str, 
                   job_type_specific_parameters: Optional[dict], checkpoint_config: dict):
        """
        Inserts a new job. 
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO jobs (id, status, hyperparameters, job_type, job_type_specific_parameters, checkpoint_config)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, status, json.dumps(hyperparameters), job_type, 
             json.dumps(job_type_specific_parameters), json.dumps(checkpoint_config))
        )
        conn.commit()
        conn.close()
    
    def insert_metric_value(self, job_id: str, step: int, scalar_metrics: Dict[str, float]):
        """
        Inserts multiple metric values for a single step efficiently.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Prepare list of tuples for bulk insertion
        rows = [
            (job_id, step, k, v) 
            for k, v in scalar_metrics.items()
        ]

        try:
            cursor.executemany('''
                INSERT INTO metrics (job_id, step, metric_name, value)
                VALUES (?, ?, ?, ?)
            ''', rows)
            
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.rollback() # Important: undo changes if one fails
            print(f"ERROR: Attempted to overwrite metrics for Job {job_id} at Step {step}")
            # Re-raise the error so your API knows to return a 409 Conflict
            raise ValueError(f"Duplicate metrics detected for step {step}") from e
        finally:
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

    # --- Read Operations ---

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """
        Returns all jobs WITHOUT metrics data.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            d = dict(row)
            # Parse JSON fields back to dicts
            d['hyperparameters'] = json.loads(d['hyperparameters']) if d['hyperparameters'] else {}
            d['job_type_specific_parameters'] = json.loads(d['job_type_specific_parameters']) if d['job_type_specific_parameters'] else {}
            d['checkpoint_config'] = json.loads(d['checkpoint_config']) if d['checkpoint_config'] else {}
            results.append(d)
            
        return results

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns job details + 'current_metrics' (the LATEST value for each metric).
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. Get Job Info
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        job_data = dict(row)
        
        # 2. Get LATEST metrics (snapshot)
        # We group by metric_name and find the row with the max step
        cursor.execute("""
            SELECT metric_name, value, step
            FROM metrics
            WHERE job_id = ?
            GROUP BY metric_name
            HAVING step = MAX(step)
        """, (job_id,))
        
        metric_rows = cursor.fetchall()
        
        # Transform into simple dict: {"loss": 0.5, "accuracy": 0.9}
        current_metrics = {(r["metric_name"], r["step"]): r["value"] for r in metric_rows}
        
        conn.close()

        # Parse JSON fields
        job_data['hyperparameters'] = json.loads(job_data['hyperparameters']) if job_data['hyperparameters'] else {}
        job_data['job_type_specific_parameters'] = json.loads(job_data['job_type_specific_parameters']) if job_data['job_type_specific_parameters'] else {}
        job_data['checkpoint_config'] = json.loads(job_data['checkpoint_config']) if job_data['checkpoint_config'] else {}
        
        # Attach snapshot metrics
        job_data['current_metrics'] = current_metrics
        
        return job_data

    def get_job_metrics(self, job_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns FULL history of metrics for graphing.
        Format:
        {
           "training_loss": [ {"step": 1, "value": 0.5, "timestamp": ...}, ... ],
           "accuracy": [ ... ]
        }
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT step, metric_name, value, timestamp 
            FROM metrics 
            WHERE job_id = ? 
            ORDER BY step ASC
        """, (job_id,))
        
        rows = cursor.fetchall()
        conn.close()

        # Organize by metric name
        history = {}
        for r in rows:
            name = r["metric_name"]
            if name not in history:
                history[name] = []
            
            history[name].append({
                "step": r["step"],
                "value": r["value"],
                "timestamp": r["timestamp"]
            })
            
        return history