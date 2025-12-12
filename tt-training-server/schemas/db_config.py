import sqlite3
from pathlib import Path

# Configuration
DB_PATH = Path("/storage/jobs.db")

def get_db_connection():
    """Returns a connected SQLite object with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def init_db():
    """Initializes the SQLite table if it doesn't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = get_db_connection()
    # The table creation logic should remain here
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            base_model_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL,
            job_type TEXT NOT NULL,
            config_json TEXT NOT NULL,       -- Stores full request body
            metrics_json TEXT DEFAULT '{}',  -- Stores {loss: 0.5, ...}
            trained_tokens INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Note: You can call init_db() here to ensure the file 
# initializes the database as soon as it's imported.
init_db()