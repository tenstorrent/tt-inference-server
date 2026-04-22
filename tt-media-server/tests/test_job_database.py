# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# test_job_database.py

from sqlite3 import IntegrityError

import pytest
from utils.job_database import JobDatabase


@pytest.fixture
def db(tmp_path):
    return JobDatabase(db_path=tmp_path / "test.db")


@pytest.fixture
def db_with_job(db):
    """DB with a single pre-inserted training job."""
    db.insert_job("job-1", "training", "model-1", {}, "in_progress", 1000)
    return db


class TestInsertJobOrgId:
    def test_insert_job_with_org_id(self, db):
        db.insert_job(
            "job-org", "training", "model-1", {}, "in_progress", 1000, org_id="org-abc"
        )
        result = db.get_job_by_id("job-org")
        assert result["org_id"] == "org-abc"

    def test_insert_job_without_org_id(self, db):
        db.insert_job("job-no-org", "video", "model-1", {}, "queued", 1000)
        result = db.get_job_by_id("job-no-org")
        assert result["org_id"] is None


class TestInsertCheckpoint:
    def test_insert_and_retrieve_single_checkpoint(self, db_with_job):
        db_with_job.insert_checkpoint(
            job_id="job-1",
            checkpoint_id="ckpt-step-100",
            step=100,
            epoch=1,
            metrics={"train_loss": 0.42},
            created_at=1001.5,
        )
        result = db_with_job.get_checkpoints("job-1")
        assert len(result) == 1
        assert result[0] == {
            "id": "ckpt-step-100",
            "step": 100,
            "epoch": 1,
            "metrics": {"train_loss": 0.42},
            "created_at": 1001.5,
        }

    def test_insert_multiple_checkpoints_ordered_by_step(self, db_with_job):
        # Insert in reverse order
        db_with_job.insert_checkpoint("job-1", "ckpt-step-200", 200, 2, {}, 1002.0)
        db_with_job.insert_checkpoint("job-1", "ckpt-step-100", 100, 1, {}, 1001.0)
        db_with_job.insert_checkpoint("job-1", "ckpt-step-300", 300, 3, {}, 1003.0)

        result = db_with_job.get_checkpoints("job-1")
        assert len(result) == 3
        assert [r["step"] for r in result] == [100, 200, 300]

    def test_insert_duplicate_checkpoint_raises_integrity_error(self, db_with_job):
        db_with_job.insert_checkpoint("job-1", "ckpt-step-100", 100, 1, {}, 1001.0)
        with pytest.raises(IntegrityError):
            db_with_job.insert_checkpoint("job-1", "ckpt-step-100", 100, 1, {}, 1002.0)

    def test_get_checkpoints_empty_for_job_with_no_checkpoints(self, db_with_job):
        assert db_with_job.get_checkpoints("job-1") == []

    def test_get_checkpoints_returns_empty_for_nonexistent_job(self, db):
        assert db.get_checkpoints("no-such-job") == []

    def test_checkpoints_cascade_deleted_with_job(self, db_with_job):
        db_with_job.insert_checkpoint("job-1", "ckpt-1", 100, 1, {}, 1001.0)
        assert len(db_with_job.get_checkpoints("job-1")) == 1
        db_with_job.delete_job("job-1")
        assert db_with_job.get_checkpoints("job-1") == []


class TestInsertLog:
    def test_insert_and_retrieve_single_log(self, db_with_job):
        db_with_job.insert_log(
            job_id="job-1",
            log_index=0,
            timestamp="2025-01-01T00:00:00",
            log_type="info",
            step=10,
            message="Training started",
        )
        result = db_with_job.get_logs("job-1")
        assert len(result) == 1
        assert result[0] == {
            "id": 0,
            "timestamp": "2025-01-01T00:00:00",
            "type": "info",
            "step": 10,
            "message": "Training started",
        }

    def test_insert_multiple_logs_ordered_by_log_index(self, db_with_job):
        db_with_job.insert_log("job-1", 2, "ts3", "info", 30, "Step 30")
        db_with_job.insert_log("job-1", 0, "ts1", "info", 10, "Step 10")
        db_with_job.insert_log("job-1", 1, "ts2", "warning", 20, "Step 20")

        result = db_with_job.get_logs("job-1")
        assert len(result) == 3
        assert [r["id"] for r in result] == [0, 1, 2]
        assert [r["message"] for r in result] == ["Step 10", "Step 20", "Step 30"]

    def test_insert_duplicate_log_raises_integrity_error(self, db_with_job):
        db_with_job.insert_log("job-1", 0, "ts", "info", 10, "msg")
        with pytest.raises(IntegrityError):
            db_with_job.insert_log("job-1", 0, "ts2", "info", 20, "msg2")

    def test_get_logs_empty_for_job_with_no_logs(self, db_with_job):
        assert db_with_job.get_logs("job-1") == []

    def test_logs_cascade_deleted_with_job(self, db_with_job):
        db_with_job.insert_log("job-1", 0, "ts", "info", 10, "msg")
        assert len(db_with_job.get_logs("job-1")) == 1
        db_with_job.delete_job("job-1")
        assert db_with_job.get_logs("job-1") == []
