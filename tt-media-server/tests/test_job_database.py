import pytest
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock
from utils.job_manager import JobManager, JobStatus, JobTypes

class TestJobManagerDatabaseIntegration:
    """Integration tests using a real temporary database"""

    @pytest.fixture
    async def manager(self):
        """Setup a JobManager with a real temporary SQLite database."""
        # 1. Reset singleton
        import utils.job_manager
        utils.job_manager._job_manager_instance = None

        # 2. Create a temporary file for the database
        fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        
        # 3. Patch settings and the DEFAULT_DB_PATH
        # We patch the path so the real JobDatabase() uses our temp file
        with patch("utils.job_manager.get_settings") as mock_settings, \
             patch("utils.job_database.DEFAULT_DB_PATH", Path(temp_db_path)):
            
            mock_settings.return_value.enable_job_persistence = True
            mock_settings.return_value.max_jobs = 10
            mock_settings.return_value.job_cleanup_interval_seconds = 1
            mock_settings.return_value.job_retention_seconds = 2

            manager = JobManager()
            yield manager

            # Cleanup: Shutdown manager and delete temp DB file
            await manager.shutdown()
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

    @pytest.mark.asyncio
    async def test_create_job_persists_to_real_db(self, manager, mock_request):
        """Verify that create_job actually puts a row in the DB."""
        job_id = "db-integration-1"
        
        async def mock_task(req): return "result.path"

        # Act
        await manager.create_job(
            job_id=job_id,
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=mock_task
        )

        # Assert: Directly check the database via the manager
        db_job = manager.db.get_job_by_id(job_id)
        
        assert db_job is not None
        assert db_job["id"] == job_id
        assert db_job["status"] == "queued"
        assert db_job["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_job_lifecycle_updates_db(self, manager, mock_request):
        """Verify the DB status changes as the job progresses."""
        job_id = "lifecycle-id"

        async def slow_task(req):
            await asyncio.sleep(0.2)
            return "done.mp4"

        await manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, slow_task)

        # 1. Check database while job is likely in progress
        await asyncio.sleep(0.1)
        db_job_mid = manager.db.get_job_by_id(job_id)
        assert db_job_mid["status"] == "in_progress"

        # 2. Check database after completion
        await asyncio.sleep(0.2)
        db_job_final = manager.db.get_job_by_id(job_id)
        assert db_job_final["status"] == "completed"
        assert db_job_final["result_path"] == "done.mp4"
        assert db_job_final["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_job_removes_from_db(self, manager, mock_request):
        """Verify that cancel_job actually deletes the row from the DB."""
        job_id = "cancel-test"
        
        async def long_task(req): await asyncio.sleep(5)
        
        await manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, long_task)
        
        # Verify it exists first
        assert manager.db.get_job_by_id(job_id) is not None
        
        # Act
        manager.cancel_job(job_id)
        
        # Assert: Should be gone from DB
        assert manager.db.get_job_by_id(job_id) is None

    @pytest.mark.asyncio
    async def test_restore_from_db_on_new_manager(self, mock_request):
        """
        Verify that if we create a NEW manager, it loads existing 
        jobs from the same database file.
        """
        fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        with patch("utils.job_manager.get_settings") as mock_settings, \
             patch("utils.job_database.DEFAULT_DB_PATH", Path(temp_db_path)):
            
            mock_settings.return_value.enable_job_persistence = True
            
            # 1. First manager creates a job
            m1 = JobManager()
            await m1.create_job("persisted-job", JobTypes.VIDEO, "m", mock_request, lambda r: None)
            await m1.shutdown()
            
            # Reset Singleton for the second manager
            import utils.job_manager
            utils.job_manager._job_manager_instance = None

            # 2. Second manager starts up using the same DB file
            m2 = JobManager()
            
            # Assert: The second manager should have the job in its memory list
            assert "persisted-job" in m2._jobs
            assert m2.get_job_metadata("persisted-job")["id"] == "persisted-job"

        os.remove(temp_db_path)