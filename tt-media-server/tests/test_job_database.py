import pytest
import os
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock
from utils.job_manager import JobManager, JobStatus, JobTypes
from domain.base_request import BaseRequest

class TestJobManagerDatabaseIntegration:
    """Integration tests using a real temporary database"""

    @pytest.fixture
    async def job_manager(self):
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

            job_manager = JobManager()
            yield job_manager

            # Cleanup: Shutdown job_manager and delete temp DB file
            await job_manager.shutdown()
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)

    @pytest.fixture
    def mock_request(self):
        """Create a mock BaseRequest"""
        request = Mock(spec=BaseRequest)
        request._task_id = "test-task-123"
        return request

    @pytest.mark.asyncio
    async def test_create_job_persists_to_real_db(self, job_manager, mock_request):
        """Verify that create_job actually puts a row in the DB."""
        job_id = "db-integration-1"
        
        async def mock_task(req): 
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id=job_id,
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=mock_task
        )

        db_job = job_manager.db.get_job_by_id(job_id)
        
        assert db_job is not None
        assert db_job["id"] == job_id
        assert db_job["job_type"] == "video"
        assert db_job["status"] == "queued"
        assert db_job["model"] == "test-model"
        assert db_job["request_parameters"] == mock_request.model_dump(mode="json")
        assert db_job["created_at"] is not None
        assert db_job["completed_at"] is None
        assert db_job["result_path"] is None
        assert db_job["error_message"] is None

    @pytest.mark.asyncio
    async def test_job_lifecycle_updates_db(self, job_manager, mock_request):
        """Verify the DB status changes as the job progresses."""
        job_id = "lifecycle-id"

        async def slow_task(req):
            await asyncio.sleep(0.3)
            return "done.mp4"

        await job_manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, slow_task)

        # Check database while job is likely in progress
        await asyncio.sleep(0.1)
        db_job_mid = job_manager.db.get_job_by_id(job_id)
        assert db_job_mid["status"] == "in_progress"

        # Check database after completion
        await asyncio.sleep(0.3)
        db_job_final = job_manager.db.get_job_by_id(job_id)
        assert db_job_final["status"] == "completed"
        assert db_job_final["result_path"] == "done.mp4"
        assert db_job_final["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_job_removes_from_db(self, job_manager, mock_request):
        """Verify that cancel_job actually deletes the row from the DB."""
        job_id = "cancel-test"
        
        async def long_task(req): 
            await asyncio.sleep(5)
            return "videos/test-123.mp4"
        
        await job_manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, long_task)
        
        # Verify it exists first
        assert job_manager.db.get_job_by_id(job_id) is not None
        
        # Cancel
        job_manager.cancel_job(job_id)
        
        # Assert: Should be gone from DB
        assert job_manager.db.get_job_by_id(job_id) is None

    @pytest.mark.asyncio
    async def test_restore_from_db_on_new_manager(self, mock_request):
        """
        Verify that if we create a NEW job_manager, it loads existing 
        jobs from the same database file.
        """
        fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        with patch("utils.job_manager.get_settings") as mock_settings, \
             patch("utils.job_database.DEFAULT_DB_PATH", Path(temp_db_path)):
            
            mock_settings.return_value.enable_job_persistence = True
            
            # 1. First job_manager creates a job
            m1 = JobManager()
            await m1.create_job("persisted-job", JobTypes.VIDEO, "m", mock_request, lambda r: None)
            await m1.shutdown()
            
            # Reset Singleton for the second job_manager
            import utils.job_manager
            utils.job_manager._job_manager_instance = None

            # 2. Second job_manager starts up using the same DB file
            m2 = JobManager()
            
            # Assert: The second job_manager should have the job in its memory list
            assert "persisted-job" in m2._jobs
            assert m2.get_job_metadata("persisted-job")["id"] == "persisted-job"

        os.remove(temp_db_path)