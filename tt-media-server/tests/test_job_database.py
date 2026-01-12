import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, Mock
from utils.job_manager import JobManager, JobTypes
from domain.base_request import BaseRequest

class TestJobManagerDatabaseIntegration:
    """Integration tests using a real temporary database"""

    @pytest.fixture
    async def job_manager(self, tmp_path):
        """Setup a JobManager with a real temporary SQLite database."""
        # Reset singleton
        import utils.job_manager
        utils.job_manager._job_manager_instance = None

        test_db_file = tmp_path / "test_jobs.db"
        
        with patch("utils.job_manager.get_settings") as mock_settings:
            
            mock_settings.return_value.enable_job_persistence = True
            mock_settings.return_value.job_database_path = test_db_file

            mock_settings.return_value.max_jobs = 10
            mock_settings.return_value.job_cleanup_interval_seconds = 1
            mock_settings.return_value.job_retention_seconds = 2

            job_manager = JobManager()
            yield job_manager

            # Cleanup
            await job_manager.shutdown()

    @pytest.fixture
    def mock_request(self):
        """Create a mock BaseRequest"""
        request = Mock(spec=BaseRequest)
        request._task_id = "test-task-123"
        return request

    @pytest.mark.asyncio
    async def test_create_job_persists_to_db(self, job_manager, mock_request):
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

        async def fast_task(req):
            await asyncio.sleep(0.1)
            return "done.mp4"

        await job_manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, fast_task)

        await asyncio.sleep(0.05)
        db_job_mid = job_manager.db.get_job_by_id(job_id)
        assert db_job_mid["status"] == "in_progress"

        await asyncio.sleep(0.1)
        db_job_final = job_manager.db.get_job_by_id(job_id)
        assert db_job_final["status"] == "completed"
        assert db_job_final["result_path"] == "done.mp4"
    
    @pytest.mark.asyncio
    async def test_cancel_job_transitions_to_cancelled_in_db(self, job_manager, mock_request):
        """Verify job moves through cancelling then cancelled using sleeps."""
        job_id = "cancel-sleep-test"
        
        async def long_task_with_slow_cleanup(req):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.5) 
                raise

        await job_manager.create_job(job_id, JobTypes.VIDEO, "m", mock_request, long_task_with_slow_cleanup)
        
        await asyncio.sleep(0.1)
        
        success = job_manager.cancel_job(job_id)
        assert success is True
        
        db_job_mid = job_manager.db.get_job_by_id(job_id)
        assert db_job_mid["status"] == "cancelling"
        
        # Wait for the job to get in the cancelled state
        await asyncio.sleep(0.6)
        
        db_job_final = job_manager.db.get_job_by_id(job_id)
        assert db_job_final["status"] == "cancelled"
        assert db_job_final["completed_at"] is not None



    @pytest.mark.asyncio
    async def test_restore_multiple_jobs_from_db(self, job_manager, mock_request):
        """Verify that multiple jobs are correctly restored upon manager restart."""

        job_ids = ["job-1", "job-2", "job-3"]
        db_path = job_manager.db.db_path 

        for jid in job_ids:
            await job_manager.create_job(
                jid, 
                JobTypes.VIDEO, 
                "m", 
                mock_request, 
                lambda r: f"results/{jid}.mp4"
            )
        
        await asyncio.sleep(0.2) 
        await job_manager.shutdown()

        # Reset Singleton for the second instance
        import utils.job_manager
        utils.job_manager._job_manager_instance = None

        # Start the second manager pointing to the same DB
        with patch("utils.job_manager.get_settings") as mock_settings:
            mock_settings.return_value.enable_job_persistence = True
            mock_settings.return_value.job_database_path = str(db_path)
            
            m2 = JobManager()
            try:
                assert len(m2._jobs) == len(job_ids)
                
                for jid in job_ids:
                    assert jid in m2._jobs
                    metadata = m2.get_job_metadata(jid)
                    assert metadata["id"] == jid
                    assert metadata["status"] == "completed"
                    
            finally:
                await m2.shutdown()