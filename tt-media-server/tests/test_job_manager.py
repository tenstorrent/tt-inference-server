# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import tempfile
import time
from unittest.mock import Mock, patch

import pytest
from config.constants import JobTypes
from domain.base_request import BaseRequest
from utils.job_manager import Job, JobManager, JobStatus, get_job_manager


class TestJob:
    """Tests for Job dataclass"""

    def test_job_initialization(self):
        """Test Job is created with correct initial values"""
        job = Job(id="test-123", object="video", model="test-model")

        assert job.id == "test-123"
        assert job.object == "video"
        assert job.model == "test-model"
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None
        assert job.completed_at is None
        assert job.result is None
        assert job.error is None
        assert job._task is None

    def test_job_auto_timestamp(self):
        """Test Job automatically sets created_at timestamp"""
        before = int(time.time())
        job = Job(id="test-123", object="video", model="test-model")
        after = int(time.time())

        assert before <= job.created_at <= after

    def test_job_custom_timestamp(self):
        """Test Job respects custom created_at timestamp"""
        custom_time = 1000000
        job = Job(
            id="test-123",
            object="video",
            model="test-model",
            created_at=custom_time,
        )

        assert job.created_at == custom_time

    def test_mark_in_progress(self):
        """Test marking job as in progress"""
        job = Job(id="test-123", object="video", model="test-model")
        job.mark_in_progress()

        assert job.status == JobStatus.IN_PROGRESS
        assert job.is_in_progress()

    def test_mark_completed(self):
        """Test marking job as completed"""
        job = Job(id="test-123", object="video", model="test-model")
        result = b"video data"

        before = int(time.time())
        job.mark_completed(result)
        after = int(time.time())

        assert job.status == JobStatus.COMPLETED
        assert job.result == result
        assert before <= job.completed_at <= after
        assert job.is_completed()
        assert job.is_terminal()

    def test_mark_failed(self):
        """Test marking job as failed"""
        job = Job(id="test-123", object="video", model="test-model")

        before = int(time.time())
        job.mark_failed("processing_error", "Something went wrong")
        after = int(time.time())

        assert job.status == JobStatus.FAILED
        assert job.error == {
            "code": "processing_error",
            "message": "Something went wrong",
        }
        assert before <= job.completed_at <= after
        assert job.is_terminal()
        assert not job.is_completed()

    def test_to_public_dict_queued(self):
        """Test converting queued job to public dict"""
        job = Job(id="test-123", object="video", model="test-model", created_at=1000)

        result = job.to_public_dict()

        assert result == {
            "id": "test-123",
            "object": "video",
            "status": "queued",
            "created_at": 1000,
            "model": "test-model",
        }

    def test_to_public_dict_completed(self):
        """Test converting completed job to public dict"""
        job = Job(id="test-123", object="video", model="test-model", created_at=1000)
        job.mark_completed(b"result")

        result = job.to_public_dict()

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert "error" not in result

    def test_to_public_dict_failed(self):
        """Test converting failed job to public dict"""
        job = Job(id="test-123", object="video", model="test-model", created_at=1000)
        job.mark_failed("test_error", "Test error message")

        result = job.to_public_dict()

        assert result["status"] == "failed"
        assert result["error"] == {
            "code": "test_error",
            "message": "Test error message",
        }
        assert result["completed_at"] is not None


class TestJobManager:
    """Tests for JobManager class"""

    @pytest.fixture
    async def job_manager(self):
        """Create a fresh JobManager instance for each test"""
        # Reset singleton
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        with patch("utils.job_manager.get_settings") as mock_settings:
            mock_settings.return_value.job_cleanup_interval_seconds = 1
            mock_settings.return_value.job_retention_seconds = 2
            mock_settings.return_value.job_max_stuck_time_seconds = 3
            mock_settings.return_value.max_jobs = 10

            manager = JobManager()
            yield manager

            # Cleanup
            if manager._cleanup_task:
                manager._cleanup_task.cancel()
                try:
                    await manager._cleanup_task
                except asyncio.CancelledError:
                    pass

    @pytest.fixture
    def mock_request(self):
        """Create a mock BaseRequest"""
        request = Mock(spec=BaseRequest)
        request._task_id = "test-task-123"
        return request

    @pytest.mark.asyncio
    async def test_create_job(self, job_manager, mock_request):
        """Test creating a job"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        job_data = await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        assert job_data["id"] == "job-123"
        assert job_data["object"] == "video"
        assert job_data["status"] == "queued"
        assert job_data["model"] == "test-model"

        # Check job is in storage
        job_metadata = job_manager.get_job_metadata("job-123")
        assert job_metadata is not None
        assert job_metadata["id"] == "job-123"

    @pytest.mark.asyncio
    async def test_create_job_max_limit(self, job_manager, mock_request):
        """Test creating job fails when max limit reached"""

        async def task_func(req):
            return b"result"

        # Fill up to max
        for i in range(10):
            await job_manager.create_job(
                job_id=f"job-{i}",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

        # Next one should fail
        with pytest.raises(Exception, match="Maximum job limit reached"):
            await job_manager.create_job(
                job_id="job-overflow",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

        @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_empty(self, job_manager):
        """Test get_all_jobs_metadata returns empty list when no jobs exist"""
        result = job_manager.get_all_jobs_metadata()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_all_jobs(self, job_manager, mock_request):
        """Test get_all_jobs_metadata returns all jobs"""
        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        # Create jobs of different types
        await job_manager.create_job(
            job_id="job-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="job-2",
            job_type=JobTypes.IMAGE,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="job-3",
            job_type=JobTypes.AUDIO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result = job_manager.get_all_jobs_metadata()

        assert len(result) == 3
        job_ids = [job["id"] for job in result]
        assert "job-1" in job_ids
        assert "job-2" in job_ids
        assert "job-3" in job_ids

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_by_type(self, job_manager, mock_request):
        """Test get_all_jobs_metadata filters by job type"""
        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        # Create jobs of different types
        await job_manager.create_job(
            job_id="video-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="video-2",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="image-1",
            job_type=JobTypes.IMAGE,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="audio-1",
            job_type=JobTypes.AUDIO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Filter by VIDEO type
        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.VIDEO)

        assert len(result) == 2
        for job in result:
            assert job["object"] == "video"
        job_ids = [job["id"] for job in result]
        assert "video-1" in job_ids
        assert "video-2" in job_ids

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_by_image_type(self, job_manager, mock_request):
        """Test get_all_jobs_metadata filters by IMAGE type"""
        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        await job_manager.create_job(
            job_id="video-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="image-1",
            job_type=JobTypes.IMAGE,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.IMAGE)

        assert len(result) == 1
        assert result[0]["id"] == "image-1"
        assert result[0]["object"] == "image"

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_no_matches(self, job_manager, mock_request):
        """Test get_all_jobs_metadata returns empty list when filter has no matches"""
        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        await job_manager.create_job(
            job_id="video-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Filter by IMAGE type when only VIDEO exists
        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.IMAGE)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_includes_all_statuses(self, job_manager, mock_request):
        """Test get_all_jobs_metadata includes jobs in all statuses"""
        async def quick_task(req):
            return b"result"

        async def slow_task(req):
            await asyncio.sleep(10)
            return b"result"

        async def failing_task(req):
            raise ValueError("Test error")

        # Create jobs with different statuses
        await job_manager.create_job(
            job_id="completed-job",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=quick_task,
        )
        await job_manager.create_job(
            job_id="in-progress-job",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=slow_task,
        )
        await job_manager.create_job(
            job_id="failed-job",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=failing_task,
        )

        # Wait for status changes
        await asyncio.sleep(0.3)

        result = job_manager.get_all_jobs_metadata()

        assert len(result) == 3

        statuses = {job["id"]: job["status"] for job in result}
        assert statuses["completed-job"] == "completed"
        assert statuses["in-progress-job"] == "in_progress"
        assert statuses["failed-job"] == "failed"

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_returns_public_dict(self, job_manager, mock_request):
        """Test get_all_jobs_metadata returns public dict format (no private fields)"""
        async def task_func(req):
            return b"result"

        await job_manager.create_job(
            job_id="job-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.2)  # Wait for completion

        result = job_manager.get_all_jobs_metadata()

        assert len(result) == 1
        job_data = result[0]

        # Check required fields are present
        assert "id" in job_data
        assert "object" in job_data
        assert "status" in job_data
        assert "created_at" in job_data
        assert "model" in job_data

        # Check private fields are NOT present
        assert "_task" not in job_data
        assert "result" not in job_data  # Result not in metadata, only in get_job_result

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_thread_safety(self, job_manager, mock_request):
        """Test get_all_jobs_metadata is thread-safe with concurrent access"""
        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        # Create some jobs
        for i in range(5):
            await job_manager.create_job(
                job_id=f"job-{i}",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

        # Concurrently read metadata multiple times
        results = await asyncio.gather(
            asyncio.to_thread(job_manager.get_all_jobs_metadata),
            asyncio.to_thread(job_manager.get_all_jobs_metadata),
            asyncio.to_thread(job_manager.get_all_jobs_metadata),
        )

        # All reads should return the same number of jobs
        assert len(results[0]) == 5
        assert len(results[1]) == 5
        assert len(results[2]) == 5

    @pytest.mark.asyncio
    async def test_get_job_metadata_not_found(self, job_manager):
        """Test getting metadata for non-existent job"""
        result = job_manager.get_job_metadata("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_job_result_not_terminal(self, job_manager, mock_request):
        """Test getting result for non-terminal job returns None"""

        async def task_func(req):
            await asyncio.sleep(10)  # Long running
            return b"result"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result = job_manager.get_job_result("job-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_job_processing_success(self, job_manager, mock_request):
        """Test successful job processing"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"video data"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Wait for processing
        await asyncio.sleep(0.3)

        metadata = job_manager.get_job_metadata("job-123")
        assert metadata["status"] == "completed"

        result = job_manager.get_job_result("job-123")
        assert result == b"video data"

    @pytest.mark.asyncio
    async def test_job_processing_failure(self, job_manager, mock_request):
        """Test job processing failure"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            raise ValueError("Processing failed")

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Wait for processing
        await asyncio.sleep(0.3)

        metadata = job_manager.get_job_metadata("job-123")
        assert metadata["status"] == "failed"
        assert metadata["error"]["code"] == "processing_error"
        assert "Processing failed" in metadata["error"]["message"]

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, job_manager, mock_request):
        """Test deleting an existing job"""

        async def task_func(req):
            await asyncio.sleep(10)
            return b"result"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result = job_manager.cancel_job("job-123")
        assert result is True

        # Job should be gone
        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, job_manager):
        """Test deleting non-existent job"""
        result = job_manager.cancel_job("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_job_cancels_task(self, job_manager, mock_request):
        """Test deleting job cancels running task"""
        cancelled = False

        async def task_func(req):
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled = True
                raise

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.1)  # Let task start

        job_manager.cancel_job("job-123")
        await asyncio.sleep(0.1)  # Let cancellation propagate

        assert cancelled is True

    @pytest.mark.asyncio
    async def test_cleanup_old_completed_jobs(self, job_manager, mock_request):
        """Test cleanup removes old completed jobs"""

        async def task_func(req):
            return b"result"

        # Create and complete a job
        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.2)  # Wait for completion

        # Manually set old completion time
        with job_manager._jobs_lock:
            job = job_manager._jobs["job-123"]
            job.completed_at = int(time.time()) - 10  # 10 seconds ago

        # Run cleanup
        job_manager._cleanup_old_jobs()

        # Job should be removed
        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_cleanup_stuck_jobs(self, job_manager, mock_request):
        """Test cleanup cancels stuck jobs"""

        async def task_func(req):
            await asyncio.sleep(10)
            return b"result"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.1)  # Let job start

        # Manually set old creation time
        with job_manager._jobs_lock:
            job = job_manager._jobs["job-123"]
            job.created_at = int(time.time()) - 10  # 10 seconds ago
            job.mark_in_progress()

        # Run cleanup
        job_manager._cleanup_old_jobs()

        # Job should be removed
        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is None

    @pytest.mark.asyncio
    async def test_cleanup_deletes_result_files(self, job_manager, mock_request):
        """Test cleanup deletes result files"""
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            file_path = f.name
            f.write(b"video data")

        try:

            async def task_func(req):
                return file_path

            await job_manager.create_job(
                job_id="job-123",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

            await asyncio.sleep(0.2)  # Wait for completion

            # File should exist
            assert os.path.exists(file_path)

            # Manually set old completion time
            with job_manager._jobs_lock:
                job = job_manager._jobs["job-123"]
                job.completed_at = int(time.time()) - 10

            # Run cleanup
            job_manager._cleanup_old_jobs()

            # File should be deleted
            assert not os.path.exists(file_path)

        finally:
            # Cleanup in case test fails
            if os.path.exists(file_path):
                os.remove(file_path)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_cleanup_task(self, job_manager):
        """Test shutdown cancels cleanup task"""
        assert job_manager._cleanup_task is not None
        assert not job_manager._cleanup_task.done()

        await job_manager.shutdown()

        # Task should be done (either cancelled or completed)
        assert job_manager._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_running_jobs(self, job_manager, mock_request):
        """Test shutdown cancels all running jobs"""

        async def task_func(req):
            await asyncio.sleep(10)
            return b"result"

        # Create multiple jobs
        for i in range(3):
            job_manager.create_job(
                job_id=f"job-{i}",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

        await asyncio.sleep(0.1)  # Let jobs start

        await job_manager.shutdown()

        # All tasks should be cancelled or done
        with job_manager._jobs_lock:
            for job in job_manager._jobs.values():
                if job._task:
                    assert job._task.done() or job._task.cancelled()

    @pytest.mark.asyncio
    async def test_get_job_manager_singleton(self):
        """Test get_job_manager returns singleton instance"""
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        manager1 = get_job_manager()
        manager2 = get_job_manager()

        assert manager1 is manager2

        # Cleanup
        if manager1._cleanup_task:
            manager1._cleanup_task.cancel()
            try:
                await manager1._cleanup_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_creates(self, job_manager, mock_request):
        """Test thread safety with concurrent job creation (asyncio version)"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        async def create_jobs(prefix):
            for i in range(5):
                try:
                    await job_manager.create_job(
                        job_id=f"job-{prefix}-{i}",
                        job_type=JobTypes.VIDEO,
                        model="test-model",
                        request=mock_request,
                        task_function=task_func,
                    )
                except Exception:
                    pass  # May hit max limit

        # Run concurrent job creation using asyncio.gather
        await asyncio.gather(
            create_jobs("A"),
            create_jobs("B"),
            create_jobs("C"),
        )

        # Should have created some jobs without crashes
        with job_manager._jobs_lock:
            assert len(job_manager._jobs) > 0

    @pytest.mark.asyncio
    async def test_task_reference_cleared_after_completion(
        self, job_manager, mock_request
    ):
        """Test task reference is cleared after job completion"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return b"result"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.3)  # Wait for completion

        with job_manager._jobs_lock:
            job = job_manager._jobs.get("job-123")
            assert job is not None
            assert job._task is None  # Should be cleared
