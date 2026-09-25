# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import asyncio
import os
import tempfile
import time
from multiprocessing import Event
from unittest.mock import patch

import pytest
from config.constants import JobTypes
from domain.adapter_merge_request import AdapterMergeRequest
from domain.base_request import BaseRequest
from utils.job_manager import (
    Job,
    JobManager,
    JobStatus,
    get_job_manager,
)


class TestJob:
    """Tests for Job dataclass"""

    def test_job_initialization(self):
        """Test Job is created with correct initial values"""
        job = Job(id="test-123", job_type="video", model="test-model")

        assert job.id == "test-123"
        assert job.job_type == "video"
        assert job.model == "test-model"
        assert job.request_parameters == {}
        assert job.status == JobStatus.QUEUED
        assert job.created_at is not None
        assert job.completed_at is None
        assert job.result_path is None
        assert job.error is None
        assert job._task is None

    def test_job_auto_timestamp(self):
        """Test Job automatically sets created_at timestamp"""
        before = int(time.time())
        job = Job(id="test-123", job_type="video", model="test-model")
        after = int(time.time())

        assert before <= job.created_at <= after

    def test_job_custom_timestamp(self):
        """Test Job respects custom created_at timestamp"""
        custom_time = 1000000
        job = Job(
            id="test-123",
            job_type="video",
            model="test-model",
            created_at=custom_time,
        )

        assert job.created_at == custom_time

    def test_mark_in_progress(self):
        """Test marking job as in progress"""
        tracker = type("Tracker", (), {"value": 0.0})()
        job = Job(
            id="test-123",
            job_type="training",
            model="test-model",
            _progress_tracker=tracker,
        )
        previous_progress = job.local_progress_time
        job.mark_in_progress()

        assert job.status == JobStatus.IN_PROGRESS
        assert job.is_in_progress()
        assert job.local_progress_time >= previous_progress
        assert tracker.value == job.local_progress_time

    def test_progress_time_reads_shared_tracker(self):
        """Worker heartbeats are reflected in the parent-side job."""

        expected_progress = time.monotonic() + 100

        class Tracker:
            value = expected_progress

        job = Job(
            id="test-123",
            job_type="training",
            model="test-model",
            _progress_tracker=Tracker(),
        )

        assert job.progress_time() == expected_progress

    def test_mark_completed(self):
        """Test marking job as completed"""
        job = Job(id="test-123", job_type="video", model="test-model")
        result_path = "videos/test-123.mp4"

        before = int(time.time())
        job.mark_completed(result_path)
        after = int(time.time())

        assert job.status == JobStatus.COMPLETED
        assert job.result_path == result_path
        assert before <= job.completed_at <= after
        assert job.is_completed()
        assert job.is_terminal()

    def test_mark_failed(self):
        """Test marking job as failed"""
        job = Job(id="test-123", job_type="video", model="test-model")

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

    def test_mark_cancelling(self):
        """Test marking job as cancelling"""
        job = Job(id="test-123", job_type="video", model="test-model")
        job.mark_cancelling()

        assert job.status == JobStatus.CANCELLING
        assert not job.is_terminal()
        assert job.completed_at is None

    def test_mark_cancelled(self):
        """Test marking job as cancelled"""
        job = Job(id="test-123", job_type="video", model="test-model")

        before = int(time.time())
        job.mark_cancelled()
        after = int(time.time())

        assert job.status == JobStatus.CANCELLED
        assert job.is_terminal()
        assert before <= job.completed_at <= after
        assert not job.is_completed()
        assert not job.is_in_progress()

    def test_is_terminal_with_cancelled_state(self):
        """Test that is_terminal includes COMPLETED, FAILED, and CANCELLED"""
        job = Job(id="test-123", job_type="video", model="test-model")

        assert not job.is_terminal()

        job.mark_completed("path/to/result")
        assert job.is_terminal()

        job.status = JobStatus.QUEUED
        job.mark_failed("error", "msg")
        assert job.is_terminal()

        job.status = JobStatus.QUEUED
        job.mark_cancelled()
        assert job.is_terminal()

    def test_to_public_dict_queued(self):
        """Test converting queued job to public dict"""
        job = Job(id="test-123", job_type="video", model="test-model", created_at=1000)

        result = job.to_public_dict()

        assert result == {
            "id": "test-123",
            "job_type": "video",
            "status": "queued",
            "created_at": 1000,
            "model": "test-model",
            "request_parameters": {},
        }

    def test_to_public_dict_completed(self):
        """Test converting completed job to public dict"""
        job = Job(id="test-123", job_type="video", model="test-model", created_at=1000)
        job.mark_completed("videos/test-123.mp4")

        result = job.to_public_dict()

        assert result["status"] == "completed"
        assert result["completed_at"] is not None
        assert "error" not in result

    def test_to_public_dict_failed(self):
        """Test converting failed job to public dict"""
        job = Job(id="test-123", job_type="video", model="test-model", created_at=1000)
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

    # We run all tests twice, once with persistence off and once with persistence on
    @pytest.fixture(params=[False, True], ids=["persistence_off", "persistence_on"])
    async def job_manager(self, request, tmp_path):
        """
        Create a fresh JobManager instance for each test
        toggling persistence based on parametrization.
        """
        # Reset singleton
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        test_db_file = tmp_path / "test_jobs.db"

        with patch("utils.job_manager.get_settings") as mock_settings:
            mock_settings.return_value.job_cleanup_interval_seconds = 1
            mock_settings.return_value.job_retention_seconds = 2
            mock_settings.return_value.job_max_stuck_time_seconds = 3
            mock_settings.return_value.max_jobs = 10
            # Keep this above max_jobs so existing tests still hit the record cap.
            mock_settings.return_value.max_queue_size = 1000

            mock_settings.return_value.enable_job_persistence = request.param

            with patch(
                "utils.job_manager.job_database_path",
                return_value=str(test_db_file),
            ):
                manager = JobManager()
            yield manager

            await manager.shutdown()

    @pytest.fixture
    def mock_request(self):
        """Create a mock BaseRequest"""

        # Creating mock request this way to ensure model_dump works correctly
        class MockRequest(BaseRequest):
            dummy_argument: str = "test-task-123"

        return MockRequest()

    @pytest.mark.asyncio
    async def test_create_job(self, job_manager, mock_request):
        """Test creating a job"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

        job_data = await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        assert job_data["id"] == "job-123"
        assert job_data["job_type"] == "video"
        assert job_data["status"] == "queued"
        assert job_data["model"] == "test-model"
        assert job_data["request_parameters"] == {"dummy_argument": "test-task-123"}

        job_metadata = job_manager.get_job_metadata("job-123")
        assert job_metadata is not None
        assert job_metadata["id"] == "job-123"

        # Check that job is correctly persisted to database if persistence is enabled
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job is not None
            assert db_job["id"] == "job-123"
            assert db_job["job_type"] == "video"
            assert db_job["status"] == "queued"
            assert db_job["model"] == "test-model"
            assert db_job["request_parameters"] == {"dummy_argument": "test-task-123"}
            assert db_job.get("created_at") is not None
            assert db_job.get("completed_at") is None
            assert db_job.get("result_path") is None
            assert db_job.get("error_message") is None

    @pytest.mark.asyncio
    async def test_create_adapter_merge_job_requires_source_job_id(
        self, job_manager, mock_request
    ):
        with pytest.raises(ValueError, match="must provide source_job_id"):
            await job_manager.create_job(
                job_id="merge-job",
                job_type=JobTypes.ADAPTER_MERGE,
                model="test-model",
                request=mock_request,
                task_function=None,
            )

    @pytest.mark.asyncio
    async def test_create_job_max_limit(self, job_manager, mock_request):
        """Test creating job fails when max limit reached"""

        async def task_func(req):
            return "videos/test-123.mp4"

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
    async def test_create_job_rejects_when_active_queue_full(
        self, job_manager, mock_request
    ):
        """Outstanding (non-terminal) jobs are capped by max_queue_size."""
        from fastapi import HTTPException

        job_manager._settings.max_queue_size = 2

        async def blockingTask(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=blockingTask,
        )
        await job_manager.create_job(
            job_id="job-2",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=blockingTask,
        )

        with pytest.raises(HTTPException) as excInfo:
            await job_manager.create_job(
                job_id="job-3",
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=blockingTask,
            )
        assert excInfo.value.status_code == 429
        assert "Task queue is full" in excInfo.value.detail

    @pytest.mark.asyncio
    async def test_create_job_completed_jobs_do_not_consume_queue(
        self, job_manager, mock_request
    ):
        """Completed jobs count toward max_jobs, not max_queue_size."""
        job_manager._settings.max_queue_size = 1

        async def quickTask(req):
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-done",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=quickTask,
        )
        await asyncio.sleep(0.15)
        assert job_manager.get_job_metadata("job-done")["status"] == "completed"

        jobData = await job_manager.create_job(
            job_id="job-next",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=quickTask,
        )
        assert jobData["id"] == "job-next"

    @pytest.mark.asyncio
    async def test_ten_concurrent_submits_honor_max_queue_size(
        self, job_manager, mock_request
    ):
        """Issue #4959: 10 concurrent jobs with max_queue_size=2 yield 8 x 429."""
        from fastapi import HTTPException

        job_manager._settings.max_queue_size = 2

        async def blockingTask(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

        async def submit(jobId):
            return await job_manager.create_job(
                job_id=jobId,
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=blockingTask,
            )

        results = await asyncio.gather(
            *[submit(f"job-{i}") for i in range(10)],
            return_exceptions=True,
        )
        accepted = [item for item in results if not isinstance(item, Exception)]
        rejected = [item for item in results if isinstance(item, HTTPException)]
        assert len(accepted) == 2
        assert len(rejected) == 8
        assert all(exc.status_code == 429 for exc in rejected)

    @pytest.mark.asyncio
    async def test_job_stays_queued_until_start_event(self, job_manager, mock_request):
        """Issue #4959: in_progress waits for device dispatch, not task creation."""
        startEvent = Event()

        async def blockingTask(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-wait",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=blockingTask,
            start_event=startEvent,
        )
        await asyncio.sleep(0.6)
        assert job_manager.get_job_metadata("job-wait")["status"] == "queued"

        startEvent.set()
        await asyncio.sleep(0.6)
        assert job_manager.get_job_metadata("job-wait")["status"] == "in_progress"
        job_manager.cancel_job("job-wait")

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
            return "videos/test-123.mp4"

        # Create multiple jobs
        await job_manager.create_job(
            job_id="job-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="job-2",
            job_type=JobTypes.VIDEO,
            model="test-model-2",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="job-3",
            job_type=JobTypes.TRAINING,
            model="test-model-3",
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
    async def test_get_all_jobs_metadata_filtered_by_type(
        self, job_manager, mock_request
    ):
        """Test get_all_jobs_metadata filters by job type"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

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
            job_id="training-1",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="training-2",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Filter by VIDEO type
        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.VIDEO)

        assert len(result) == 2
        for job in result:
            assert job["job_type"] == "video"
        job_ids = [job["id"] for job in result]
        assert "video-1" in job_ids
        assert "video-2" in job_ids

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_by_training_type(
        self, job_manager, mock_request
    ):
        """Test get_all_jobs_metadata filters by TRAINING type"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="video-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await job_manager.create_job(
            job_id="training-1",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.TRAINING)

        assert len(result) == 1
        assert result[0]["id"] == "training-1"
        assert result[0]["job_type"] == "training"

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_no_matches(
        self, job_manager, mock_request
    ):
        """Test get_all_jobs_metadata returns empty list when filter has no matches"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="video-1",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        # Filter by TRAINING type when only VIDEO exists
        result = job_manager.get_all_jobs_metadata(job_type=JobTypes.TRAINING)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_includes_all_statuses(
        self, job_manager, mock_request
    ):
        """Test get_all_jobs_metadata includes jobs in all statuses"""

        async def quick_task(req):
            return "videos/test-123.mp4"

        async def slow_task(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

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
    async def test_get_all_jobs_metadata_returns_public_dict(
        self, job_manager, mock_request
    ):
        """Test get_all_jobs_metadata returns public dict format (no private fields)"""

        async def task_func(req):
            return "videos/test-123.mp4"

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
        assert "job_type" in job_data
        assert "status" in job_data
        assert "created_at" in job_data
        assert "model" in job_data

        # Check private fields are NOT present
        assert "_task" not in job_data
        assert (
            "result_path" not in job_data
        )  # Result not in metadata, only in get_job_result_path

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_thread_safety(self, job_manager, mock_request):
        """Test get_all_jobs_metadata is thread-safe with concurrent access"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

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
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        result_path = job_manager.get_job_result_path("job-123")
        assert result_path is None

    @pytest.mark.asyncio
    async def test_get_job_result_path_set_on_creation(self, job_manager, mock_request):
        """Test getting result path set on creation"""

        async def task_func(req):
            await asyncio.sleep(10)  # Long running
            return "my_model_final_path.pt"

        await job_manager.create_job(
            job_id="job-with-result-path",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            result_path="my_model_initial_path.pt",
        )

        assert (
            job_manager.get_job_result_path("job-with-result-path")
            == "my_model_initial_path.pt"
        )
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-with-result-path")
            assert db_job is not None
            assert db_job.get("result_path") == "my_model_initial_path.pt"

    @pytest.mark.asyncio
    async def test_job_processing_success(self, job_manager, mock_request):
        """Test successful job processing"""

        async def task_func(req):
            await asyncio.sleep(0.5)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        await asyncio.sleep(0.05)

        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is not None
        assert metadata["status"] == "in_progress"
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job is not None
            assert db_job["status"] == "in_progress"

        await asyncio.sleep(0.5)

        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is not None
        assert metadata["status"] == "completed"
        result_path = job_manager.get_job_result_path("job-123")
        assert result_path == "videos/test-123.mp4"
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job is not None
            assert db_job["status"] == "completed"
            assert db_job["result_path"] == "videos/test-123.mp4"

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

        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job is not None
            assert db_job["status"] == "failed"
            assert db_job["error_message"]["code"] == "processing_error"

    @pytest.mark.asyncio
    async def test_job_processing_failure_invalid_result_path(
        self, job_manager, mock_request
    ):
        """Test job fails when task function returns non-string result_path"""

        async def task_func_returns_list(req):
            await asyncio.sleep(0.1)
            return ["path/one", "path/two"]  # Invalid: should be string

        # Test with list return
        await job_manager.create_job(
            job_id="job-list",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func_returns_list,
        )

        # Wait for processing
        await asyncio.sleep(0.3)

        job_list = job_manager.get_job_metadata("job-list")
        assert job_list["status"] == "failed"
        assert "result_path must be str" in job_list["error"]["message"]

        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-list")
            assert db_job is not None
            assert db_job["status"] == "failed"
            assert "result_path must be str" in db_job["error_message"]["message"]

    @pytest.mark.asyncio
    async def test_cancel_job_transitions_to_cancelled(self, job_manager, mock_request):
        """Verify job moves through cancelling then cancelled state."""

        async def long_task_with_slow_cleanup(req):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.5)
                raise

        await job_manager.create_job(
            job_id="job-123",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=long_task_with_slow_cleanup,
        )

        await asyncio.sleep(0.1)

        result = job_manager.cancel_job("job-123")
        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == "job-123"

        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is not None
        assert metadata["status"] == JobStatus.CANCELLING
        if job_manager.db:
            db_job_mid = job_manager.db.get_job_by_id("job-123")
            assert db_job_mid["status"] == "cancelling"

        await asyncio.sleep(0.6)

        # Job should not be deleted, but its status should be CANCELLED
        metadata = job_manager.get_job_metadata("job-123")
        assert metadata is not None
        assert metadata["status"] == JobStatus.CANCELLED
        assert metadata["completed_at"] is not None
        if job_manager.db:
            db_job_final = job_manager.db.get_job_by_id("job-123")
            assert db_job_final["status"] == "cancelled"
            assert db_job_final["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, job_manager):
        """Test deleting non-existent job"""
        result = job_manager.cancel_job("nonexistent")
        assert result is None

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
        assert job_manager.get_job_metadata("job-123")["status"] == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self, job_manager, mock_request):
        """QUEUED job can be cancelled immediately."""
        start_event_1 = Event()

        async def long_task(req):
            start_event_1.set()
            await asyncio.sleep(10)
            return "result.pt"

        await job_manager.create_job(
            job_id="job-running",
            job_type=JobTypes.TRAINING,
            model="m",
            request=mock_request,
            task_function=long_task,
            result_path="result.pt",
            start_event=start_event_1,
        )
        await job_manager.create_job(
            job_id="job-queued",
            job_type=JobTypes.TRAINING,
            model="m",
            request=mock_request,
            task_function=long_task,
            result_path="result.pt",
            start_event=Event(),
        )

        await asyncio.sleep(0.3)
        assert job_manager.get_job_metadata("job-running")["status"] == "in_progress"
        assert job_manager.get_job_metadata("job-queued")["status"] == "queued"

        result = job_manager.cancel_job("job-queued")
        assert result["status"] == "cancelled"

        if job_manager.db:
            assert (
                job_manager.db.get_job_by_id("job-running")["status"] == "in_progress"
            )
            assert job_manager.db.get_job_by_id("job-queued")["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_task_sets_cancel_event(self, job_manager, mock_request):
        """When a training job's task receives CancelledError, _cleanup_job in the handler sets cancel_event."""
        start_event = Event()
        cancel_event = Event()

        async def training_task(req):
            start_event.set()
            await asyncio.sleep(10)
            return "result.pt"

        await job_manager.create_job(
            job_id="job-ce",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=training_task,
            result_path="result.pt",
            start_event=start_event,
            cancel_event=cancel_event,
        )

        await asyncio.sleep(0.3)
        assert job_manager.get_job_metadata("job-ce")["status"] == "in_progress"
        assert not cancel_event.is_set()

        with job_manager._jobs_lock:
            job = job_manager._jobs["job-ce"]
        job._task.cancel()

        await asyncio.sleep(0.2)

        assert cancel_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_old_completed_jobs(self, job_manager, mock_request):
        """Test cleanup removes old completed jobs"""

        async def task_func(req):
            return "videos/test-123.mp4"

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
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job is None

    @pytest.mark.asyncio
    async def test_cleanup_skips_training_jobs_by_default(
        self, job_manager, mock_request
    ):
        async def task_func(req):
            return None

        await job_manager.create_job(
            job_id="training-job",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await asyncio.sleep(0.1)

        with job_manager._jobs_lock:
            job = job_manager._jobs["training-job"]
            job.completed_at = int(time.time()) - 10

        job_manager._cleanup_old_jobs()

        assert job_manager.get_job_metadata("training-job") is not None
        if job_manager.db:
            assert job_manager.db.get_job_by_id("training-job") is not None

    @pytest.mark.asyncio
    async def test_manual_delete_supports_video_jobs(self, job_manager):
        video_job = Job(
            id="video-job",
            job_type=JobTypes.VIDEO.value,
            model="test-model",
            status=JobStatus.COMPLETED,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[video_job.id] = video_job

        assert job_manager.delete_job(video_job.id) is True
        assert job_manager.get_job_metadata(video_job.id) is None

    @pytest.mark.asyncio
    async def test_training_delete_enforces_org(self, job_manager):
        job = Job(
            id="org-job",
            job_type=JobTypes.TRAINING.value,
            model="test-model",
            org_id="org-a",
            status=JobStatus.COMPLETED,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[job.id] = job

        assert job_manager.delete_job(job.id, org_id="org-b") is False

    @pytest.mark.asyncio
    async def test_manual_delete_rejects_training_result_outside_root(
        self, job_manager, mock_request, tmp_path
    ):
        result_dir = tmp_path / "outside-adapters-root"
        result_dir.mkdir()

        async def task_func(req):
            return str(result_dir)

        await job_manager.create_job(
            job_id="job-delete-failure",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            result_path=str(result_dir),
        )
        await asyncio.sleep(0.1)

        with pytest.raises(ValueError, match="Refusing to delete result outside"):
            job_manager.delete_job("job-delete-failure")

        assert job_manager.get_job_metadata("job-delete-failure") is not None
        assert result_dir.exists()
        if job_manager.db:
            assert job_manager.db.get_job_by_id("job-delete-failure") is not None

    @pytest.mark.asyncio
    async def test_cleanup_skips_adapter_merge_jobs_by_default(
        self, job_manager, mock_request, tmp_path
    ):
        result_dir = tmp_path / "merged-model"
        result_dir.mkdir()

        async def training_task(req):
            return None

        async def task_func(req):
            return str(result_dir)

        await job_manager.create_job(
            job_id="training-parent",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=training_task,
        )
        await asyncio.sleep(0.1)

        await job_manager.create_job(
            job_id="merge-job",
            job_type=JobTypes.ADAPTER_MERGE,
            model="test-model",
            request=AdapterMergeRequest(
                source_job_id="training-parent",
                checkpoint_id="checkpoint-1",
            ),
            task_function=task_func,
            result_path=str(result_dir),
        )
        await asyncio.sleep(0.1)

        with job_manager._jobs_lock:
            job_manager._jobs["merge-job"].completed_at = int(time.time()) - 10

        job_manager._cleanup_old_jobs()

        assert job_manager.get_job_metadata("merge-job") is not None
        assert result_dir.exists()

    @pytest.mark.asyncio
    async def test_manual_delete_removes_adapter_merge_result_directory(
        self, job_manager, mock_request, tmp_path
    ):
        merged_models_dir = tmp_path / "merged-models"
        result_dir = merged_models_dir / "merged-model"
        result_dir.mkdir(parents=True)

        async def training_task(req):
            return None

        async def task_func(req):
            return str(result_dir)

        await job_manager.create_job(
            job_id="training-parent",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=training_task,
        )
        await asyncio.sleep(0.1)

        await job_manager.create_job(
            job_id="merge-job-delete",
            job_type=JobTypes.ADAPTER_MERGE,
            model="test-model",
            request=AdapterMergeRequest(
                source_job_id="training-parent",
                checkpoint_id="checkpoint-1",
            ),
            task_function=task_func,
            result_path=str(result_dir),
        )
        await asyncio.sleep(0.1)

        with patch(
            "utils.job_manager.merged_models_root",
            return_value=str(merged_models_dir),
        ):
            assert job_manager.delete_job("merge-job-delete") is True
        assert job_manager.get_job_metadata("merge-job-delete") is None
        assert not result_dir.exists()

    @pytest.mark.asyncio
    async def test_manual_delete_removes_terminal_training_results(
        self, job_manager, mock_request, tmp_path
    ):
        adapters_dir = tmp_path / "adapters"
        result_dir = adapters_dir / "job-delete"
        result_dir.mkdir(parents=True)
        (result_dir / "adapter.bin").write_bytes(b"adapter")

        async def task_func(req):
            return str(result_dir)

        await job_manager.create_job(
            job_id="job-delete",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            result_path=str(result_dir),
        )
        await asyncio.sleep(0.1)

        with patch("utils.job_manager.adapters_root", return_value=str(adapters_dir)):
            assert job_manager.delete_job("job-delete")

        assert not result_dir.exists()
        assert job_manager.get_job_metadata("job-delete") is None
        if job_manager.db:
            assert job_manager.db.get_job_by_id("job-delete") is None

    @pytest.mark.asyncio
    async def test_manual_delete_training_job_cascades_to_adapter_merge_jobs(
        self, job_manager, mock_request, tmp_path
    ):
        adapters_dir = tmp_path / "adapters"
        training_result_dir = adapters_dir / "training-job"
        training_result_dir.mkdir(parents=True)
        merged_models_dir = tmp_path / "merged-models"
        merge_result_dir = merged_models_dir / "merge-job"
        merge_result_dir.mkdir(parents=True)

        async def training_task(req):
            return str(training_result_dir)

        async def merge_task(req):
            return str(merge_result_dir)

        await job_manager.create_job(
            job_id="training-job",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=training_task,
            result_path=str(training_result_dir),
        )
        await asyncio.sleep(0.1)

        merge_request = AdapterMergeRequest(
            source_job_id="training-job",
            checkpoint_id="checkpoint-1",
        )
        await job_manager.create_job(
            job_id="merge-job",
            job_type=JobTypes.ADAPTER_MERGE,
            model="test-model",
            request=merge_request,
            task_function=merge_task,
            result_path=str(merge_result_dir),
        )
        await asyncio.sleep(0.1)

        assert job_manager.get_job_metadata("training-job")[
            "adapter_merge_job_ids"
        ] == ["merge-job"]

        with patch(
            "utils.job_manager.adapters_root", return_value=str(adapters_dir)
        ), patch(
            "utils.job_manager.merged_models_root",
            return_value=str(merged_models_dir),
        ):
            assert job_manager.delete_job("training-job") is True

        assert job_manager.get_job_metadata("training-job") is None
        assert job_manager.get_job_metadata("merge-job") is None
        assert not training_result_dir.exists()
        assert not merge_result_dir.exists()
        if job_manager.db:
            assert job_manager.db.get_job_by_id("training-job") is None
            assert job_manager.db.get_job_by_id("merge-job") is None

    @pytest.mark.asyncio
    async def test_manual_delete_training_job_rejects_active_merge_child(
        self, job_manager
    ):
        training_job = Job(
            id="training-job",
            job_type=JobTypes.TRAINING.value,
            model="test-model",
            status=JobStatus.COMPLETED,
            adapter_merge_job_ids={"merge-job"},
        )
        merge_job = Job(
            id="merge-job",
            job_type=JobTypes.ADAPTER_MERGE.value,
            model="test-model",
            request_parameters={"source_job_id": "training-job"},
            status=JobStatus.IN_PROGRESS,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[training_job.id] = training_job
            job_manager._jobs[merge_job.id] = merge_job

        with pytest.raises(ValueError, match="'merge-job' is in_progress"):
            job_manager.delete_job("training-job")

        assert job_manager.get_job_metadata("training-job") is not None
        assert job_manager.get_job_metadata("merge-job") is not None

    @pytest.mark.asyncio
    async def test_manual_delete_training_job_rejects_child_from_other_org(
        self, job_manager
    ):
        training_job = Job(
            id="training-job",
            job_type=JobTypes.TRAINING.value,
            model="test-model",
            org_id="org-a",
            status=JobStatus.COMPLETED,
            adapter_merge_job_ids={"merge-job"},
        )
        merge_job = Job(
            id="merge-job",
            job_type=JobTypes.ADAPTER_MERGE.value,
            model="test-model",
            request_parameters={"source_job_id": "training-job"},
            org_id="org-b",
            status=JobStatus.COMPLETED,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[training_job.id] = training_job
            job_manager._jobs[merge_job.id] = merge_job

        with pytest.raises(ValueError, match="different organization"):
            job_manager.delete_job("training-job", org_id="org-a")

        assert job_manager.get_job_metadata("training-job") is not None
        assert job_manager.get_job_metadata("merge-job") is not None

    @pytest.mark.asyncio
    async def test_cascade_deletion_keeps_remaining_jobs_after_child_failure(
        self, job_manager
    ):
        training_job = Job(
            id="training-job",
            job_type=JobTypes.TRAINING.value,
            model="test-model",
            status=JobStatus.COMPLETED,
            adapter_merge_job_ids={"merge-a", "merge-b"},
        )
        merge_a = Job(
            id="merge-a",
            job_type=JobTypes.ADAPTER_MERGE.value,
            model="test-model",
            request_parameters={"source_job_id": "training-job"},
            status=JobStatus.COMPLETED,
        )
        merge_b = Job(
            id="merge-b",
            job_type=JobTypes.ADAPTER_MERGE.value,
            model="test-model",
            request_parameters={"source_job_id": "training-job"},
            status=JobStatus.COMPLETED,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[training_job.id] = training_job
            job_manager._jobs[merge_a.id] = merge_a
            job_manager._jobs[merge_b.id] = merge_b

        with patch.object(
            job_manager,
            "_delete_job_and_result",
            side_effect=[None, OSError("filesystem unavailable")],
        ), pytest.raises(OSError, match="filesystem unavailable"):
            job_manager.delete_job("training-job")

        assert job_manager.get_job_metadata("merge-a") is None
        assert job_manager.get_job_metadata("merge-b") is not None
        assert job_manager.get_job_metadata("training-job")[
            "adapter_merge_job_ids"
        ] == ["merge-b"]
        assert job_manager._deleting_job_ids == set()

    @pytest.mark.asyncio
    async def test_manual_delete_merge_job_unlinks_it_from_parent(self, job_manager):
        training_job = Job(
            id="training-job",
            job_type=JobTypes.TRAINING.value,
            model="test-model",
            status=JobStatus.COMPLETED,
            adapter_merge_job_ids={"merge-job"},
        )
        merge_job = Job(
            id="merge-job",
            job_type=JobTypes.ADAPTER_MERGE.value,
            model="test-model",
            request_parameters={"source_job_id": "training-job"},
            status=JobStatus.COMPLETED,
        )
        with job_manager._jobs_lock:
            job_manager._jobs[training_job.id] = training_job
            job_manager._jobs[merge_job.id] = merge_job

        assert job_manager.delete_job("merge-job") is True

        assert (
            job_manager.get_job_metadata("training-job")["adapter_merge_job_ids"] == []
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_manual_delete_db_failure_keeps_job_after_result_deletion(
        self, job_manager, mock_request, tmp_path
    ):
        adapters_dir = tmp_path / "adapters"
        result_dir = adapters_dir / "job-delete-db-failure"
        result_dir.mkdir(parents=True)

        async def task_func(req):
            return str(result_dir)

        await job_manager.create_job(
            job_id="job-delete-db-failure",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            result_path=str(result_dir),
        )
        await asyncio.sleep(0.1)

        with patch(
            "utils.job_manager.adapters_root", return_value=str(adapters_dir)
        ), patch.object(
            job_manager.db,
            "delete_job",
            side_effect=RuntimeError("db unavailable"),
        ), pytest.raises(RuntimeError, match="db unavailable"):
            job_manager.delete_job("job-delete-db-failure")

        assert not result_dir.exists()
        assert job_manager.get_job_metadata("job-delete-db-failure") is not None
        assert job_manager.db.get_job_by_id("job-delete-db-failure") is not None

    @pytest.mark.asyncio
    async def test_manual_delete_rejects_active_job(self, job_manager, mock_request):
        async def task_func(req):
            await asyncio.sleep(10)

        await job_manager.create_job(
            job_id="job-active-delete",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )
        await asyncio.sleep(0.1)

        with pytest.raises(ValueError, match="Only terminal jobs"):
            job_manager.delete_job("job-active-delete")

    @pytest.mark.asyncio
    async def test_cleanup_stuck_jobs(self, job_manager, mock_request):
        """Stuck jobs fail but remain available until retention cleanup."""

        async def task_func(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

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
            job.mark_in_progress()
            job.local_progress_time = time.monotonic() - 10

        # Run cleanup
        job_manager._cleanup_old_jobs()

        # Job should remain as a diagnosable terminal record.
        metadata = job_manager.get_job_metadata("job-123")
        assert metadata["status"] == "failed"
        assert metadata["error"]["code"] == "stale_job"
        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("job-123")
            assert db_job["status"] == "failed"

    @pytest.mark.asyncio
    async def test_cleanup_keeps_old_job_with_recent_progress(
        self, job_manager, mock_request
    ):
        """Total job age does not make an actively progressing job stale."""

        async def task_func(req):
            await asyncio.sleep(10)
            return "videos/test-123.mp4"

        await job_manager.create_job(
            job_id="job-active",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
        )

        with job_manager._jobs_lock:
            job = job_manager._jobs["job-active"]
            job.created_at = int(time.time()) - 100
            job.mark_in_progress()
            job.local_progress_time = time.monotonic()

        job_manager._cleanup_old_jobs()

        assert job_manager.get_job_metadata("job-active")["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_video_stuck_timeout_remains_age_based(self, job_manager):
        """Jobs without a heartbeat retain the existing age-based behavior."""
        job = Job(
            id="video-job",
            job_type=JobTypes.VIDEO.value,
            model="test-model",
            local_progress_time=100.0,
        )

        job.mark_in_progress()

        assert job.local_progress_time == 100.0
        assert job_manager._is_job_stuck(job, progress_cutoff=101.0)

    @pytest.mark.asyncio
    async def test_cleanup_rechecks_progress_before_failing(
        self, job_manager, mock_request
    ):
        """A heartbeat arriving after the scan prevents stale cancellation."""

        async def task_func(req):
            await asyncio.sleep(10)
            return "models/result"

        tracker = type("Tracker", (), {"value": time.monotonic() - 100})()
        await job_manager.create_job(
            job_id="job-race",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            progress_tracker=tracker,
        )

        with job_manager._jobs_lock:
            job = job_manager._jobs["job-race"]
            job.mark_in_progress()
            stale_time = time.monotonic() - 100
            job.local_progress_time = stale_time
            tracker.value = stale_time
            original_progress_time = job.progress_time
            progress_read_count = 0

            def progress_time_with_racing_heartbeat():
                nonlocal progress_read_count
                progress_read_count += 1
                if progress_read_count == 2:
                    tracker.value = time.monotonic()
                return original_progress_time()

            job.progress_time = progress_time_with_racing_heartbeat

        job_manager._cleanup_old_jobs()

        assert job_manager.get_job_metadata("job-race")["status"] == "in_progress"

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
    async def test_shutdown_workflow(self, job_manager, mock_request):
        """
        Test shutdown deletes jobs from memory but persists them in the database
        with the correct statuses.
        """

        async def task_func_long(req):
            await asyncio.sleep(10)
            return "videos/test-long.mp4"

        async def task_func_short(req):
            await asyncio.sleep(0.1)
            return "videos/test-short.mp4"

        await job_manager.create_job(
            job_id="job-running",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func_long,
        )

        await job_manager.create_job(
            job_id="job-finished",
            job_type=JobTypes.VIDEO,
            model="test-model",
            request=mock_request,
            task_function=task_func_short,
        )

        await asyncio.sleep(0.3)
        await job_manager.shutdown()
        await asyncio.sleep(0.1)

        # Verify jobs are not in memory but persisted in the database
        assert job_manager.get_job_metadata("job-finished") is None
        assert job_manager.get_job_metadata("job-running") is None

        if job_manager.db:
            finished_job = job_manager.db.get_job_by_id("job-finished")
            running_job = job_manager.db.get_job_by_id("job-running")

            assert finished_job is not None
            assert running_job is not None
            assert finished_job["status"] == "completed"
            assert running_job["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_get_job_manager_singleton(self):
        """Test get_job_manager returns singleton instance"""
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        with patch("utils.job_manager.get_settings") as mock_settings:
            # disabling persistence to prevent file creation during the test
            mock_settings.return_value.enable_job_persistence = False

            manager1 = get_job_manager()
            manager2 = get_job_manager()

            assert manager1 is manager2

        # Manual cleanup since no fixture is used
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
            return "videos/test-123.mp4"

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
            return "videos/test-123.mp4"

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

    @pytest.mark.asyncio
    async def test_cooperative_cancel_via_cancel_event(self, job_manager, mock_request):
        """Test cooperative cancellation where task checks cancel_event and returns normally."""

        start_event = Event()
        cancel_event = Event()

        async def cooperative_task(req):
            start_event.set()
            # Simulates a training loop checking cancel_event — same as the real runner
            while not cancel_event.is_set():
                await asyncio.sleep(0.05)
            return "models_save/result.pt"

        await job_manager.create_job(
            job_id="job-coop",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=cooperative_task,
            result_path="models_save/result.pt",
            start_event=start_event,
            cancel_event=cancel_event,
        )

        await asyncio.sleep(0.3)

        job_metadata1 = job_manager.get_job_metadata("job-coop")
        assert job_metadata1["status"] == "in_progress"

        job_metadata2 = job_manager.cancel_job("job-coop")
        assert job_metadata2["status"] == "cancelling"

        if job_manager.db:
            db_job2 = job_manager.db.get_job_by_id("job-coop")
            assert db_job2["status"] == "cancelling"

        # _cleanup_job should set the event, NOT cancel the task
        assert cancel_event.is_set()

        await asyncio.sleep(0.5)

        job_metadata3 = job_manager.get_job_metadata("job-coop")
        assert job_metadata3["status"] == "cancelled"
        assert job_metadata3["completed_at"] is not None

        if job_manager.db:
            db_job3 = job_manager.db.get_job_by_id("job-coop")
            assert db_job3["status"] == "cancelled"
            assert db_job3["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_queued_job_starts_after_in_progress_job_cancelled(
        self, job_manager, mock_request
    ):
        """When a running job is cancelled, the next queued job should start processing."""

        start_event_1 = Event()
        cancel_event_1 = Event()
        start_event_2 = Event()

        runner_available = asyncio.Event()

        async def task_1(req):
            start_event_1.set()
            while not cancel_event_1.is_set():
                await asyncio.sleep(0.05)
            runner_available.set()
            return "result_1.pt"

        async def task_2(req):
            await runner_available.wait()
            start_event_2.set()
            await asyncio.sleep(5)
            return "result_2.pt"

        await job_manager.create_job(
            job_id="job-1",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_1,
            result_path="result_1.pt",
            start_event=start_event_1,
            cancel_event=cancel_event_1,
        )

        await asyncio.sleep(0.3)

        assert job_manager.get_job_metadata("job-1")["status"] == "in_progress"

        # Second job should stay queued while job-1 is running
        await job_manager.create_job(
            job_id="job-2",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_2,
            start_event=start_event_2,
        )

        await asyncio.sleep(0.1)

        assert job_manager.get_job_metadata("job-2")["status"] == "queued"

        if job_manager.db:
            db_job2 = job_manager.db.get_job_by_id("job-2")
            assert db_job2["status"] == "queued"

        # Cancel job-1
        job_manager.cancel_job("job-1")
        await asyncio.sleep(0.5)

        assert job_manager.get_job_metadata("job-1")["status"] == "cancelled"
        assert job_manager.get_job_metadata("job-2")["status"] == "in_progress"

        if job_manager.db:
            db_job2 = job_manager.db.get_job_by_id("job-2")
            assert db_job2["status"] == "in_progress"

    # ------------------------------------------------------------------------------------------------
    # database-only tests
    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_restore_jobs_workflow(self, job_manager, mock_request):
        """Verify that multiple jobs are correctly restored upon manager restart."""

        job_ids = ["job-1", "job-2", "job-3"]
        db_path = job_manager.db.db_path

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "videos/test-123.mp4"

        for jid in job_ids:
            await job_manager.create_job(
                job_id=jid,
                job_type=JobTypes.VIDEO,
                model="test-model",
                request=mock_request,
                task_function=task_func,
            )

        await asyncio.sleep(0.2)
        await job_manager.shutdown()

        # Reset Singleton for the second instance
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        # Start the second manager pointing to the same DB
        with patch("utils.job_manager.get_settings") as mock_settings, patch(
            "utils.job_manager.job_database_path", return_value=str(db_path)
        ):
            mock_settings.return_value.enable_job_persistence = True

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_restore_rebuilds_adapter_merge_relationship(self, job_manager):
        db_path = job_manager.db.db_path
        job_manager.db.insert_job(
            "training-job",
            JobTypes.TRAINING.value,
            "test-model",
            {},
            "completed",
            1000,
            org_id="org-a",
        )
        job_manager.db.insert_job(
            "merge-job",
            JobTypes.ADAPTER_MERGE.value,
            "test-model",
            {
                "source_job_id": "training-job",
                "checkpoint_id": "checkpoint-1",
            },
            "completed",
            1001,
            org_id="org-a",
        )

        with patch("utils.job_manager.get_settings") as mock_settings, patch(
            "utils.job_manager.job_database_path", return_value=str(db_path)
        ):
            mock_settings.return_value.enable_job_persistence = True
            restored_manager = JobManager()
            try:
                assert restored_manager.get_job_metadata(
                    "training-job", org_id="org-a"
                )["adapter_merge_job_ids"] == ["merge-job"]
            finally:
                await restored_manager.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_restore_stuck_jobs_from_db(self, job_manager):
        """Verify stuck jobs are synced to terminal states and completed jobs stay completed."""

        db_path = job_manager.db.db_path

        job_manager.db.insert_job("job-done", "video", "m1", {}, "completed", 1000)
        job_manager.db.insert_job("job-stuck", "video", "m1", {}, "in_progress", 1001)
        job_manager.db.insert_job("job-aborting", "video", "m1", {}, "cancelling", 1002)
        job_manager.db.insert_job("job-stuck2", "video", "m1", {}, "queued", 1003)

        # Reset Singleton to simulate a fresh server start
        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        # Start a new manager and verify its auto-restore behavior
        with patch("utils.job_manager.get_settings") as mock_settings, patch(
            "utils.job_manager.job_database_path", return_value=str(db_path)
        ):
            mock_settings.return_value.enable_job_persistence = True

            m2 = JobManager()
            try:
                assert "job-done" in m2._jobs
                assert m2._jobs["job-done"].status == JobStatus.COMPLETED

                assert "job-stuck" in m2._jobs
                assert m2._jobs["job-stuck"].status == JobStatus.FAILED

                assert "job-stuck2" in m2._jobs
                assert m2._jobs["job-stuck2"].status == JobStatus.FAILED

                assert "job-aborting" in m2._jobs
                assert m2._jobs["job-aborting"].status == JobStatus.CANCELLED

                # Verify the database was updated to match the new memory states
                db_stuck = m2.db.get_job_by_id("job-stuck")
                db_aborting = m2.db.get_job_by_id("job-aborting")
                db_stuck2 = m2.db.get_job_by_id("job-stuck2")
                db_done = m2.db.get_job_by_id("job-done")
                assert db_stuck.get("status") == "failed"
                assert db_aborting.get("status") == "cancelled"
                assert db_stuck2.get("status") == "failed"
                assert db_done.get("status") == "completed"

            finally:
                await m2.shutdown()

    # ------------------------------------------------------------------------------------------------
    # metrics/logs/checkpoints-related tests
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    def test_insert_metric_duplicate_raises_exception(self, job_manager):
        """Test that inserting a metric with the same primary key raises an exception."""
        job_manager.db.insert_job(
            job_id="job-1",
            job_type=JobTypes.TRAINING.value,
            model="model-1",
            request_parameters={},
            status="in_progress",
            created_at=1000,
        )
        job_manager.db.insert_metric(
            job_id="job-1",
            global_step=10,
            epoch=1,
            metric_name="loss",
            value=0.5,
            timestamp=1000,
        )
        with pytest.raises(Exception):
            job_manager.db.insert_metric(
                job_id="job-1",
                global_step=10,
                epoch=1,
                metric_name="loss",
                value=0.9,
                timestamp=2000,
            )

    @pytest.mark.asyncio
    async def test_get_job_metrics_returns_correct_list(
        self, job_manager, mock_request
    ):
        metrics_list = [
            {
                "global_step": 1,
                "epoch": 1,
                "metric_name": "loss",
                "value": 0.5,
                "timestamp": 1000,
            }
        ]

        async def task_func(req):
            await asyncio.sleep(10)
            return "result.pt"

        await job_manager.create_job(
            job_id="train-1",
            job_type=JobTypes.TRAINING,
            model="m1",
            request=mock_request,
            task_function=task_func,
            result_path="result.pt",
            job_metrics=metrics_list,
        )

        result = job_manager.get_job_metrics("train-1")
        assert result is metrics_list

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_persist_job_data_to_db(self, job_manager, mock_request):
        """Metrics, checkpoints, and logs appended during training are all persisted to the database."""

        start_event = Event()
        start_event.set()
        metrics_list = []
        checkpoints_list = []
        logs_list = []

        async def task_func(req):
            # fmt: off
            metrics_list.append(
                {"global_step": 1, "epoch": 1, "metric_name": "loss", "value": 0.5, "learning_rate": 0.001, "timestamp": 1000}
            )
            metrics_list.append(
                {"global_step": 2, "epoch": 1, "metric_name": "loss", "value": 0.4, "learning_rate": 0.001, "timestamp": 1001}
            )
            checkpoints_list.append(
                {"id": "ckpt-step-100", "step": 100, "epoch": 1, "metrics": {"train_loss": 0.5}, "created_at": 1000.0}
            )
            checkpoints_list.append(
                {"id": "ckpt-step-200", "step": 200, "epoch": 2, "metrics": {"train_loss": 0.3}, "created_at": 1001.0}
            )
            logs_list.append(
                {"timestamp": "2025-01-01T00:00:00", "type": "info", "step": 1, "message": "Training started"}
            )
            logs_list.append(
                {"timestamp": "2025-01-01T00:01:00", "type": "info", "step": 10, "message": "Step 10 complete"}
            )
            # fmt: on
            await asyncio.sleep(1.5)
            return "result_dir"

        await job_manager.create_job(
            job_id="train-persist",
            job_type=JobTypes.TRAINING,
            model="m1",
            request=mock_request,
            task_function=task_func,
            result_path="result_dir",
            start_event=start_event,
            job_metrics=metrics_list,
            job_checkpoints=checkpoints_list,
            job_logs=logs_list,
        )

        for _ in range(20):
            db_metrics = job_manager.db.get_metrics_flat("train-persist")
            db_ckpts = job_manager.db.get_checkpoints("train-persist")
            db_logs = job_manager.db.get_logs("train-persist")
            if len(db_metrics) == 2 and len(db_ckpts) == 2 and len(db_logs) == 2:
                break
            await asyncio.sleep(0.3)

        db_metrics = job_manager.db.get_metrics_flat("train-persist")
        assert len(db_metrics) == 2
        assert db_metrics[0]["value"] == 0.5
        assert db_metrics[1]["value"] == 0.4
        assert db_metrics[0]["learning_rate"] == 0.001

        db_ckpts = job_manager.db.get_checkpoints("train-persist")
        assert len(db_ckpts) == 2
        assert db_ckpts[0]["id"] == "ckpt-step-100"
        assert db_ckpts[1]["id"] == "ckpt-step-200"

        db_logs = job_manager.db.get_logs("train-persist")
        assert len(db_logs) == 2
        assert db_logs[0]["message"] == "Training started"
        assert db_logs[1]["message"] == "Step 10 complete"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    async def test_restore_training_job_restores_metrics_checkpoints_and_logs(
        self, job_manager, tmp_path
    ):
        """Test full restore workflow: metrics, checkpoints (with disk validation), and logs."""

        db_path = job_manager.db.db_path
        result_dir = tmp_path / "adapter_dir"
        result_dir.mkdir()
        (result_dir / "ckpt-step-100").mkdir()
        # ckpt-step-200 deliberately missing from disk

        job_manager.db.insert_job(
            job_id="train-1",
            job_type=JobTypes.TRAINING.value,
            model="m1",
            request_parameters={},
            status="completed",
            created_at=1000,
        )
        job_manager.db.update_result_path("train-1", str(result_dir))

        # fmt: off
        job_manager.db.insert_metric(job_id="train-1", global_step=20, epoch=2, metric_name="loss", value=0.3, timestamp=1002)
        job_manager.db.insert_metric(job_id="train-1", global_step=10, epoch=1, metric_name="loss", value=0.5, timestamp=1000)
        job_manager.db.insert_metric(job_id="train-1", global_step=10, epoch=1, metric_name="accuracy", value=0.8, learning_rate=0.001, timestamp=1001)
        job_manager.db.insert_checkpoint("train-1", "ckpt-step-100", 100, 1, {"loss": 0.5}, 1001.0)
        job_manager.db.insert_checkpoint("train-1", "ckpt-step-200", 200, 2, {"loss": 0.3}, 1002.0)
        job_manager.db.insert_log("train-1", 0, "ts1", "info", 1, "Started")
        job_manager.db.insert_log("train-1", 1, "ts2", "info", 10, "Step 10")
        # fmt: on

        import utils.job_manager

        utils.job_manager._job_manager_instance = None

        with patch("utils.job_manager.get_settings") as mock_settings, patch(
            "utils.job_manager.job_database_path", return_value=str(db_path)
        ):
            mock_settings.return_value.enable_job_persistence = True

            m2 = JobManager()
            try:
                # Metrics: all restored, ordered by metric_name ASC, global_step ASC
                metrics = m2.get_job_metrics("train-1")
                assert len(metrics) == 3
                assert metrics[0]["metric_name"] == "accuracy"
                assert metrics[0]["learning_rate"] == 0.001
                assert metrics[1]["global_step"] == 10
                assert metrics[2]["timestamp"] == 1002

                # Checkpoints: only ckpt-step-100 survives disk validation
                ckpts = m2.get_job_checkpoints("train-1")
                assert len(ckpts) == 1
                assert ckpts[0]["id"] == "ckpt-step-100"

                # Logs: both restored
                logs = m2.get_job_logs("train-1")
                assert len(logs) == 2
                assert logs[0]["message"] == "Started"
                assert logs[1]["message"] == "Step 10"
            finally:
                await m2.shutdown()

    @pytest.mark.parametrize("job_manager", [True], indirect=True)
    def test_get_metrics_flat_sorted_by_metric_name_then_global_step(self, job_manager):
        """Test that get_metrics_flat returns metrics sorted by metric_name ASC, then global_step ASC."""

        job_manager.db.insert_job(
            job_id="job-sort",
            job_type=JobTypes.TRAINING.value,
            model="model-1",
            request_parameters={},
            status="in_progress",
            created_at=1000,
        )

        # fmt: off
        # Insert metrics in deliberately unsorted order
        job_manager.db.insert_metric(job_id="job-sort", global_step=30, epoch=3, metric_name="loss", value=0.1, timestamp=3000)
        job_manager.db.insert_metric(job_id="job-sort", global_step=10, epoch=1, metric_name="accuracy", value=0.7, timestamp=1000)
        job_manager.db.insert_metric(job_id="job-sort", global_step=20, epoch=2, metric_name="loss", value=0.3, timestamp=2000)
        job_manager.db.insert_metric(job_id="job-sort", global_step=10, epoch=1, metric_name="loss", value=0.5, timestamp=1000)
        job_manager.db.insert_metric(job_id="job-sort", global_step=30, epoch=3, metric_name="accuracy", value=0.95, timestamp=3000)
        job_manager.db.insert_metric(job_id="job-sort", global_step=20, epoch=2, metric_name="accuracy", value=0.85, timestamp=2000)
        # fmt: on

        results = job_manager.db.get_metrics_flat("job-sort")

        assert len(results) == 6

        metric_names = [r["metric_name"] for r in results]
        global_steps = [r["global_step"] for r in results]
        # fmt: off
        assert metric_names == ["accuracy", "accuracy", "accuracy", "loss", "loss", "loss"]
        assert global_steps == [10, 20, 30, 10, 20, 30]
        # fmt: on

    def test_some_checkpoints_missing_from_disk(self, job_manager, tmp_path):
        result_dir = tmp_path / "adapters"
        result_dir.mkdir()
        (result_dir / "ckpt-step-100").mkdir()
        # ckpt-step-200 deliberately missing
        job = Job(
            id="j1",
            job_type="training",
            model="m",
            result_path=str(result_dir),
            job_checkpoints=[
                {
                    "id": "ckpt-step-100",
                    "step": 100,
                    "epoch": 1,
                    "metrics": {},
                    "created_at": 1.0,
                },
                {
                    "id": "ckpt-step-200",
                    "step": 200,
                    "epoch": 2,
                    "metrics": {},
                    "created_at": 2.0,
                },
            ],
        )
        result = job_manager._validate_checkpoints_on_disk(job)
        assert len(result) == 1
        assert result[0]["id"] == "ckpt-step-100"

    @pytest.mark.asyncio
    async def test_get_job_logs_returns_correct_list(self, job_manager, mock_request):
        logs_list = [
            {
                "sample_log_entry": "Training started",
            }
        ]

        async def task_func(req):
            await asyncio.sleep(10)
            return "result.pt"

        await job_manager.create_job(
            job_id="train-logs-1",
            job_type=JobTypes.TRAINING,
            model="m1",
            request=mock_request,
            task_function=task_func,
            result_path="result.pt",
            job_logs=logs_list,
        )

        result = job_manager.get_job_logs("train-logs-1")
        assert result is logs_list

    @pytest.mark.asyncio
    async def test_get_job_logs_returns_none_for_unknown_job(self, job_manager):
        result = job_manager.get_job_logs("nonexistent")
        assert result is None

    # ------------------------------------------------------------------------------------------------
    # org_id scoping tests
    @pytest.mark.asyncio
    async def test_create_job_with_org_id(self, job_manager, mock_request):
        """Test creating a job with org_id stores it correctly"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "result.pt"

        await job_manager.create_job(
            job_id="org-job-1",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            org_id="org-abc",
        )

        if job_manager.db:
            db_job = job_manager.db.get_job_by_id("org-job-1")
            assert db_job["org_id"] == "org-abc"

    @pytest.mark.asyncio
    async def test_get_job_metadata_org_mismatch_returns_none(
        self, job_manager, mock_request
    ):
        """Test that accessing a job with wrong org_id returns None"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "result.pt"

        await job_manager.create_job(
            job_id="org-job-2",
            job_type=JobTypes.TRAINING,
            model="test-model",
            request=mock_request,
            task_function=task_func,
            org_id="org-abc",
        )

        assert job_manager.get_job_metadata("org-job-2", org_id="org-abc") is not None
        assert job_manager.get_job_metadata("org-job-2", org_id="org-other") is None
        assert job_manager.get_job_metadata("org-job-2", org_id=None) is not None

    @pytest.mark.asyncio
    async def test_get_all_jobs_metadata_filtered_by_org(
        self, job_manager, mock_request
    ):
        """Test that get_all_jobs_metadata filters by org_id"""

        async def task_func(req):
            await asyncio.sleep(0.1)
            return "result.pt"

        await job_manager.create_job(
            job_id="org-a-job",
            job_type=JobTypes.TRAINING,
            model="m",
            request=mock_request,
            task_function=task_func,
            org_id="org-a",
        )
        await job_manager.create_job(
            job_id="org-b-job",
            job_type=JobTypes.TRAINING,
            model="m",
            request=mock_request,
            task_function=task_func,
            org_id="org-b",
        )
        await job_manager.create_job(
            job_id="no-org-job",
            job_type=JobTypes.VIDEO,
            model="m",
            request=mock_request,
            task_function=task_func,
        )

        org_a_jobs = job_manager.get_all_jobs_metadata(org_id="org-a")
        assert len(org_a_jobs) == 1
        assert org_a_jobs[0]["id"] == "org-a-job"

        org_b_jobs = job_manager.get_all_jobs_metadata(org_id="org-b")
        assert len(org_b_jobs) == 1
        assert org_b_jobs[0]["id"] == "org-b-job"

        all_jobs = job_manager.get_all_jobs_metadata(org_id=None)
        assert len(all_jobs) == 3

    @pytest.mark.asyncio
    async def test_cancel_job_org_mismatch_returns_none(
        self, job_manager, mock_request
    ):
        """Test that cancelling a job with wrong org_id returns None"""

        async def task_func(req):
            await asyncio.sleep(10)
            return "result.pt"

        await job_manager.create_job(
            job_id="org-cancel-job",
            job_type=JobTypes.TRAINING,
            model="m",
            request=mock_request,
            task_function=task_func,
            org_id="org-abc",
        )

        await asyncio.sleep(0.1)

        assert job_manager.cancel_job("org-cancel-job", org_id="org-other") is None
        assert job_manager.cancel_job("org-cancel-job", org_id="org-abc") is not None
