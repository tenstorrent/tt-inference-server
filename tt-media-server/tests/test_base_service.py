# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Skip all tests in this file - module-level mocking causes test isolation issues
pytestmark = pytest.mark.skip(
    reason="Disabled due to module-level mocking causing test isolation issues"
)

# Mock external dependencies
sys.modules["ttnn"] = Mock()

# Mock settings
mock_settings = Mock()
mock_settings.download_weights_from_service = False
mock_settings.max_queue_size = 10
mock_settings.device_mesh_shape = "(1,1)"
mock_settings.device = "metal"
mock_settings.model_runner = "vllm"
mock_settings.request_processing_timeout_seconds = 30
mock_settings.model_weights_path = "/tmp/model"

sys.modules["config.settings"] = Mock()
sys.modules["config.settings"].settings = mock_settings

# Mock logger
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=mock_logger)

# Mock decorators
sys.modules["utils.decorators"] = Mock()
sys.modules["utils.decorators"].log_execution_time = (
    lambda *args, **kwargs: lambda func: func
)

# Mock job manager
mock_job_manager = Mock()
sys.modules["utils.job_manager"] = Mock()
sys.modules["utils.job_manager"].get_job_manager = Mock(return_value=mock_job_manager)

# Mock HuggingFaceUtils
sys.modules["utils.hugging_face_utils"] = Mock()
sys.modules["utils.hugging_face_utils"].HuggingFaceUtils = Mock()

# Mock config constants
sys.modules["config.constants"] = Mock()
sys.modules["config.constants"].JobTypes = Mock()

# Mock telemetry
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["telemetry.telemetry_client"].TelemetryEvent = Mock()

# Mock domain
sys.modules["domain.base_request"] = Mock()
sys.modules["domain.base_request"].BaseRequest = Mock

# Import after mocking
from model_services.base_service import BaseService


class ConcreteBaseService(BaseService):
    """Concrete implementation for testing abstract BaseService"""

    async def post_process(self, result, input_request=None):
        return result

    async def pre_process(self, request):
        return request


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler"""
    scheduler = Mock()
    scheduler.check_is_model_ready = Mock(return_value=True)
    scheduler.task_queue = Mock()
    scheduler.task_queue.qsize = Mock(return_value=2)
    scheduler.get_worker_info = Mock(return_value={"worker_0": "ready"})
    scheduler.process_request = Mock()
    scheduler.start_workers = Mock()
    scheduler.stop_workers = Mock()
    scheduler.result_queues = {}
    scheduler.deep_restart_workers = AsyncMock()
    scheduler.restart_worker = Mock()
    return scheduler


@pytest.fixture
def service(mock_scheduler):
    """Create a ConcreteBaseService instance for testing"""
    with patch(
        "model_services.base_service.get_scheduler", return_value=mock_scheduler
    ):
        return ConcreteBaseService()


class TestBaseServiceInitialization:
    """Test BaseService initialization"""

    def test_init_creates_scheduler(self, mock_scheduler):
        """Test that __init__ creates scheduler instance"""
        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            service = ConcreteBaseService()
            assert service.scheduler == mock_scheduler

    def test_init_creates_logger(self, mock_scheduler):
        """Test that __init__ creates logger instance"""
        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            service = ConcreteBaseService()
            assert service.logger is not None

    def test_init_creates_job_manager(self, mock_scheduler):
        """Test that __init__ gets job manager"""
        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            service = ConcreteBaseService()
            assert service._job_manager == mock_job_manager


class TestCheckIsModelReady:
    """Test check_is_model_ready method"""

    def test_check_is_model_ready_success(self, service, mock_scheduler):
        """Test successful model ready check"""
        result = service.check_is_model_ready()

        assert result["model_ready"] is True
        assert result["queue_size"] == 2
        assert result["max_queue_size"] == 10
        assert result["device_mesh_shape"] == "(1,1)"
        assert result["device"] == "metal"
        assert result["worker_info"] == {"worker_0": "ready"}
        assert result["runner_in_use"] == "vllm"

    def test_check_is_model_ready_calls_scheduler(self, service, mock_scheduler):
        """Test that check_is_model_ready calls scheduler methods"""
        service.check_is_model_ready()

        mock_scheduler.check_is_model_ready.assert_called_once()
        mock_scheduler.get_worker_info.assert_called_once()


class TestWorkerManagement:
    """Test worker management methods"""

    def test_start_workers(self, service, mock_scheduler):
        """Test start_workers delegates to scheduler"""
        service.start_workers()
        mock_scheduler.start_workers.assert_called_once()

    def test_stop_workers(self, service, mock_scheduler):
        """Test stop_workers delegates to scheduler"""
        service.stop_workers()
        mock_scheduler.stop_workers.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_reset(self, service, mock_scheduler):
        """Test deep_reset creates background task"""
        result = await service.deep_reset()

        assert result is True
        mock_scheduler.deep_restart_workers.assert_called_once()
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_device_reset(self, service, mock_scheduler):
        """Test device_reset creates background task"""
        await service.device_reset("0")

        mock_logger.info.assert_called()
        # Task is created in background


class TestProcessRequest:
    """Test process_request method"""

    @pytest.mark.asyncio
    async def test_process_request_single_request(self, service, mock_scheduler):
        """Test process_request with single request (no segments)"""
        # Create mock request
        mock_request = Mock()
        mock_request._task_id = "test_task_1"
        mock_request._segments = None

        # Mock the process method to return a result
        service.process = AsyncMock(return_value={"result": "test_result"})

        # Execute
        result = await service.process_request(mock_request)

        # Verify
        assert result == {"result": "test_result"}
        service.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_request_with_segments(self, service, mock_scheduler):
        """Test process_request with segmented request"""
        # Create mock request with segments
        mock_request = Mock()
        mock_request._task_id = "test_task_2"
        mock_request._segments = ["segment1", "segment2"]

        # Mock segment processing
        service.create_segment_request = Mock(return_value=mock_request)
        service.process = AsyncMock(side_effect=["result1", "result2"])
        service.combine_results = Mock(return_value="combined_result")

        # Execute
        result = await service.process_request(mock_request)

        # Verify segments were processed
        assert service.create_segment_request.call_count == 2
        assert service.process.call_count == 2
        assert result == "combined_result"

    @pytest.mark.asyncio
    async def test_process_request_timeout(self, service, mock_scheduler):
        """Test process_request timeout"""
        # Create mock request
        mock_request = Mock()
        mock_request._task_id = "test_task_timeout"
        mock_request._segments = None

        # Mock process to raise timeout
        service.process = AsyncMock(side_effect=asyncio.TimeoutError())

        # Execute and verify timeout is raised
        with pytest.raises(asyncio.TimeoutError):
            await service.process_request(mock_request)

    @pytest.mark.asyncio
    async def test_process_request_queue_cleanup(self, service, mock_scheduler):
        """Test that result queue is cleaned up after processing"""
        mock_request = Mock()
        mock_request._task_id = "test_task_cleanup"
        mock_request._segments = None

        # Mock process to return a result
        service.process = AsyncMock(return_value="result")

        # Execute
        await service.process_request(mock_request)

        # No specific cleanup to verify in process_request, but ensure it doesn't error
        assert True


class TestProcessStreaming:
    """Test process_streaming method"""

    @pytest.mark.asyncio
    async def test_process_streaming_request_basic(self, service, mock_scheduler):
        """Test that process_streaming_request properly chains pre/post process"""
        mock_request = Mock()
        mock_request._task_id = "test_streaming"

        # Mock process_streaming to return chunks
        async def mock_gen(req):
            yield "chunk1"
            yield "chunk2"

        service.process_streaming = mock_gen
        service.pre_process = AsyncMock(return_value=mock_request)
        service.post_process = AsyncMock(side_effect=lambda x: x)

        # Collect results
        results = []
        async for chunk in service.process_streaming_request(mock_request):
            results.append(chunk)

        # Verify pre_process was called
        service.pre_process.assert_called_once_with(mock_request)
        # Verify post_process was called for each chunk
        assert service.post_process.call_count == 2
        # Verify we got both chunks
        assert len(results) == 2

    """Test streaming process in detail using mocks"""

    @pytest.mark.asyncio
    async def test_process_streaming_integration(self, service, mock_scheduler):
        """Test process_streaming_request integration"""
        mock_request = Mock()
        mock_request._task_id = "test_streaming_integration"

        # Mock process_streaming to return an async generator
        async def mock_stream_gen(req):
            yield "chunk1"
            yield "chunk2"

        service.process_streaming = mock_stream_gen

        # Collect results
        results = []
        async for chunk in service.process_streaming_request(mock_request):
            results.append(chunk)

        # Verify we got chunks (post_process returns them as-is)
        assert len(results) == 2


class TestJobManagement:
    """Test job management methods"""

    @pytest.mark.asyncio
    async def test_create_job(self, service):
        """Test create_job delegates to job manager"""
        mock_job_type = Mock()
        mock_request = Mock()
        mock_request._task_id = "job_1"

        mock_job_manager.create_job = AsyncMock(return_value={"job_id": "job_1"})

        result = await service.create_job(mock_job_type, mock_request)

        mock_job_manager.create_job.assert_called_once()
        assert result == {"job_id": "job_1"}

    def test_get_all_jobs_metadata(self, service):
        """Test get_all_jobs_metadata delegates to job manager"""
        mock_job_manager.get_all_jobs_metadata = Mock(
            return_value=[{"job_id": "job_1"}]
        )

        result = service.get_all_jobs_metadata()

        mock_job_manager.get_all_jobs_metadata.assert_called_once()
        assert result == [{"job_id": "job_1"}]

    def test_get_job_metadata(self, service):
        """Test get_job_metadata delegates to job manager"""
        mock_job_manager.get_job_metadata = Mock(return_value={"job_id": "job_1"})

        result = service.get_job_metadata("job_1")

        mock_job_manager.get_job_metadata.assert_called_once_with("job_1")
        assert result == {"job_id": "job_1"}

    def test_get_job_result(self, service):
        """Test get_job_result delegates to job manager"""
        mock_job_manager.get_job_result = Mock(return_value="job_result")

        result = service.get_job_result("job_1")

        mock_job_manager.get_job_result.assert_called_once_with("job_1")
        assert result == "job_result"

    def test_cancel_job(self, service):
        """Test cancel_job delegates to job manager"""
        mock_job_manager.cancel_job = Mock(return_value=True)

        result = service.cancel_job("job_1")

        mock_job_manager.cancel_job.assert_called_once_with("job_1")
        assert result is True


class TestProcessingSegmentation:
    """Test segment-based request processing"""

    def test_create_segment_request_default(self, service):
        """Test default create_segment_request returns original request"""
        mock_request = Mock()
        segment = {"data": "segment_data"}

        result = service.create_segment_request(mock_request, segment, 0)

        assert result == mock_request

    def test_combine_results_default(self, service):
        """Test default combine_results returns first result"""
        results = ["result1", "result2", "result3"]

        result = service.combine_results(results)

        assert result == "result1"

    def test_combine_results_empty(self, service):
        """Test combine_results with empty results"""
        results = []

        result = service.combine_results(results)

        assert result is None
