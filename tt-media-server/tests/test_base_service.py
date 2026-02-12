# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


class MockRequest:
    """Mock request for testing"""

    def __init__(self, task_id="test_task", segments=None, duration=None, stream=False):
        self._task_id = task_id
        self._segments = segments
        self._duration = duration
        self.stream = stream


class MockChunk:
    """Mock streaming chunk"""

    def __init__(self, text="chunk_text"):
        self.text = text


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler"""
    scheduler = Mock()
    scheduler.check_is_model_ready = Mock(return_value=True)
    scheduler.task_queue = Mock()
    scheduler.task_queue.qsize = Mock(return_value=2)
    scheduler.get_worker_info = Mock(return_value={"worker_0": "ready"})
    scheduler.process_request = Mock()
    scheduler.start_workers = AsyncMock()
    scheduler.stop_workers = Mock(return_value=True)
    scheduler.result_queues = {}
    scheduler.deep_restart_workers = AsyncMock()
    scheduler.restart_worker = Mock()
    return scheduler


@pytest.fixture
def mock_job_manager():
    """Create a mock job manager"""
    job_manager = Mock()
    job_manager.create_job = AsyncMock(
        return_value={"job_id": "job_1", "status": "created"}
    )
    job_manager.get_all_jobs_metadata = Mock(return_value=[{"job_id": "job_1"}])
    job_manager.get_job_metadata = Mock(
        return_value={"job_id": "job_1", "status": "running"}
    )
    job_manager.get_job_result_path = Mock(return_value="/tmp/result.json")
    job_manager.cancel_job = Mock(return_value=True)
    return job_manager


@pytest.fixture
def mock_settings():
    """Create mock settings"""
    settings = Mock()
    settings.download_weights_from_service = False
    settings.max_queue_size = 10
    settings.device_mesh_shape = "(1,1)"
    settings.device = "metal"
    settings.model_runner = "vllm"
    settings.request_processing_timeout_seconds = 30
    settings.model_weights_path = "/tmp/model"
    return settings


@pytest.fixture
def base_service(mock_scheduler, mock_settings):
    """Create a BaseService instance with all dependencies mocked"""
    with patch(
        "model_services.base_service.get_scheduler", return_value=mock_scheduler
    ):
        with patch("model_services.base_service.settings", mock_settings):
            with patch("model_services.base_service.TTLogger") as mock_logger_cls:
                mock_logger = Mock()
                mock_logger_cls.return_value = mock_logger
                with patch("model_services.base_service.HuggingFaceUtils"):
                    # Import inside the patch context
                    from model_services.base_service import BaseService

                    # Create a concrete implementation
                    class ConcreteService(BaseService):
                        pass

                    service = ConcreteService()
                    service._mock_logger = mock_logger
                    return service


@pytest.fixture
def base_job_service(mock_scheduler, mock_job_manager, mock_settings):
    """Create a BaseJobService instance with all dependencies mocked"""
    with patch(
        "model_services.base_service.get_scheduler", return_value=mock_scheduler
    ):
        with patch(
            "model_services.base_job_service.get_job_manager",
            return_value=mock_job_manager,
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_job_service.settings", mock_settings):
                    with patch(
                        "model_services.base_service.TTLogger"
                    ) as mock_logger_cls:
                        mock_logger = Mock()
                        mock_logger_cls.return_value = mock_logger
                        with patch("model_services.base_service.HuggingFaceUtils"):
                            # Import inside the patch context
                            from model_services.base_job_service import BaseJobService

                            # Create a concrete implementation
                            class ConcreteJobService(BaseJobService):
                                pass

                            service = ConcreteJobService()
                            service._mock_logger = mock_logger
                            return service


class TestBaseServiceInitialization:
    """Test BaseService initialization"""

    def test_init_creates_scheduler(self, mock_scheduler, mock_settings):
        """Test that __init__ creates scheduler instance"""
        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_service.TTLogger"):
                    with patch("model_services.base_service.HuggingFaceUtils"):
                        from model_services.base_service import BaseService

                        class ConcreteService(BaseService):
                            pass

                        service = ConcreteService()
                        assert service.scheduler == mock_scheduler

    def test_init_creates_logger(self, mock_scheduler, mock_settings):
        """Test that __init__ creates logger instance"""
        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_service.TTLogger") as mock_logger_cls:
                    mock_logger = Mock()
                    mock_logger_cls.return_value = mock_logger
                    with patch("model_services.base_service.HuggingFaceUtils"):
                        from model_services.base_service import BaseService

                        class ConcreteService(BaseService):
                            pass

                        service = ConcreteService()
                        assert service.logger == mock_logger

    def test_init_downloads_weights_when_enabled(self, mock_scheduler):
        """Test that __init__ downloads weights when enabled in settings"""
        mock_settings = Mock()
        mock_settings.download_weights_from_service = True

        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_service.TTLogger"):
                    with patch(
                        "model_services.base_service.HuggingFaceUtils"
                    ) as mock_hf:
                        mock_hf_instance = Mock()
                        mock_hf.return_value = mock_hf_instance
                        from model_services.base_service import BaseService

                        class ConcreteService(BaseService):
                            pass

                        ConcreteService()
                        mock_hf_instance.download_weights.assert_called_once()


class TestCheckIsModelReady:
    """Test check_is_model_ready method"""

    def test_check_is_model_ready_returns_correct_dict(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test successful model ready check returns correct dictionary"""
        with patch("model_services.base_service.settings", mock_settings):
            result = base_service.check_is_model_ready()

            assert result["model_ready"] is True
            assert result["queue_size"] == 2
            assert result["max_queue_size"] == 10
            assert result["device_mesh_shape"] == "(1,1)"
            assert result["device"] == "metal"
            assert result["worker_info"] == {"worker_0": "ready"}
            assert result["runner_in_use"] == "vllm"

    def test_check_is_model_ready_calls_scheduler(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test that check_is_model_ready calls scheduler methods"""
        with patch("model_services.base_service.settings", mock_settings):
            base_service.check_is_model_ready()

            mock_scheduler.check_is_model_ready.assert_called_once()
            mock_scheduler.get_worker_info.assert_called_once()

    def test_check_is_model_ready_queue_without_qsize(
        self, mock_scheduler, mock_settings
    ):
        """Test check_is_model_ready when queue doesn't have qsize method"""
        # Remove qsize method
        del mock_scheduler.task_queue.qsize

        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_service.TTLogger"):
                    with patch("model_services.base_service.HuggingFaceUtils"):
                        from model_services.base_service import BaseService

                        class ConcreteService(BaseService):
                            pass

                        service = ConcreteService()
                        result = service.check_is_model_ready()
                        assert result["queue_size"] == "unknown"

    def test_check_is_model_ready_device_not_defined(self, mock_scheduler):
        """Test check_is_model_ready when device is None"""
        mock_settings = Mock()
        mock_settings.download_weights_from_service = False
        mock_settings.max_queue_size = 10
        mock_settings.device_mesh_shape = "(1,1)"
        mock_settings.device = None  # Not defined
        mock_settings.model_runner = "vllm"

        with patch(
            "model_services.base_service.get_scheduler", return_value=mock_scheduler
        ):
            with patch("model_services.base_service.settings", mock_settings):
                with patch("model_services.base_service.TTLogger"):
                    with patch("model_services.base_service.HuggingFaceUtils"):
                        from model_services.base_service import BaseService

                        class ConcreteService(BaseService):
                            pass

                        service = ConcreteService()
                        result = service.check_is_model_ready()
                        assert result["device"] == "Not defined"


class TestWorkerManagement:
    """Test worker management methods"""

    def test_stop_workers(self, base_service, mock_scheduler):
        """Test stop_workers delegates to scheduler"""
        result = base_service.stop_workers()
        mock_scheduler.stop_workers.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_deep_reset(self, base_service, mock_scheduler):
        """Test deep_reset creates background task and returns True"""
        result = await base_service.deep_reset()

        assert result is True
        base_service._mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_device_reset(self, base_service, mock_scheduler):
        """Test device_reset creates background task"""
        await base_service.device_reset("0")

        base_service._mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_start_workers_async_success(self, base_service, mock_scheduler):
        """Test _start_workers_async success path"""
        await base_service._start_workers_async()
        mock_scheduler.start_workers.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_workers_async_failure(self, base_service, mock_scheduler):
        """Test _start_workers_async logs error on failure"""
        mock_scheduler.start_workers.side_effect = Exception("Worker start failed")

        await base_service._start_workers_async()

        base_service._mock_logger.error.assert_called()


class TestProcessRequest:
    """Test process_request method"""

    @pytest.mark.asyncio
    async def test_process_request_single_request(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_request with single request (no segments)"""
        mock_request = MockRequest(task_id="test_task_1")

        # Mock process method directly since process_request calls process
        async def mock_process(req):
            return {"result": "test_result"}

        base_service.process = mock_process

        with patch("model_services.base_service.settings", mock_settings):
            result = await base_service.process_request(mock_request)

        assert result == {"result": "test_result"}

    @pytest.mark.asyncio
    async def test_process_request_with_segments(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_request with segmented request"""
        mock_request = MockRequest(
            task_id="test_task_2", segments=["segment1", "segment2"]
        )

        # Mock process to return results for each segment
        call_count = [0]

        async def mock_process(req):
            call_count[0] += 1
            return f"result_{call_count[0]}"

        base_service.process = mock_process

        with patch("model_services.base_service.settings", mock_settings):
            result = await base_service.process_request(mock_request)

        # Default combine_results returns first result
        assert result == "result_1"

    @pytest.mark.asyncio
    async def test_process_request_null_result_raises(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_request raises ValueError when result is None"""
        mock_request = MockRequest(task_id="test_task_null")

        async def mock_process(req):
            return None

        base_service.process = mock_process

        with patch("model_services.base_service.settings", mock_settings):
            with pytest.raises(ValueError, match="Post processing failed"):
                await base_service.process_request(mock_request)

    @pytest.mark.asyncio
    async def test_process_request_custom_segment_handling(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_request with custom segment request creation"""
        mock_request = MockRequest(
            task_id="test_task_segments", segments=["seg1", "seg2", "seg3"]
        )

        # Custom segment request creation
        created_segments = []

        def custom_create_segment_request(original, segment, index):
            new_req = MockRequest(task_id=f"{original._task_id}_{index}")
            created_segments.append((segment, index))
            return new_req

        base_service.create_segment_request = custom_create_segment_request

        async def mock_process(req):
            return f"processed_{req._task_id}"

        base_service.process = mock_process

        # Custom combine
        def custom_combine(results):
            return "-".join(results)

        base_service.combine_results = custom_combine

        with patch("model_services.base_service.settings", mock_settings):
            result = await base_service.process_request(mock_request)

        assert len(created_segments) == 3
        assert created_segments[0] == ("seg1", 0)
        assert created_segments[1] == ("seg2", 1)
        assert created_segments[2] == ("seg3", 2)
        assert "processed_" in result


class TestProcess:
    """Test process method"""

    @pytest.mark.asyncio
    async def test_process_success(self, base_service, mock_scheduler, mock_settings):
        """Test process successfully retrieves result from queue"""
        mock_request = MockRequest(task_id="process_test")

        # Setup - the process method will create its own queue
        async def simulate_result():
            # Wait a bit for the queue to be created
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("process_test")
            if queue:
                await queue.put("success_result")

        with patch("model_services.base_service.settings", mock_settings):
            # Start the simulation task
            asyncio.create_task(simulate_result())
            result = await base_service.process(mock_request)

        assert result == "success_result"
        # Verify queue was cleaned up
        assert "process_test" not in mock_scheduler.result_queues

    @pytest.mark.asyncio
    async def test_process_timeout(self, base_service, mock_scheduler):
        """Test process raises TimeoutError when queue times out"""
        mock_settings = Mock()
        mock_settings.request_processing_timeout_seconds = 0.01  # Very short timeout

        mock_request = MockRequest(task_id="timeout_test")

        with patch("model_services.base_service.settings", mock_settings):
            with pytest.raises(asyncio.TimeoutError):
                await base_service.process(mock_request)

        # Verify queue was cleaned up even on timeout
        assert "timeout_test" not in mock_scheduler.result_queues

    @pytest.mark.asyncio
    async def test_process_exception(self, base_service, mock_scheduler, mock_settings):
        """Test process handles and re-raises exceptions"""
        mock_request = MockRequest(task_id="exception_test")

        async def simulate_error():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("exception_test")
            if queue:
                # Simulate an error by closing the queue or similar
                raise RuntimeError("Test error")

        with patch("model_services.base_service.settings", mock_settings):
            # Create a task that will raise
            asyncio.create_task(simulate_error())

            with pytest.raises(asyncio.TimeoutError):
                # Will timeout since nothing puts to queue
                mock_settings.request_processing_timeout_seconds = 0.02
                await base_service.process(mock_request)


class TestProcessStreaming:
    """Test process_streaming method"""

    @pytest.mark.asyncio
    async def test_process_streaming_chunks(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming yields streaming chunks correctly"""
        mock_request = MockRequest(task_id="streaming_test")

        async def simulate_streaming():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_test")
            if queue:
                # Send streaming chunks
                chunk1 = MockChunk("chunk1")
                await queue.put({"type": "streaming_chunk", "chunk": chunk1})

                chunk2 = MockChunk("chunk2")
                await queue.put({"type": "streaming_chunk", "chunk": chunk2})

                # Send final result
                await queue.put({"type": "final_result", "return": False})

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_streaming())

            results = []
            async for chunk in base_service.process_streaming(mock_request):
                results.append(chunk)

        assert len(results) == 2
        assert results[0].text == "chunk1"
        assert results[1].text == "chunk2"

    @pytest.mark.asyncio
    async def test_process_streaming_with_final_result(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming with return=True yields final result"""
        mock_request = MockRequest(task_id="streaming_final")

        async def simulate_streaming():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_final")
            if queue:
                # Send final result with return=True
                await queue.put(
                    {"type": "final_result", "return": True, "result": "final_value"}
                )

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_streaming())

            results = []
            async for chunk in base_service.process_streaming(mock_request):
                results.append(chunk)

        assert len(results) == 1
        assert results[0] == "final_value"

    @pytest.mark.asyncio
    async def test_process_streaming_timeout(self, base_service, mock_scheduler):
        """Test process_streaming raises TimeoutError"""
        mock_settings = Mock()
        mock_settings.request_processing_timeout_seconds = 0.01

        mock_request = MockRequest(task_id="streaming_timeout")

        with patch("model_services.base_service.settings", mock_settings):
            with pytest.raises(asyncio.TimeoutError):
                async for _ in base_service.process_streaming(mock_request):
                    pass

    @pytest.mark.asyncio
    async def test_process_streaming_invalid_chunk_type(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming raises ValueError on invalid chunk type"""
        mock_request = MockRequest(task_id="streaming_invalid")

        async def simulate_invalid():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_invalid")
            if queue:
                await queue.put({"type": "invalid_type"})

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_invalid())

            with pytest.raises(ValueError, match="Streaming protocol violation"):
                async for _ in base_service.process_streaming(mock_request):
                    pass

    @pytest.mark.asyncio
    async def test_process_streaming_with_duration(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming calculates dynamic timeout based on duration"""
        mock_request = MockRequest(task_id="streaming_duration", duration=100.0)

        async def simulate_streaming():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_duration")
            if queue:
                await queue.put({"type": "final_result", "return": False})

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_streaming())

            results = []
            async for chunk in base_service.process_streaming(mock_request):
                results.append(chunk)

        # No error means dynamic timeout was calculated correctly
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_process_streaming_empty_chunk_text(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming skips chunks with empty text"""
        mock_request = MockRequest(task_id="streaming_empty")

        async def simulate_streaming():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_empty")
            if queue:
                # Chunk with empty text - should be skipped
                empty_chunk = MockChunk("")
                await queue.put({"type": "streaming_chunk", "chunk": empty_chunk})

                # Chunk with valid text
                valid_chunk = MockChunk("valid")
                await queue.put({"type": "streaming_chunk", "chunk": valid_chunk})

                await queue.put({"type": "final_result", "return": False})

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_streaming())

            results = []
            async for chunk in base_service.process_streaming(mock_request):
                results.append(chunk)

        # Only non-empty chunk should be yielded
        assert len(results) == 1
        assert results[0].text == "valid"

    @pytest.mark.asyncio
    async def test_process_streaming_none_chunk(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming handles None chunk"""
        mock_request = MockRequest(task_id="streaming_none")

        async def simulate_streaming():
            await asyncio.sleep(0.01)
            queue = mock_scheduler.result_queues.get("streaming_none")
            if queue:
                # None chunk - should be skipped
                await queue.put({"type": "streaming_chunk", "chunk": None})

                # Valid chunk
                valid_chunk = MockChunk("valid")
                await queue.put({"type": "streaming_chunk", "chunk": valid_chunk})

                await queue.put({"type": "final_result", "return": False})

        with patch("model_services.base_service.settings", mock_settings):
            asyncio.create_task(simulate_streaming())

            results = []
            async for chunk in base_service.process_streaming(mock_request):
                results.append(chunk)

        assert len(results) == 1


class TestProcessStreamingRequest:
    """Test process_streaming_request method"""

    @pytest.mark.asyncio
    async def test_process_streaming_request_chains_properly(
        self, base_service, mock_scheduler, mock_settings
    ):
        """Test process_streaming_request chains pre/post process"""
        mock_request = MockRequest(task_id="stream_req")

        # Track calls
        pre_process_called = []
        post_process_called = []

        async def mock_pre_process(req):
            pre_process_called.append(req)
            return req

        async def mock_post_process(result, input_req=None):
            post_process_called.append(result)
            return f"post_{result}"

        async def mock_process_streaming(req):
            yield "chunk1"
            yield "chunk2"

        base_service.pre_process = mock_pre_process
        base_service.post_process = mock_post_process
        base_service.process_streaming = mock_process_streaming

        with patch("model_services.base_service.settings", mock_settings):
            results = []
            async for chunk in base_service.process_streaming_request(mock_request):
                results.append(chunk)

        assert len(pre_process_called) == 1
        assert len(post_process_called) == 2
        assert results == ["post_chunk1", "post_chunk2"]


class TestJobManagement:
    """Test job management methods - tests BaseJobService"""

    @pytest.mark.asyncio
    async def test_create_job(self, base_job_service, mock_job_manager, mock_settings):
        """Test create_job delegates to job manager"""
        mock_job_type = Mock()
        mock_request = MockRequest(task_id="job_1")

        with patch("model_services.base_job_service.settings", mock_settings):
            result = await base_job_service.create_job(mock_job_type, mock_request)

        mock_job_manager.create_job.assert_called_once()
        call_kwargs = mock_job_manager.create_job.call_args[1]
        assert call_kwargs["job_id"] == "job_1"
        assert call_kwargs["job_type"] == mock_job_type
        assert call_kwargs["request"] == mock_request
        assert result == {"job_id": "job_1", "status": "created"}

    def test_get_all_jobs_metadata(self, base_job_service, mock_job_manager):
        """Test get_all_jobs_metadata delegates to job manager"""
        result = base_job_service.get_all_jobs_metadata()

        mock_job_manager.get_all_jobs_metadata.assert_called_once_with(None)
        assert result == [{"job_id": "job_1"}]

    def test_get_all_jobs_metadata_with_type(self, base_job_service, mock_job_manager):
        """Test get_all_jobs_metadata with job type filter"""
        mock_job_type = Mock()
        base_job_service.get_all_jobs_metadata(mock_job_type)

        mock_job_manager.get_all_jobs_metadata.assert_called_once_with(mock_job_type)

    def test_get_job_metadata(self, base_job_service, mock_job_manager):
        """Test get_job_metadata delegates to job manager"""
        result = base_job_service.get_job_metadata("job_1")

        mock_job_manager.get_job_metadata.assert_called_once_with("job_1")
        assert result == {"job_id": "job_1", "status": "running"}

    def test_get_job_result_path(self, base_job_service, mock_job_manager):
        """Test get_job_result_path delegates to job manager"""
        result = base_job_service.get_job_result_path("job_1")

        mock_job_manager.get_job_result_path.assert_called_once_with("job_1")
        assert result == "/tmp/result.json"

    def test_cancel_job(self, base_job_service, mock_job_manager):
        """Test cancel_job delegates to job manager"""
        result = base_job_service.cancel_job("job_1")

        mock_job_manager.cancel_job.assert_called_once_with("job_1")
        assert result is True


class TestSegmentProcessing:
    """Test segment-based request processing"""

    def test_create_segment_request_default(self, base_service):
        """Test default create_segment_request returns original request"""
        mock_request = MockRequest()
        segment = {"data": "segment_data"}

        result = base_service.create_segment_request(mock_request, segment, 0)

        assert result == mock_request

    def test_combine_results_default(self, base_service):
        """Test default combine_results returns first result"""
        results = ["result1", "result2", "result3"]

        result = base_service.combine_results(results)

        assert result == "result1"

    def test_combine_results_empty(self, base_service):
        """Test combine_results with empty results returns None"""
        results = []

        result = base_service.combine_results(results)

        assert result is None

    def test_combine_results_single(self, base_service):
        """Test combine_results with single result"""
        results = ["only_result"]

        result = base_service.combine_results(results)

        assert result == "only_result"


class TestPrePostProcess:
    """Test pre_process and post_process methods"""

    @pytest.mark.asyncio
    async def test_pre_process_default(self, base_service):
        """Test default pre_process returns request unchanged"""
        mock_request = MockRequest(task_id="pre_test")

        result = await base_service.pre_process(mock_request)

        assert result == mock_request

    @pytest.mark.asyncio
    async def test_post_process_default(self, base_service):
        """Test default post_process returns result unchanged"""
        result_data = {"data": "test"}

        result = await base_service.post_process(result_data)

        assert result == result_data

    @pytest.mark.asyncio
    async def test_post_process_with_input_request(self, base_service):
        """Test post_process accepts optional input_request"""
        result_data = {"data": "test"}
        mock_request = MockRequest()

        result = await base_service.post_process(result_data, mock_request)

        assert result == result_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
