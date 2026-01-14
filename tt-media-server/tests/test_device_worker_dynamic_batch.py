# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import sys
from unittest.mock import Mock, patch

import pytest

# Mock all external dependencies before importing
sys.modules["ttnn"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_unet"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_embedding"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.sdxl_utility"] = Mock()

# Mock config settings - must be done before any imports that use settings
mock_settings = Mock()
mock_settings.max_batch_size = 4
mock_settings.default_throttle_level = "5"
mock_settings.enable_telemetry = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)
mock_settings.request_processing_timeout_seconds = 100

# Mock the settings module completely
mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.Settings = Mock(return_value=mock_settings)
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module

# Mock telemetry before it gets imported
sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["telemetry.telemetry_client"].get_telemetry_client = Mock()

# Mock torch utils before it gets imported
sys.modules["utils.torch_utils"] = Mock()
sys.modules["utils.torch_utils"].set_torch_thread_limits = Mock()

# Mock device manager to prevent actual device detection
sys.modules["utils.device_manager"] = Mock()


# Mock domain objects
class MockCompletionRequest:
    def __init__(self, task_id, prompt="test prompt", stream=False):
        self._task_id = task_id
        self.prompt = prompt
        self.stream = stream


sys.modules["domain.completion_request"] = Mock()
sys.modules["domain.completion_request"].CompletionRequest = MockCompletionRequest

# Mock device runner and fabric
mock_device_runner = Mock()
mock_device_runner.set_device.return_value = Mock()
mock_device_runner.warmup = Mock(return_value=asyncio.Future())
mock_device_runner.warmup.return_value.set_result(None)
mock_device_runner.run.return_value = [Mock()]

# Mock logger
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger.return_value = mock_logger

# Now import the module under test
from device_workers.device_worker_dynamic_batch import device_worker


# Module level fixtures
@pytest.fixture
def mock_queues():
    """Create mock queues for testing"""
    task_queue = Mock()
    task_queue.get = Mock()

    result_queue = Mock()
    result_queue.put = Mock()

    warmup_signals_queue = Mock()
    warmup_signals_queue.put = Mock()
    warmup_signals_queue._closed = False

    error_queue = Mock()
    error_queue.put = Mock()

    return task_queue, result_queue, warmup_signals_queue, error_queue


class TestDeviceWorkerDynamicBatch:
    """Test cases for dynamic batch device_worker function"""

    def test_device_worker_initialization_success(self, mock_queues):
        """Test successful worker initialization"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Mock immediate shutdown
        task_queue.get.return_value = None

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify warmup signal was sent
        warmup_signals_queue.put.assert_called_once_with("worker_0", timeout=2.0)

    def test_device_worker_initialization_failure(self, mock_queues):
        """Test worker initialization failure"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Mock initialization failure
        mock_get_device_runner = Mock(
            side_effect=Exception("Device initialization failed")
        )

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify error handling
        error_queue.put.assert_called_once_with(
            ("worker_0", -1, "Device initialization failed")
        )

    def test_device_worker_non_streaming_request(self, mock_queues):
        """Test handling of non-streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create non-streaming request
        regular_request = MockCompletionRequest("task_1", stream=False)

        # Mock queue to return request then shutdown
        task_queue.get.side_effect = [regular_request, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        mock_response = Mock()
        fresh_device_runner.run.return_value = [mock_response]

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify result was queued
        assert len(results_captured) == 1
        assert results_captured[0] == ("worker_0", "task_1", mock_response)

    def test_device_worker_streaming_request(self, mock_queues):
        """Test handling of streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create streaming request
        streaming_request = MockCompletionRequest("stream_task_1", stream=True)

        # Mock queue to return request then shutdown
        task_queue.get.side_effect = [streaming_request, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Create async generator for streaming
        async def mock_async_generator():
            yield "chunk_1"
            yield "chunk_2"
            yield "chunk_3"

        async def mock_run_async(requests):
            return mock_async_generator()

        fresh_device_runner._run_async = mock_run_async

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify streaming chunks were queued
        assert len(results_captured) == 3
        assert results_captured[0] == ("worker_0", "stream_task_1", "chunk_1")
        assert results_captured[1] == ("worker_0", "stream_task_1", "chunk_2")
        assert results_captured[2] == ("worker_0", "stream_task_1", "chunk_3")

    def test_device_worker_multiple_streaming_requests(self, mock_queues):
        """Test handling multiple streaming requests concurrently"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create multiple streaming requests
        stream_req_1 = MockCompletionRequest("stream_1", stream=True)
        stream_req_2 = MockCompletionRequest("stream_2", stream=True)

        # Mock queue to return requests then shutdown
        task_queue.get.side_effect = [stream_req_1, stream_req_2, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Create async generators for each request
        call_count = [0]

        async def mock_run_async(requests):
            call_count[0] += 1
            task_id = requests[0]._task_id

            async def generator():
                if task_id == "stream_1":
                    yield f"{task_id}_chunk_A"
                    yield f"{task_id}_chunk_B"
                else:
                    yield f"{task_id}_chunk_X"
                    yield f"{task_id}_chunk_Y"

            return generator()

        fresh_device_runner._run_async = mock_run_async

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify all chunks were queued (order may vary due to concurrency)
        assert len(results_captured) == 4

        # Check that chunks from both streams are present
        task_1_chunks = [r for r in results_captured if r[1] == "stream_1"]
        task_2_chunks = [r for r in results_captured if r[1] == "stream_2"]

        assert len(task_1_chunks) == 2
        assert len(task_2_chunks) == 2

    def test_device_worker_mixed_requests(self, mock_queues):
        """Test handling mix of streaming and non-streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create mixed requests
        regular_req = MockCompletionRequest("regular", stream=False)
        stream_req = MockCompletionRequest("streaming", stream=True)

        # Mock queue
        task_queue.get.side_effect = [regular_req, stream_req, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Setup responses
        mock_response = Mock()
        fresh_device_runner.run.return_value = [mock_response]

        async def mock_async_generator():
            yield "stream_chunk"

        async def mock_run_async(requests):
            return mock_async_generator()

        fresh_device_runner._run_async = mock_run_async

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify both types of results were queued
        assert len(results_captured) == 2

        # Check that both task results are present (order may vary)
        task_ids = [r[1] for r in results_captured]
        assert "regular" in task_ids
        assert "streaming" in task_ids

    def test_device_worker_non_streaming_error(self, mock_queues):
        """Test error handling for non-streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create request that will fail
        failing_request = MockCompletionRequest("fail_task", stream=False)

        task_queue.get.side_effect = [failing_request, None]

        # Create fresh device runner that raises exception
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        fresh_device_runner.run.side_effect = Exception("Execution failed")

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify error was queued
        error_queue.put.assert_called()
        error_call_args = error_queue.put.call_args_list
        # Find the error call (not the initial -1 error)
        error_calls = [call for call in error_call_args if call[0][0][1] == "fail_task"]
        assert len(error_calls) >= 1
        assert "Execution failed" in error_calls[0][0][0][2]

    def test_device_worker_streaming_error(self, mock_queues):
        """Test error handling for streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create streaming request that will fail
        failing_stream = MockCompletionRequest("stream_fail", stream=True)

        task_queue.get.side_effect = [failing_stream, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Mock async that raises exception
        async def mock_run_async_fail(requests):
            raise Exception("Streaming failed")

        fresh_device_runner._run_async = mock_run_async_fail

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify error was queued
        error_queue.put.assert_called()
        error_call_args = error_queue.put.call_args_list
        # Find the streaming error
        error_calls = [
            call for call in error_call_args if call[0][0][1] == "stream_fail"
        ]
        assert len(error_calls) >= 1
        assert "Streaming failed" in error_calls[0][0][0][2]

    def test_device_worker_non_streaming_no_response(self, mock_queues):
        """Test handling when non-streaming request returns no response"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create request
        request = MockCompletionRequest("no_response", stream=False)

        task_queue.get.side_effect = [request, None]

        # Create fresh device runner that returns None
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        fresh_device_runner.run.return_value = None

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify error was queued
        error_queue.put.assert_called()
        error_call_args = error_queue.put.call_args_list
        error_calls = [
            call for call in error_call_args if call[0][0][1] == "no_response"
        ]
        assert len(error_calls) >= 1
        assert "No response generated" in error_calls[0][0][0][2]

    def test_device_worker_keyboard_interrupt(self, mock_queues):
        """Test graceful shutdown on KeyboardInterrupt"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Mock queue to raise KeyboardInterrupt
        task_queue.get.side_effect = KeyboardInterrupt()

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                # Should not raise exception
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify warning was logged
        mock_logger.warning.assert_any_call(
            "Worker worker_0 interrupted - shutting down"
        )


# Pytest fixtures for module-level setup
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    mock_logger.reset_mock()
    mock_device_runner.reset_mock()

    # Reset device runner defaults
    mock_device_runner.set_device.return_value = Mock()
    mock_device_runner.warmup = Mock(return_value=asyncio.Future())
    mock_device_runner.warmup.return_value.set_result(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
