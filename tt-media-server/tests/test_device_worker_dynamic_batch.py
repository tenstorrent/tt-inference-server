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

from domain.completion_request import CompletionRequest


def create_test_request(
    task_id: str, prompt: str = "test prompt", stream: bool = False
) -> CompletionRequest:
    """Create a CompletionRequest with explicit task_id for testing."""
    request = CompletionRequest(prompt=prompt, stream=stream)
    request._task_id = task_id
    return request


# Mock device runner and fabric
mock_device_runner = Mock()
mock_device_runner.set_device.return_value = Mock()
mock_device_runner.warmup = Mock(return_value=asyncio.Future())
mock_device_runner.warmup.return_value.set_result(None)
mock_device_runner.run.return_value = [Mock()]

# Mock logger at module level
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=mock_logger)

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

    def test_device_worker_streaming_with_memory_queue(self, mock_queues):
        """✅ Test streaming requests with SharedMemoryChunkQueue format"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(*args):
            results_captured.append(args)

        result_queue.put = capture_put

        # Create streaming request
        streaming_request = create_test_request("stream_task_1", stream=True)

        # Mock queue to return request then shutdown
        task_queue.get.side_effect = [streaming_request, None]

        # Create fresh device runner
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Create async generator that yields tuples (task_id, is_final, text)
        async def mock_async_generator():
            yield ("stream_task_1", 0, "token_0")
            yield ("stream_task_1", 0, "token_1")
            yield ("stream_task_1", 1, "[DONE]")  # ✅ is_final=1

        async def mock_run_async(requests):
            return mock_async_generator()

        fresh_device_runner._run_async = mock_run_async

        # ✅ Patch at correct location in device_worker_dynamic_batch
        with patch(
            "device_workers.device_worker_dynamic_batch.initialize_device_worker",
            return_value=(fresh_device_runner, asyncio.new_event_loop()),
        ):
            device_worker(
                "worker_0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        # ✅ Verify streaming chunks were put with (task_id, is_final, text) format
        assert len(results_captured) >= 3, (
            f"Expected at least 3 results, got {len(results_captured)}: {results_captured}"
        )
        assert results_captured[0] == ("stream_task_1", 0, "token_0"), (
            f"Got {results_captured[0]}"
        )
        assert results_captured[1] == ("stream_task_1", 0, "token_1"), (
            f"Got {results_captured[1]}"
        )
        assert results_captured[2] == ("stream_task_1", 1, "[DONE]"), (
            f"Got {results_captured[2]}"
        )

    def test_device_worker_non_streaming_request(self, mock_queues):
        """✅ Test handling of non-streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create non-streaming request
        regular_request = create_test_request("task_1", stream=False)

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
            "device_workers.device_worker_dynamic_batch.initialize_device_worker",
            return_value=(fresh_device_runner, asyncio.new_event_loop()),
        ):
            device_worker(
                "worker_0",
                task_queue,
                result_queue,
                warmup_signals_queue,
                error_queue,
            )

        # Verify result was queued
        assert len(results_captured) >= 1, (
            f"Expected at least 1 result, got {len(results_captured)}"
        )
        assert results_captured[0] == ("worker_0", "task_1", mock_response)

    def test_device_worker_streaming_error(self, mock_queues):
        """✅ Test error handling for streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create streaming request that will fail
        failing_stream = create_test_request("stream_fail", stream=True)

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

        # ✅ Patch at correct location
        with patch(
            "device_workers.device_worker_dynamic_batch.initialize_device_worker",
            return_value=(fresh_device_runner, asyncio.new_event_loop()),
        ):
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

        # Find the streaming error - error format is (worker_id, task_id, error_message)
        assert len(error_call_args) > 0, "Expected at least one error_queue.put call"

    def test_device_worker_keyboard_interrupt(self, mock_queues):
        """✅ Test graceful shutdown on KeyboardInterrupt"""
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

        # ✅ Patch at correct location
        with patch(
            "device_workers.device_worker_dynamic_batch.initialize_device_worker",
            return_value=(fresh_device_runner, asyncio.new_event_loop()),
        ):
            with patch(
                "device_workers.device_worker_dynamic_batch.TTLogger",
                return_value=mock_logger,
            ):
                # Should not raise exception
                device_worker(
                    "worker_0",
                    task_queue,
                    result_queue,
                    warmup_signals_queue,
                    error_queue,
                )

        # Verify the function was called and handled the interrupt gracefully
        # (no exception should be raised)
        assert True


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
