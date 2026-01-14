# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import sys
from unittest.mock import Mock, patch

import pytest

# Mock all external dependencies before importing
# Note: tt_model_runners mocking is handled in conftest.py
sys.modules["ttnn"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_unet"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.tt_embedding"] = Mock()
sys.modules["models.experimental.stable_diffusion_xl_base.tt.sdxl_utility"] = Mock()

# Mock config settings - must be done before any imports that use settings
mock_settings = Mock()
mock_settings.max_batch_size = 4
mock_settings.default_throttle_level = "5"  # Must be string for os.environ
mock_settings.enable_telemetry = False
mock_settings.is_galaxy = False
mock_settings.device_mesh_shape = (1, 1)
mock_settings.request_processing_timeout_seconds = 100
mock_settings.max_batch_delay_time_ms = 0.01  # Small timeout for tests

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
class MockImageGenerateRequest:
    def __init__(self, task_id, prompt="test prompt", num_inference_steps=30):
        self._task_id = task_id
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps


sys.modules["domain.image_generate_request"] = Mock()
sys.modules[
    "domain.image_generate_request"
].ImageGenerateRequest = MockImageGenerateRequest

# Mock device runner and fabric
mock_device_runner = Mock()
mock_device_runner.set_device.return_value = Mock()
mock_device_runner.warmup = Mock(return_value=asyncio.Future())
mock_device_runner.warmup.return_value.set_result(None)
mock_device_runner.run.return_value = [Mock(), Mock()]

# Note: tt_model_runners mocking (including base_device_runner) is handled in conftest.py

# Mock image manager
mock_image_manager = Mock()
mock_image_manager.convert_image_to_bytes.return_value = b"fake_image_bytes"
sys.modules["utils.image_manager"] = Mock()
sys.modules["utils.image_manager"].ImageManager.return_value = mock_image_manager

# Mock logger
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger.return_value = mock_logger

# Now import the module under test
from device_workers.device_worker import device_worker, get_greedy_batch


# Module level fixtures that can be used by all test classes
@pytest.fixture
def mock_queues():
    """Create mock queues for testing"""
    task_queue = Mock()
    task_queue.put = Mock()
    task_queue.get = Mock()
    task_queue.get_nowait = Mock()

    result_queue = Mock()
    result_queue.put = Mock()

    warmup_signals_queue = Mock()
    warmup_signals_queue.put = Mock()
    warmup_signals_queue._closed = False  # Must be False to allow put operations

    error_queue = Mock()
    error_queue.put = Mock()

    return task_queue, result_queue, warmup_signals_queue, error_queue


class TestDeviceWorker:
    """Test cases for device_worker function"""

    @pytest.fixture
    def mock_requests(self):
        """Create mock image generation requests"""
        return [
            MockImageGenerateRequest("task_1", "prompt 1", 25),
            MockImageGenerateRequest("task_2", "prompt 2", 30),
        ]

    def test_device_worker_initialization_success(self, mock_queues):
        """Test successful worker initialization"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create fresh mocks for this test
        mock_device_runner = Mock()
        mock_device_runner.set_device.return_value = "mock_device"

        # Mock the device runner factory
        mock_get_device_runner = Mock(return_value=mock_device_runner)

        # Mock immediate shutdown to avoid full execution
        mock_get_batch = Mock(return_value=[None])

        # Mock the event loop to avoid actually running async code
        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value=None)
        mock_loop.close = Mock()

        # Apply patches
        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("device_workers.device_worker.get_greedy_batch", mock_get_batch):
                with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                    with patch("asyncio.new_event_loop", return_value=mock_loop):
                        with patch("asyncio.set_event_loop", Mock()):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

                            # Verify initialization calls
                            mock_get_device_runner.assert_called_once_with(
                                "worker_0", 1
                            )
                            mock_device_runner.set_device.assert_called_once()

                            # Verify the event loop was used for warmup
                            mock_loop.run_until_complete.assert_called_once()

                            # Verify the warmup signal was sent
                            warmup_signals_queue.put.assert_called_once_with(
                                "worker_0", timeout=2.0
                            )

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

    @patch("device_workers.device_worker.get_greedy_batch")
    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_successful_inference(
        self, mock_timer, mock_get_batch, mock_queues, mock_requests
    ):
        """Test successful inference processing"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [
            mock_requests,
            [None],
        ]  # Process batch then shutdown

        # Create fresh device runner for this test
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = [
            Mock(),
            Mock(),
        ]  # Return mock images

        # Mock the event loop to avoid actually running async code
        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value=None)
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                with patch("asyncio.new_event_loop", return_value=mock_loop):
                    with patch("asyncio.set_event_loop", Mock()):
                        device_worker(
                            "worker_0",
                            task_queue,
                            result_queue,
                            warmup_signals_queue,
                            error_queue,
                        )

        # Verify inference was called with the request objects
        assert fresh_device_runner.run.call_count == 1
        call_args = fresh_device_runner.run.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]._task_id == "task_1"
        assert call_args[1]._task_id == "task_2"

        # Verify timer was started and cancelled
        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()

        # Verify results were queued - format is (worker_id, task_id, response)
        assert result_queue.put.call_count == 2
        # The responses come from fresh_device_runner.run.return_value which is [Mock(), Mock()]
        calls = result_queue.put.call_args_list
        # Extract the tuples passed to put()
        first_call_args = calls[0][0][0]  # (worker_id, task_id, response)
        second_call_args = calls[1][0][0]  # (worker_id, task_id, response)

        assert first_call_args[0] == "worker_0"  # worker_id
        assert first_call_args[1] == "task_1"  # task_id
        assert second_call_args[0] == "worker_0"  # worker_id
        assert second_call_args[1] == "task_2"  # task_id

    @patch("device_workers.device_worker.get_greedy_batch")
    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_inference_error(
        self, mock_timer, mock_get_batch, mock_queues, mock_requests
    ):
        """Test inference error handling"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [mock_requests, [None]]

        # Create fresh device runner for this test to avoid side_effect from previous test
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        # Don't set warmup as a Mock with return_value - let it be auto-mocked
        # This avoids asyncio.run issues
        fresh_device_runner.run.side_effect = Exception("Inference failed")

        # Mock the event loop to avoid actually running async code
        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value=None)
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                with patch("asyncio.new_event_loop", return_value=mock_loop):
                    with patch("asyncio.set_event_loop", Mock()):
                        device_worker(
                            "worker_0",
                            task_queue,
                            result_queue,
                            warmup_signals_queue,
                            error_queue,
                        )

        # Verify error handling - when run raises an exception,
        # the code calls error_queue.put once per request in the batch
        assert error_queue.put.call_count == 2  # One for each request
        # Verify both tasks got error messages
        calls = error_queue.put.call_args_list
        task_ids = [c[0][0][1] for c in calls]
        assert "task_1" in task_ids
        assert "task_2" in task_ids

    @patch("device_workers.device_worker.get_greedy_batch")
    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_no_images_generated(
        self, mock_timer, mock_get_batch, mock_queues, mock_requests
    ):
        """Test handling when no images are generated"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [mock_requests, [None]]

        # Create fresh device runner for this test
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = []  # No images

        # Mock the event loop to avoid actually running async code
        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value=None)
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner",
            return_value=fresh_device_runner,
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                with patch("asyncio.new_event_loop", return_value=mock_loop):
                    with patch("asyncio.set_event_loop", Mock()):
                        device_worker(
                            "worker_0",
                            task_queue,
                            result_queue,
                            warmup_signals_queue,
                            error_queue,
                        )

        # Verify error handling for no images
        assert error_queue.put.call_count == 2
        # Verify both tasks got error messages
        calls = error_queue.put.call_args_list
        task_ids = [c[0][0][1] for c in calls]
        assert "task_1" in task_ids
        assert "task_2" in task_ids

    @patch("device_workers.device_worker.get_greedy_batch")
    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_streaming_request(
        self, mock_timer, mock_get_batch, mock_queues
    ):
        """Test handling of streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results directly since Mock queue might not work with async
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create a streaming request
        streaming_request = MockImageGenerateRequest("stream_task_1")
        streaming_request.stream = True

        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [[streaming_request], [None]]

        # Create fresh device runner for this test
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        # Make warmup an async function that returns immediately
        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        # Create an async generator for streaming results
        async def mock_async_generator():
            yield "chunk_1"
            yield "chunk_2"
            yield "chunk_3"

        # _run_async is a coroutine that returns an async generator when awaited
        async def mock_run_async(requests):
            # Return the generator directly
            return mock_async_generator()

        fresh_device_runner._run_async = mock_run_async

        # Also add is_request_batchable for get_greedy_batch
        fresh_device_runner.is_request_batchable = lambda req: True

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
        assert len(results_captured) == 3  # 3 chunks

        # Verify each chunk was sent with correct format
        assert results_captured[0] == ("worker_0", "stream_task_1", "chunk_1")
        assert results_captured[1] == ("worker_0", "stream_task_1", "chunk_2")
        assert results_captured[2] == ("worker_0", "stream_task_1", "chunk_3")

        # Verify timer was started and cancelled
        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()

    @patch("device_workers.device_worker.get_greedy_batch")
    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_mixed_streaming_and_regular_requests(
        self, mock_timer, mock_get_batch, mock_queues
    ):
        """Test handling batch with both streaming and regular requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Track results directly
        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        # Create mixed requests
        streaming_request = MockImageGenerateRequest("stream_task")
        streaming_request.stream = True
        regular_request = MockImageGenerateRequest("regular_task")

        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [[streaming_request, regular_request], [None]]

        # Create fresh device runner for this test
        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        # Make warmup an async function that returns immediately
        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

        regular_response = Mock()
        fresh_device_runner.run.return_value = [regular_response]  # For regular request

        # Also add is_request_batchable
        fresh_device_runner.is_request_batchable = lambda req: True

        # Create an async generator for streaming results
        async def mock_async_generator():
            yield "stream_chunk"

        # _run_async is a coroutine that returns an async generator when awaited
        async def mock_run_async(requests):
            # Return the generator directly
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

        # Verify both streaming and regular results were queued
        assert len(results_captured) == 2

        # Check that we got both types of results
        # First should be streaming chunk, second should be regular response
        assert results_captured[0] == ("worker_0", "stream_task", "stream_chunk")
        assert results_captured[1] == ("worker_0", "regular_task", regular_response)

        # Verify timer was started and cancelled
        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()


class TestGetGreedyBatch:
    """Test cases for get_greedy_batch function"""

    @pytest.fixture
    def mock_queue(self):
        """Create a mock queue for testing"""
        # Create a mock with the required methods explicitly defined
        mock = Mock()
        mock.get = Mock()
        mock.peek_next = Mock()
        mock.return_item = Mock()
        mock.put = Mock()
        return mock

    @pytest.fixture
    def mock_requests(self):
        """Create mock requests for testing"""
        return [
            MockImageGenerateRequest("task_1"),
            MockImageGenerateRequest("task_2"),
            MockImageGenerateRequest("task_3"),
        ]

    def test_get_greedy_batch_single_item(self, mock_queue):
        """Test getting a single item batch"""
        mock_queue.get.return_value = MockImageGenerateRequest("task_1")
        mock_queue.peek_next.side_effect = Exception("Queue empty")

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert len(result) == 1
        assert result[0]._task_id == "task_1"
        mock_queue.get.assert_called_once()

    def test_get_greedy_batch_multiple_items(self, mock_queue, mock_requests):
        """Test getting multiple items in batch"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.peek_next.side_effect = [
            mock_requests[1],
            mock_requests[2],
            Exception("Queue empty"),
        ]

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert len(result) == 3
        assert result[0]._task_id == "task_1"
        assert result[1]._task_id == "task_2"
        assert result[2]._task_id == "task_3"

    def test_get_greedy_batch_max_batch_size_limit(self, mock_queue, mock_requests):
        """Test that batch size is limited by max_batch_size"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.peek_next.side_effect = [mock_requests[1], Exception("Queue empty")]

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 2, batching_predicate)  # Limit to 2 items

        assert len(result) == 2
        assert mock_queue.peek_next.call_count == 1  # Only called once due to limit

    def test_get_greedy_batch_shutdown_signal(self, mock_queue):
        """Test handling shutdown signal (None)"""
        mock_queue.get.return_value = None

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert result == [None]
        mock_queue.get.assert_called_once()

    def test_get_greedy_batch_shutdown_signal_in_peek_next(
        self, mock_queue, mock_requests
    ):
        """Test handling shutdown signal in peek_next"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.peek_next.side_effect = [None]  # Shutdown signal

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert len(result) == 2
        assert result[0]._task_id == "task_1"
        assert result[1] is None

    def test_get_greedy_batch_keyboard_interrupt(self, mock_queue):
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        """Test handling KeyboardInterrupt"""
        mock_queue.get.side_effect = KeyboardInterrupt("Test interrupt")

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert result == [None]
        mock_logger.warning.assert_called_with(
            "KeyboardInterrupt received - shutting down gracefully"
        )

    def test_get_greedy_batch_general_exception(self, mock_queue):
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        """Test handling general exceptions"""
        mock_queue.get.side_effect = Exception("Connection lost")

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert result == [None]
        mock_logger.error.assert_called_with(
            "Error getting first item from queue: Connection lost"
        )

    def test_get_greedy_batch_empty_queue_after_first_item(self, mock_queue):
        """Test behavior when queue becomes empty after first item"""
        mock_queue.get.return_value = MockImageGenerateRequest("task_1")
        mock_queue.peek_next.side_effect = Exception("Queue empty")

        def batching_predicate(item, batch):
            return True

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert len(result) == 1
        assert result[0]._task_id == "task_1"

    def test_get_greedy_batch_batching_predicate_rejects(
        self, mock_queue, mock_requests
    ):
        """Test that batching_predicate can reject items and they get returned"""
        mock_queue.get.return_value = mock_requests[0]
        # Second item will be rejected by predicate
        mock_queue.peek_next.return_value = mock_requests[1]

        # Predicate that only allows first item
        def batching_predicate(item, batch):
            return len(batch) < 1

        result = get_greedy_batch(mock_queue, 4, batching_predicate)

        assert len(result) == 1
        assert result[0]._task_id == "task_1"
        # Verify the rejected item was returned to the queue
        mock_queue.return_item.assert_called_once_with(mock_requests[1])


class TestDeviceWorkerIntegration:
    """Integration tests for device worker components"""

    def test_timeout_handler_creation(self, mock_queues):
        """Test that timeout handler is created correctly"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create a test scenario that just verifies the timer is created
        with patch("device_workers.device_worker.get_greedy_batch") as mock_get_batch:
            with patch("device_workers.device_worker.threading.Timer") as mock_timer:
                # Mock the request
                mock_request = MockImageGenerateRequest(
                    "timeout_task", "test prompt", 30
                )
                mock_get_batch.side_effect = [[mock_request], [None]]

                # Mock timer
                mock_timer_instance = Mock()
                mock_timer.return_value = mock_timer_instance

                # Create fresh device runner
                fresh_device_runner = Mock()
                fresh_device_runner.set_device.return_value = Mock()
                fresh_device_runner.close_device = Mock()
                fresh_device_runner.run.return_value = [Mock()]

                # Mock the event loop to avoid actually running async code
                mock_loop = Mock()
                mock_loop.run_until_complete = Mock(return_value=None)
                mock_loop.close = Mock()

                with patch(
                    "device_workers.worker_utils.get_device_runner",
                    return_value=fresh_device_runner,
                ):
                    with patch(
                        "device_workers.worker_utils.get_telemetry_client", Mock()
                    ):
                        with patch("asyncio.new_event_loop", return_value=mock_loop):
                            with patch("asyncio.set_event_loop", Mock()):
                                # Run the worker
                                device_worker(
                                    "worker_0",
                                    task_queue,
                                    result_queue,
                                    warmup_signals_queue,
                                    error_queue,
                                )

                # Verify timer was created
                mock_timer.assert_called_once()

                # Extract the timeout callback for verification
                timeout_callback = mock_timer.call_args[0][1]
                assert timeout_callback is not None

                # Verify the timer was cancelled (successful inference cancels the timer)
                mock_timer_instance.cancel.assert_called_once()

    def test_timeout_triggered(self, mock_queues):
        """Test timeout behavior when inference takes too long"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        # Create a test scenario where timeout is triggered
        with patch("device_workers.device_worker.get_greedy_batch") as mock_get_batch:
            with patch("device_workers.device_worker.threading.Timer") as mock_timer:
                # Mock the request
                mock_request = MockImageGenerateRequest(
                    "timeout_task", "test prompt", 30
                )
                mock_get_batch.side_effect = [[mock_request], [None]]

                # Setup timer - we'll manually trigger it
                mock_timer_instance = Mock()
                mock_timer_instance.callback = None

                def create_timer(timeout, callback):
                    # Store the callback so we can trigger it
                    mock_timer_instance.callback = callback
                    return mock_timer_instance

                mock_timer.side_effect = create_timer

                # Create fresh device runner with slow inference
                fresh_device_runner = Mock()
                fresh_device_runner.set_device.return_value = Mock()
                fresh_device_runner.close_device = Mock()

                # Set up run to trigger timeout and then return
                def slow_inference(*args, **kwargs):
                    # Trigger the timeout callback
                    if mock_timer_instance.callback:
                        mock_timer_instance.callback()
                    # Then return the result (too late)
                    return [Mock()]

                fresh_device_runner.run.side_effect = slow_inference

                # Mock the event loop to avoid actually running async code
                mock_loop = Mock()
                mock_loop.run_until_complete = Mock(return_value=None)
                mock_loop.close = Mock()

                with patch(
                    "device_workers.worker_utils.get_device_runner",
                    return_value=fresh_device_runner,
                ):
                    with patch(
                        "device_workers.worker_utils.get_telemetry_client", Mock()
                    ):
                        with patch("asyncio.new_event_loop", return_value=mock_loop):
                            with patch("asyncio.set_event_loop", Mock()):
                                # Run the worker
                                device_worker(
                                    "worker_0",
                                    task_queue,
                                    result_queue,
                                    warmup_signals_queue,
                                    error_queue,
                                )

                # Verify error was reported for timeout
                assert error_queue.put.call_count >= 1
                # Find the timeout error message
                calls = error_queue.put.call_args_list
                # Should have at least one call with timeout error
                found_timeout_error = False
                for c in calls:
                    if len(c[0]) > 0 and len(c[0][0]) >= 3:
                        error_tuple = c[0][0]
                        if "ran out of time" in error_tuple[2]:
                            found_timeout_error = True
                            assert error_tuple[0] == "worker_0"
                            assert error_tuple[1] == "timeout_task"
                            break
                assert found_timeout_error, f"No timeout error found in calls: {calls}"


# Pytest fixtures for module-level setup
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    mock_logger.reset_mock()
    mock_device_runner.reset_mock()
    mock_image_manager.reset_mock()

    # Reset device runner defaults
    mock_device_runner.set_device.return_value = Mock()
    mock_device_runner.warmup = Mock(return_value=asyncio.Future())
    mock_device_runner.warmup.return_value.set_result(None)
    mock_device_runner.run.return_value = [Mock(), Mock()]


if __name__ == "__main__":
    pytest.main([__file__])
