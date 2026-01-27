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
mock_settings.max_batch_delay_time_ms = 0.01

mock_settings_module = Mock()
mock_settings_module.settings = mock_settings
mock_settings_module.Settings = Mock(return_value=mock_settings)
mock_settings_module.get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"] = mock_settings_module

sys.modules["telemetry.telemetry_client"] = Mock()
sys.modules["telemetry.telemetry_client"].get_telemetry_client = Mock()

sys.modules["utils.torch_utils"] = Mock()
sys.modules["utils.torch_utils"].set_torch_thread_limits = Mock()

sys.modules["utils.device_manager"] = Mock()


class MockImageGenerateRequest:
    def __init__(self, task_id, prompt="test prompt", num_inference_steps=30):
        self._task_id = task_id
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.stream = False  # Default to non-streaming


sys.modules["domain.image_generate_request"] = Mock()
sys.modules[
    "domain.image_generate_request"
].ImageGenerateRequest = MockImageGenerateRequest

mock_device_runner = Mock()
mock_device_runner.set_device.return_value = Mock()
mock_device_runner.warmup = Mock(return_value=asyncio.Future())
mock_device_runner.warmup.return_value.set_result(None)
mock_device_runner.run.return_value = [Mock(), Mock()]

mock_image_manager = Mock()
mock_image_manager.convert_image_to_bytes.return_value = b"fake_image_bytes"
sys.modules["utils.image_manager"] = Mock()
sys.modules["utils.image_manager"].ImageManager.return_value = mock_image_manager

mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger.return_value = mock_logger

from device_workers.device_worker import device_worker


class WorkerExitException(Exception):
    """Exception used to exit the device_worker loop in tests."""
    pass


def create_get_many_side_effect(return_values):
    """Create a side_effect function that returns values then raises WorkerExitException.
    
    The device_worker runs in a `while True` loop that only does `continue` on None/empty,
    it never breaks. So we need to raise an exception to exit the loop in tests.
    """
    iterator = iter(return_values)

    def side_effect(*args, **kwargs):
        try:
            return next(iterator)
        except StopIteration:
            # Raise exception to break out of the while True loop
            raise WorkerExitException("Test complete - exiting worker loop")

    return side_effect


@pytest.fixture
def mock_queues():
    """Create mock queues for testing"""
    task_queue = Mock()
    task_queue.put = Mock()
    task_queue.get = Mock()
    task_queue.get_nowait = Mock()
    task_queue.get_many = Mock()

    result_queue = Mock()
    result_queue.put = Mock()
    result_queue.put_many = Mock()

    warmup_signals_queue = Mock()
    warmup_signals_queue.put = Mock()
    warmup_signals_queue._closed = False

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

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = "mock_device"

        mock_get_device_runner = Mock(return_value=fresh_device_runner)

        # Use the helper to create side_effect - raises WorkerExitException to exit loop
        task_queue.get_many.side_effect = create_get_many_side_effect([])

        mock_loop = Mock()
        mock_loop.run_until_complete = Mock(return_value=None)
        mock_loop.close = Mock()

        with patch(
            "device_workers.worker_utils.get_device_runner", mock_get_device_runner
        ):
            with patch("device_workers.worker_utils.get_telemetry_client", Mock()):
                with patch("asyncio.new_event_loop", return_value=mock_loop):
                    with patch("asyncio.set_event_loop", Mock()):
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        mock_get_device_runner.assert_called_once_with("worker_0", 1)
        fresh_device_runner.set_device.assert_called_once()
        mock_loop.run_until_complete.assert_called_once()
        warmup_signals_queue.put.assert_called_once_with("worker_0", timeout=2.0)

    def test_device_worker_initialization_failure(self, mock_queues):
        """Test worker initialization failure"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

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

        error_queue.put.assert_called_once_with(
            ("worker_0", -1, "Device initialization failed")
        )

    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_successful_inference(
        self, mock_timer, mock_queues, mock_requests
    ):
        """Test successful inference processing"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        # First call returns requests, then raises WorkerExitException to exit loop
        task_queue.get_many.side_effect = create_get_many_side_effect(
            [mock_requests]
        )

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = [Mock(), Mock()]

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        assert fresh_device_runner.run.call_count == 1
        call_args = fresh_device_runner.run.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]._task_id == "task_1"
        assert call_args[1]._task_id == "task_2"

        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()

        result_queue.put_many.assert_called_once()
        call_args = result_queue.put_many.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0][0] == "worker_0"
        assert call_args[0][1] == "task_1"
        assert call_args[1][0] == "worker_0"
        assert call_args[1][1] == "task_2"

    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_inference_error(
        self, mock_timer, mock_queues, mock_requests
    ):
        """Test inference error handling"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        task_queue.get_many.side_effect = create_get_many_side_effect(
            [mock_requests]
        )

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.side_effect = Exception("Inference failed")

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        assert error_queue.put.call_count == 2
        calls = error_queue.put.call_args_list
        task_ids = [c[0][0][1] for c in calls]
        assert "task_1" in task_ids
        assert "task_2" in task_ids

    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_no_responses_generated(
        self, mock_timer, mock_queues, mock_requests
    ):
        """Test handling when no responses are generated"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        task_queue.get_many.side_effect = create_get_many_side_effect(
            [mock_requests]
        )

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = []

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        assert error_queue.put.call_count == 2
        calls = error_queue.put.call_args_list
        task_ids = [c[0][0][1] for c in calls]
        assert "task_1" in task_ids
        assert "task_2" in task_ids

    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_streaming_request(self, mock_timer, mock_queues):
        """Test handling of streaming requests"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        results_captured = []

        def capture_put(item):
            results_captured.append(item)

        result_queue.put = capture_put

        streaming_request = MockImageGenerateRequest("stream_task_1")
        streaming_request.stream = True

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        task_queue.get_many.side_effect = create_get_many_side_effect(
            [[streaming_request]]
        )

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        async def mock_warmup():
            return None

        fresh_device_runner.warmup = mock_warmup

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
                with pytest.raises(WorkerExitException):
                    device_worker(
                        "worker_0",
                        task_queue,
                        result_queue,
                        warmup_signals_queue,
                        error_queue,
                    )

        assert len(results_captured) == 3
        assert results_captured[0] == ("worker_0", "stream_task_1", "chunk_1")
        assert results_captured[1] == ("worker_0", "stream_task_1", "chunk_2")
        assert results_captured[2] == ("worker_0", "stream_task_1", "chunk_3")

        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()

    @patch("device_workers.device_worker.threading.Timer")
    def test_device_worker_empty_batch_continues(self, mock_timer, mock_queues):
        """Test that empty batch causes continue"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        request = MockImageGenerateRequest("task_1")

        # First call returns empty list, second returns valid request, then raises to exit
        task_queue.get_many.side_effect = create_get_many_side_effect(
            [[], [request]]
        )

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = [Mock()]

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        assert task_queue.get_many.call_count == 3
        assert fresh_device_runner.run.call_count == 1


class TestDeviceWorkerIntegration:
    """Integration tests for device worker components"""

    @patch("device_workers.device_worker.threading.Timer")
    def test_timeout_handler_creation(self, mock_timer, mock_queues):
        """Test that timeout handler is created correctly"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_request = MockImageGenerateRequest("timeout_task", "test prompt", 30)
        task_queue.get_many.side_effect = create_get_many_side_effect(
            [[mock_request]]
        )

        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()
        fresh_device_runner.run.return_value = [Mock()]

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        mock_timer.assert_called_once()
        timeout_callback = mock_timer.call_args[0][1]
        assert timeout_callback is not None
        mock_timer_instance.cancel.assert_called_once()

    @patch("device_workers.device_worker.threading.Timer")
    def test_timeout_triggered(self, mock_timer, mock_queues):
        """Test timeout behavior when inference takes too long"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues

        mock_request = MockImageGenerateRequest("timeout_task", "test prompt", 30)
        task_queue.get_many.side_effect = create_get_many_side_effect(
            [[mock_request]]
        )

        mock_timer_instance = Mock()
        mock_timer_instance.callback = None

        def create_timer(timeout, callback):
            mock_timer_instance.callback = callback
            return mock_timer_instance

        mock_timer.side_effect = create_timer

        fresh_device_runner = Mock()
        fresh_device_runner.set_device.return_value = Mock()
        fresh_device_runner.close_device = Mock()

        def slow_inference(*args, **kwargs):
            if mock_timer_instance.callback:
                mock_timer_instance.callback()
            return [Mock()]

        fresh_device_runner.run.side_effect = slow_inference

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
                        with pytest.raises(WorkerExitException):
                            device_worker(
                                "worker_0",
                                task_queue,
                                result_queue,
                                warmup_signals_queue,
                                error_queue,
                            )

        assert error_queue.put.call_count >= 1
        calls = error_queue.put.call_args_list
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


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    mock_logger.reset_mock()
    mock_device_runner.reset_mock()
    mock_image_manager.reset_mock()

    mock_device_runner.set_device.return_value = Mock()
    mock_device_runner.warmup = Mock(return_value=asyncio.Future())
    mock_device_runner.warmup.return_value.set_result(None)
    mock_device_runner.run.return_value = [Mock(), Mock()]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=10"])
