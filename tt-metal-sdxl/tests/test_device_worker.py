# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
import asyncio
import threading
from multiprocessing import Queue

# Mock all external dependencies before importing
sys.modules['ttnn'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.tt_unet'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.tt_embedding'] = Mock()
sys.modules['models.experimental.stable_diffusion_xl_base.tt.sdxl_utility'] = Mock()
sys.modules['tt_model_runners.sdxl_runner'] = Mock()

# Mock config settings
mock_settings = Mock()
mock_settings.max_batch_size = 4
mock_settings.num_inference_steps = 30
sys.modules['config.settings'] = Mock()
sys.modules['config.settings'].settings = mock_settings

# Mock domain objects
class MockImageGenerateRequest:
    def __init__(self, task_id, prompt="test prompt", num_inference_step=30):
        self._task_id = task_id
        self.prompt = prompt
        self.num_inference_step = num_inference_step

sys.modules['domain.image_generate_request'] = Mock()
sys.modules['domain.image_generate_request'].ImageGenerateRequest = MockImageGenerateRequest

# Mock device runner and fabric
mock_device_runner = Mock()
mock_device_runner.get_device.return_value = Mock()
mock_device_runner.load_model = Mock(return_value=asyncio.Future())
mock_device_runner.load_model.return_value.set_result(None)
mock_device_runner.run_inference.return_value = [Mock(), Mock()]  # Mock images

mock_runner_fabric = Mock()
mock_runner_fabric.get_device_runner = Mock(return_value=mock_device_runner)
sys.modules['tt_model_runners.base_device_runner'] = Mock()
sys.modules['tt_model_runners.runner_fabric'] = Mock()
sys.modules['tt_model_runners.runner_fabric'].get_device_runner = mock_runner_fabric.get_device_runner

# Mock image manager
mock_image_manager = Mock()
mock_image_manager.convert_image_to_bytes.return_value = b"fake_image_bytes"
sys.modules['utils.image_manager'] = Mock()
sys.modules['utils.image_manager'].ImageManager.return_value = mock_image_manager

# Mock logger
mock_logger = Mock()
sys.modules['utils.logger'] = Mock()
sys.modules['utils.logger'].TTLogger.return_value = mock_logger

# Now import the module under test
from model_services.device_worker import device_worker, get_greedy_batch

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
        mock_device_runner.get_device.return_value = "mock_device"
        
        # Mock asyncio.run to avoid actually running the coroutine
        async def mock_coro(*args, **kwargs):
            return None
            
        mock_load_model_future = mock_coro()
        mock_device_runner.load_model = Mock(return_value=mock_load_model_future)
        
        # Mock the device runner factory
        mock_get_device_runner = Mock(return_value=mock_device_runner)
        
        # Mock immediate shutdown to avoid full execution
        mock_get_batch = Mock(return_value=[None])
        
        # Apply patches
        with patch('model_services.device_worker.get_device_runner', mock_get_device_runner):
            with patch('model_services.device_worker.get_greedy_batch', mock_get_batch):
                with patch('asyncio.run', Mock(return_value=None)) as mock_asyncio_run:
                    device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
                    
                    # Verify initialization calls
                    mock_get_device_runner.assert_called_once_with("worker_0")
                    mock_device_runner.get_device.assert_called_once()
                    
                    # Check the asyncio.run was called (not load_model directly since it's passed to asyncio.run)
                    mock_asyncio_run.assert_called_once()
                    
                    # Verify the warmup signal was sent
                    warmup_signals_queue.put.assert_called_once_with("worker_0")
    
    def test_device_worker_initialization_failure(self, mock_queues):
        """Test worker initialization failure"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Mock initialization failure
        mock_get_device_runner = Mock(side_effect=Exception("Device initialization failed"))
        
        with patch('model_services.device_worker.get_device_runner', mock_get_device_runner):
            device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
        
        # Verify error handling
        error_queue.put.assert_called_once_with(("worker_0", "Device initialization failed"))
    
    @patch('model_services.device_worker.get_greedy_batch')
    @patch('model_services.device_worker.threading.Timer')
    def test_device_worker_successful_inference(self, mock_timer, mock_get_batch, mock_queues, mock_requests):
        """Test successful inference processing"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [mock_requests, [None]]  # Process batch then shutdown
        
        # Reset mocks to avoid initialization side effects
        if hasattr(mock_runner_fabric.get_device_runner, 'side_effect'):
            mock_runner_fabric.get_device_runner.side_effect = None
        
        device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
        
        # Verify inference was called
        mock_device_runner.run_inference.assert_called_once_with(
            ["prompt 1", "prompt 2"], 
            mock_settings.num_inference_steps
        )
        
        # Verify timer was started and cancelled
        mock_timer_instance.start.assert_called_once()
        mock_timer_instance.cancel.assert_called_once()
        
        # Verify results were queued
        assert result_queue.put.call_count == 2
        result_queue.put.assert_any_call(("task_1", b"fake_image_bytes"))
        result_queue.put.assert_any_call(("task_2", b"fake_image_bytes"))
    
    @patch('model_services.device_worker.get_greedy_batch')
    @patch('model_services.device_worker.threading.Timer')
    def test_device_worker_inference_error(self, mock_timer, mock_get_batch, mock_queues, mock_requests):
        """Test inference error handling"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [mock_requests, [None]]
        mock_device_runner.run_inference.side_effect = Exception("Inference failed")
        
        # Reset initialization mock
        if hasattr(mock_runner_fabric.get_device_runner, 'side_effect'):
            mock_runner_fabric.get_device_runner.side_effect = None
        
        device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
        
        # Verify error handling
        assert error_queue.put.call_count == 2  # One for each request
        error_queue.put.assert_any_call(("task_1", "Worker worker_0 inference error: Inference failed"))
        error_queue.put.assert_any_call(("task_2", "Worker worker_0 inference error: Inference failed"))
    
    @patch('model_services.device_worker.get_greedy_batch')
    @patch('model_services.device_worker.threading.Timer')
    def test_device_worker_no_images_generated(self, mock_timer, mock_get_batch, mock_queues, mock_requests):
        """Test handling when no images are generated"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Setup mocks
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        mock_get_batch.side_effect = [mock_requests, [None]]
        mock_device_runner.run_inference.return_value = []  # No images
        
        # Reset initialization mock
        if hasattr(mock_runner_fabric.get_device_runner, 'side_effect'):
            mock_runner_fabric.get_device_runner.side_effect = None
        
        device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
        
        # Verify error handling for no images
        assert error_queue.put.call_count == 2
        error_queue.put.assert_any_call(("task_1", "No images generated"))
        error_queue.put.assert_any_call(("task_2", "No images generated"))


class TestGetGreedyBatch:
    """Test cases for get_greedy_batch function"""
    
    @pytest.fixture
    def mock_queue(self):
        """Create a mock queue for testing"""
        # Create a mock with the required methods explicitly defined
        mock = Mock()
        mock.get = Mock()
        mock.get_nowait = Mock()
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
        mock_queue.get_nowait.side_effect = Exception("Queue empty")
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert len(result) == 1
        assert result[0]._task_id == "task_1"
        mock_queue.get.assert_called_once()
    
    def test_get_greedy_batch_multiple_items(self, mock_queue, mock_requests):
        """Test getting multiple items in batch"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.get_nowait.side_effect = [mock_requests[1], mock_requests[2], Exception("Queue empty")]
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert len(result) == 3
        assert result[0]._task_id == "task_1"
        assert result[1]._task_id == "task_2"
        assert result[2]._task_id == "task_3"
    
    def test_get_greedy_batch_max_batch_size_limit(self, mock_queue, mock_requests):
        """Test that batch size is limited by max_batch_size"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.get_nowait.side_effect = [mock_requests[1], Exception("Queue empty")]
        
        result = get_greedy_batch(mock_queue, 2)  # Limit to 2 items
        
        assert len(result) == 2
        assert mock_queue.get_nowait.call_count == 1  # Only called once due to limit
    
    def test_get_greedy_batch_shutdown_signal(self, mock_queue):
        """Test handling shutdown signal (None)"""
        mock_queue.get.return_value = None
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert result == [None]
        mock_queue.get.assert_called_once()
    
    def test_get_greedy_batch_shutdown_signal_in_nowait(self, mock_queue, mock_requests):
        """Test handling shutdown signal in get_nowait"""
        mock_queue.get.return_value = mock_requests[0]
        mock_queue.get_nowait.side_effect = [None]  # Shutdown signal
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert len(result) == 2
        assert result[0]._task_id == "task_1"
        assert result[1] is None
    
    def test_get_greedy_batch_keyboard_interrupt(self, mock_queue):
        """Test handling KeyboardInterrupt"""
        mock_queue.get.side_effect = KeyboardInterrupt("Test interrupt")
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert result == [None]
        mock_logger.warning.assert_called_with("KeyboardInterrupt received - shutting down gracefully")
    
    def test_get_greedy_batch_general_exception(self, mock_queue):
        """Test handling general exceptions"""
        mock_queue.get.side_effect = Exception("Connection lost")
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert result == [None]
        mock_logger.error.assert_called_with("Error getting first item from queue: Connection lost")
    
    def test_get_greedy_batch_empty_queue_after_first_item(self, mock_queue):
        """Test behavior when queue becomes empty after first item"""
        mock_queue.get.return_value = MockImageGenerateRequest("task_1")
        mock_queue.get_nowait.side_effect = Exception("Queue empty")
        
        result = get_greedy_batch(mock_queue, 4)
        
        assert len(result) == 1
        assert result[0]._task_id == "task_1"


class TestDeviceWorkerIntegration:
    """Integration tests for device worker components"""
    
    def test_timeout_handler_creation(self, mock_queues):
        """Test that timeout handler is created correctly"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Create a test scenario that just verifies the timer is created
        with patch('model_services.device_worker.get_greedy_batch') as mock_get_batch:
            with patch('model_services.device_worker.threading.Timer') as mock_timer:
                # Mock the request
                mock_request = MockImageGenerateRequest("timeout_task", "test prompt", 30)
                mock_get_batch.side_effect = [[mock_request], [None]]
                
                # Mock timer
                mock_timer_instance = Mock()
                mock_timer.return_value = mock_timer_instance
                
                # Reset initialization mock
                if hasattr(mock_runner_fabric.get_device_runner, 'side_effect'):
                    mock_runner_fabric.get_device_runner.side_effect = None
                
                # Make sure inference completes successfully so timeout doesn't actually trigger
                mock_device_runner.run_inference.return_value = [Mock()]
                
                # Run the worker
                device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
                
                # Verify timer was created with correct timeout calculation (10 + inference_steps*2)
                mock_timer.assert_called_once()
                
                # Extract the timeout callback for verification
                timeout_callback = mock_timer.call_args[0][1]
                self.assertIsNotNone(timeout_callback)
                
                # Verify the timer was cancelled (successful inference cancels the timer)
                mock_timer_instance.cancel.assert_called_once()
    
    def test_timeout_triggered(self, mock_queues):
        """Test timeout behavior when inference takes too long"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Create a test scenario where timeout is triggered
        with patch('model_services.device_worker.get_greedy_batch') as mock_get_batch:
            with patch('model_services.device_worker.threading.Timer') as mock_timer:
                # Mock the request
                mock_request = MockImageGenerateRequest("timeout_task", "test prompt", 30)
                mock_get_batch.side_effect = [[mock_request], [None]]
                
                # Setup timer to capture and execute the callback
                def create_timer(timeout, callback):
                    timer_mock = Mock()
                    # Store the callback for later execution
                    timer_mock.callback = callback
                    return timer_mock
                
                mock_timer.side_effect = create_timer
                
                # Set up run_inference to be slow - simulate by side effect
                def slow_inference(*args, **kwargs):
                    # Execute the timeout callback before returning
                    timer_mock = mock_timer.return_value
                    timer_mock.callback()
                    # Then return the result (too late)
                    return [Mock()]
                
                mock_device_runner.run_inference.side_effect = slow_inference
                
                # Reset initialization mock
                if hasattr(mock_runner_fabric.get_device_runner, 'side_effect'):
                    mock_runner_fabric.get_device_runner.side_effect = None
                
                # Run the worker
                device_worker("worker_0", task_queue, result_queue, warmup_signals_queue, error_queue)
                
                # Verify error was reported
                error_queue.put.assert_called()
                error_msg = error_queue.put.call_args[0][0]
                self.assertEqual(error_msg[0], "timeout_task")  # Task ID
                self.assertIn("timed out", error_msg[1])  # Error message


# Pytest fixtures for module-level setup
@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks before each test"""
    mock_logger.reset_mock()
    mock_device_runner.reset_mock()
    mock_runner_fabric.reset_mock()
    mock_image_manager.reset_mock()
    
    # Reset device runner defaults
    mock_device_runner.get_device.return_value = Mock()
    mock_device_runner.load_model = Mock(return_value=asyncio.Future())
    mock_device_runner.load_model.return_value.set_result(None)
    mock_device_runner.run_inference.return_value = [Mock(), Mock()]
    mock_runner_fabric.get_device_runner.return_value = mock_device_runner


if __name__ == "__main__":
    pytest.main([__file__])
