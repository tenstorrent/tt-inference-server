# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import sys
from multiprocessing import Process, Queue
from threading import Lock

# Mock external dependencies
sys.modules['ttnn'] = Mock()
sys.modules['tt_model_runners.sdxl_runner'] = Mock()

# Mock settings
mock_settings = Mock()
mock_settings.device_ids = "0,1"
mock_settings.max_queue_size = 10
mock_settings.num_inference_steps = 30
sys.modules['config.settings'] = Mock()
sys.modules['config.settings'].get_settings = Mock(return_value=mock_settings)
sys.modules['config.settings'].Settings = Mock()

# Mock helpers and logger
sys.modules['utils.helpers'] = Mock()
sys.modules['utils.helpers'].log_execution_time = lambda x: lambda func: func
mock_logger = Mock()
sys.modules['utils.logger'] = Mock()
sys.modules['utils.logger'].TTLogger = Mock(return_value=mock_logger)

# Import module under test after mocking dependencies
from model_services.scheduler import Scheduler


class TestScheduler:
    """Test cases for the Scheduler class"""
    
    @pytest.fixture
    def mock_queues(self):
        """Create mock queues for testing"""
        task_queue = Mock(spec=Queue)
        task_queue.put = Mock()
        task_queue.get = Mock()
        task_queue.full = Mock(return_value=False)
        task_queue.close = Mock()
        task_queue.join_thread = Mock()
        
        result_queue = Mock(spec=Queue)
        result_queue.put = Mock()
        result_queue.get = Mock()
        result_queue.close = Mock()
        result_queue.join_thread = Mock()
        
        warmup_signals_queue = Mock(spec=Queue)
        warmup_signals_queue.put = Mock()
        warmup_signals_queue.get = Mock()
        warmup_signals_queue.close = Mock()
        warmup_signals_queue.join_thread = Mock()
        
        error_queue = Mock(spec=Queue)
        error_queue.put = Mock()
        error_queue.get = Mock()
        error_queue.close = Mock()
        error_queue.join_thread = Mock()
        
        return task_queue, result_queue, warmup_signals_queue, error_queue
    
    @pytest.fixture
    def mock_process(self):
        """Create a mock process for testing"""
        process = Mock(spec=Process)
        process.start = Mock()
        process.join = Mock()
        process.is_alive = Mock(return_value=False)
        process.terminate = Mock()
        process.kill = Mock()
        return process
    
    @pytest.fixture
    def mock_locks(self):
        """Create mock locks for testing"""
        # Create mocks without spec to avoid issues with magic methods
        ready_devices_lock = MagicMock()
        # Configure the context manager behavior
        ready_devices_lock.__enter__ = MagicMock(return_value=ready_devices_lock)
        ready_devices_lock.__exit__ = MagicMock(return_value=None)
        ready_devices_lock.acquire = MagicMock()
        ready_devices_lock.release = MagicMock()
        
        result_futures_lock = MagicMock()
        # Configure the context manager behavior
        result_futures_lock.__enter__ = MagicMock(return_value=result_futures_lock)
        result_futures_lock.__exit__ = MagicMock(return_value=None)
        result_futures_lock.acquire = MagicMock()
        result_futures_lock.release = MagicMock()
        
        return ready_devices_lock, result_futures_lock
    
    @pytest.fixture
    def mock_asyncio(self):
        """Create mock asyncio components for testing"""
        mock_future = AsyncMock()
        mock_future.set_result = Mock()
        mock_future.set_exception = Mock()
        mock_future.cancelled = Mock(return_value=False)
        mock_future.done = Mock(return_value=False)
        mock_future.cancel = Mock()
        
        mock_task = AsyncMock()
        
        return mock_future, mock_task
    
    @pytest.fixture
    def scheduler(self, mock_queues, mock_locks):
        """Create a scheduler instance with mocked components"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        ready_devices_lock, result_futures_lock = mock_locks
        
        with patch('multiprocessing.Queue') as mock_queue_constructor:
            mock_queue_constructor.side_effect = [
                warmup_signals_queue, task_queue, result_queue, error_queue
            ]
            with patch('threading.Lock') as mock_lock_constructor:
                mock_lock_constructor.side_effect = [ready_devices_lock, result_futures_lock]
                scheduler = Scheduler()
                
        return scheduler
    
    def test_initialization(self, scheduler, mock_queues):
        """Test scheduler initialization"""
        task_queue, result_queue, warmup_signals_queue, error_queue = mock_queues
        
        # Verify initial state
        assert scheduler.isReady == False
        assert scheduler.worker_count == 2  # From the mock_settings.device_ids
        assert scheduler.task_queue == task_queue
        assert scheduler.result_queue == result_queue
        assert scheduler.warmup_signals_queue == warmup_signals_queue
        assert scheduler.error_queue == error_queue
        assert scheduler.workers == []
        assert scheduler.ready_devices == []
        assert scheduler.listener_running == True
        assert scheduler.device_warmup_listener_running == True
        
        # Verify logger was initialized
        mock_logger.info.assert_not_called()  # No logs yet
    
    def test_check_is_model_ready_when_ready(self, scheduler):
        """Test check_is_model_ready when model is ready"""
        scheduler.isReady = True
        result = scheduler.check_is_model_ready()
        assert result == True
    
    def test_check_is_model_ready_when_not_ready(self, scheduler):
        """Test ccheck_is_model_ready when model is not ready"""
        scheduler.isReady = False
        
        with pytest.raises(Exception) as exc_info:
            scheduler.check_is_model_ready()
        
        assert "405" in str(exc_info.value) or "Model is not ready" in str(exc_info.value)
    
    def test_process_request_success(self, scheduler):
        """Test process_request when successful"""
        # Setup
        scheduler.isReady = True
        mock_request = Mock()
        
        # Execute
        scheduler.process_request(mock_request)
        
        # Verify
        scheduler.task_queue.put.assert_called_once_with(mock_request, timeout=1.0)
    
    def test_process_request_queue_full(self, scheduler):
        """Test process_request when queue is full"""
        # Setup
        scheduler.isReady = True
        scheduler.task_queue.full.return_value = True
        mock_request = Mock()
        
        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler.process_request(mock_request)
        
        assert "429" in str(exc_info.value) or "Task queue is full" in str(exc_info.value)
        scheduler.task_queue.put.assert_not_called()
    
    def test_process_request_queue_put_timeout(self, scheduler):
        """Test process_request when queue.put times out"""
        # Setup
        scheduler.isReady = True
        scheduler.task_queue.put.side_effect = Exception("Timeout")
        mock_request = Mock()
        
        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler.process_request(mock_request)
        
        assert "429" in str(exc_info.value) or "Unable to queue request" in str(exc_info.value)
    
    def test_process_request_not_ready(self, scheduler):
        """Test process_request when model is not ready"""
        # Setup
        scheduler.isReady = False
        mock_request = Mock()
        
        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler.process_request(mock_request)
        
        assert "405" in str(exc_info.value) or "Model is not ready" in str(exc_info.value)
        scheduler.task_queue.put.assert_not_called()
    
    @patch('asyncio.create_task')
    @patch('model_services.scheduler.Process')
    def test_start_workers(self, mock_process_constructor, mock_create_task, scheduler, mock_process):
        """Test start_workers method"""
        # Setup
        mock_process_constructor.return_value = mock_process
        scheduler.worker_count = 2  # Should create 2 workers
        
        # Execute
        scheduler.start_workers()
        
        # Verify tasks were created
        assert mock_create_task.call_count == 3  # result_listener, device_warmup_listener, error_listener
        
        # Verify workers were created and started
        assert mock_process_constructor.call_count == 2
        assert mock_process.start.call_count == 2
        assert len(scheduler.workers) == 2
        assert scheduler.workers[0] == mock_process
        assert scheduler.workers[1] == mock_process
        
        # Verify log message
        mock_logger.info.assert_any_call("Workers started: 1")
    
    @patch('asyncio.to_thread')
    async def test_result_listener(self, mock_to_thread, scheduler, mock_asyncio):
        """Test the result_listener method"""
        mock_future, _ = mock_asyncio
        
        # Setup test data
        test_task_id = "test_task"
        test_image = b"test_image_data"
        
        # Mock result_queue.get to return our test data then None to stop the loop
        async_result1 = AsyncMock()
        async_result1.return_value = (test_task_id, test_image)
        
        async_result2 = AsyncMock()
        async_result2.return_value = (None, None)
        
        mock_to_thread.side_effect = [async_result1, async_result2]
        
        # Add a future to the result_futures dictionary
        scheduler.result_futures = {test_task_id: mock_future}
        
        # Execute
        await scheduler.result_listener()
        
        # Verify
        assert mock_to_thread.call_count == 2
        assert mock_to_thread.call_args_list[0][0][0] == scheduler.result_queue.get
        
        # Verify future was set with the result
        mock_future.set_result.assert_called_once_with(test_image)
        
        # Verify listener is stopped
        assert scheduler.listener_running == False
        
        # Verify shutdown signals
        scheduler.warmup_signals_queue.put.assert_called_with(None, timeout=1.0)
        
        # Verify log message
        mock_logger.info.assert_any_call("Result listener stopped")
    
    @patch('asyncio.to_thread')
    async def test_error_listener(self, mock_to_thread, scheduler, mock_asyncio):
        """Test the error_listener method"""
        mock_future, _ = mock_asyncio
        
        # Setup test data
        test_task_id = "test_task"
        test_error = "Test error message"
        
        # Mock error_queue.get to return our test data then None to stop the loop
        async_result1 = AsyncMock()
        async_result1.return_value = (test_task_id, test_error)
        
        async_result2 = AsyncMock()
        async_result2.return_value = (None, None)
        
        mock_to_thread.side_effect = [async_result1, async_result2]
        
        # Add a future to the result_futures dictionary
        scheduler.result_futures = {test_task_id: mock_future}
        
        # Execute
        await scheduler.error_listener()
        
        # Verify
        assert mock_to_thread.call_count == 2
        assert mock_to_thread.call_args_list[0][0][0] == scheduler.error_queue.get
        
        # Verify future was set with an exception
        mock_future.set_exception.assert_called_once()
        assert isinstance(mock_future.set_exception.call_args[0][0], Exception)
        assert test_error in str(mock_future.set_exception.call_args[0][0])
        
        # Verify listener is stopped
        assert scheduler.listener_running == False
        
        # Verify log messages
        mock_logger.error.assert_any_call(f"Error in worker {test_task_id}: {test_error}")
        mock_logger.info.assert_any_call("Error listener stopped")
    
    @patch('asyncio.to_thread')
    async def test_device_warmup_listener(self, mock_to_thread, scheduler):
        """Test the device_warmup_listener method"""
        # Setup test data
        test_device_id = "0"
        
        # Mock warmup_signals_queue.get to return our test data then None to stop the loop
        async_result1 = AsyncMock()
        async_result1.return_value = test_device_id
        
        async_result2 = AsyncMock()
        async_result2.return_value = None
        
        mock_to_thread.side_effect = [async_result1, async_result2]
        
        # Setup workers
        mock_worker = Mock()
        scheduler.workers = [mock_worker]
        
        # Execute
        await scheduler.device_warmup_listener()
        
        # Verify
        assert mock_to_thread.call_count == 2
        assert mock_to_thread.call_args_list[0][0][0] == scheduler.warmup_signals_queue.get
        
        # Verify device is tracked as ready
        assert test_device_id in scheduler.ready_devices
        assert scheduler.isReady == True
        
        # Verify log messages
        mock_logger.info.assert_any_call(f"Device {test_device_id} is warmed up")
        mock_logger.info.assert_any_call("Device warmup listener is done")
    
    @patch('model_services.scheduler.get_device_runner')
    def test_stop_workers(self, mock_get_device_runner, scheduler, mock_process):
        """Test stop_workers method"""
        # Setup
        scheduler.workers = [mock_process, mock_process]  # Two mock workers
        scheduler.isReady = True
        
        # Mock device
        mock_device = Mock()
        scheduler.main_device = mock_device
        
        # Mock runner
        mock_runner = Mock()
        mock_get_device_runner.return_value = mock_runner
        
        # Execute
        scheduler.stop_workers()
        
        # Verify status change
        assert scheduler.isReady == False
        
        # Verify shutdown signals were sent
        assert scheduler.task_queue.put.call_count == 2  # One for each worker
        scheduler.task_queue.put.assert_has_calls([call(None, timeout=2.0), call(None, timeout=2.0)])
        
        # Verify listeners were stopped
        assert scheduler.listener_running == False
        assert scheduler.device_warmup_listener_running == False
        
        # Verify workers were joined and terminated if needed
        assert mock_process.join.call_count == 2
        
        # Verify queues were closed
        assert scheduler.task_queue.close.called
        assert scheduler.result_queue.close.called
        assert scheduler.warmup_signals_queue.close.called
        assert scheduler.error_queue.close.called
        
        assert scheduler.task_queue.join_thread.called
        assert scheduler.result_queue.join_thread.called
        assert scheduler.warmup_signals_queue.join_thread.called
        assert scheduler.error_queue.join_thread.called
        
        # Verify device was closed
        mock_runner.close_device.assert_called_once_with(mock_device)
        
        # Verify workers list was cleared
        assert len(scheduler.workers) == 0
        
        # Verify log messages
        mock_logger.info.assert_any_call("Workers stopped")
        mock_logger.info.assert_any_call("Main device closed")
    
    def test_close_queues(self, scheduler):
        """Test _close_queues method"""
        # Setup
        mock_queue1 = Mock()
        mock_queue1.close = Mock()
        mock_queue1.join_thread = Mock()
        
        mock_queue2 = Mock()
        mock_queue2.close = Mock()
        mock_queue2.join_thread = Mock()
        
        mock_queue3 = Mock()
        mock_queue3.close = Mock(side_effect=Exception("Test error"))
        
        queues = [mock_queue1, mock_queue2, mock_queue3]
        
        # Execute
        scheduler._close_queues(queues)
        
        # Verify
        mock_queue1.close.assert_called_once()
        mock_queue1.join_thread.assert_called_once()
        
        mock_queue2.close.assert_called_once()
        mock_queue2.join_thread.assert_called_once()
        
        mock_queue3.close.assert_called_once()
        mock_queue3.join_thread.assert_not_called()
        
        # Verify log message
        mock_logger.info.assert_any_call("Queues (2) closed successfully")
        mock_logger.error.assert_called_once()  # For the error on mock_queue3
    
    def test_getWorkerCount(self, scheduler):
        """Test _getWorkerCount method with valid settings"""
        # Setup
        mock_settings = Mock()
        mock_settings.device_ids = "0,1,2"  # 3 devices
        
        # Execute
        result = scheduler._getWorkerCount(mock_settings)
        
        # Verify
        assert result == 3
    
    def test_getWorkerCount_error(self, scheduler):
        """Test _getWorkerCount method with invalid settings"""
        # Setup
        mock_settings = Mock()
        mock_settings.device_ids = ""  # No devices
        
        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler._getWorkerCount(mock_settings)
        
        assert "500" in str(exc_info.value) or "Workers cannot be initialized" in str(exc_info.value)
        mock_logger.error.assert_called_once()
    
    def test_get_max_queue_size(self, scheduler):
        """Test _get_max_queue_size method with valid settings"""
        # Setup
        mock_settings = Mock()
        mock_settings.max_queue_size = 20
        
        # Execute
        result = scheduler._get_max_queue_size(mock_settings)
        
        # Verify
        assert result == 20
    
    def test_get_max_queue_size_error(self, scheduler):
        """Test _get_max_queue_size method with invalid settings"""
        # Setup
        mock_settings = Mock()
        mock_settings.max_queue_size = 0  # Invalid size
        
        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler._get_max_queue_size(mock_settings)
        
        assert "500" in str(exc_info.value) or "Max queue size not provided" in str(exc_info.value)
        mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
