# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import sys
from multiprocessing import Process, Queue
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock external dependencies
sys.modules["ttnn"] = Mock()
# Note: tt_model_runners mocking is handled in conftest.py

# Mock settings
mock_settings = Mock()
mock_settings.device_ids = "(0),(1)"
mock_settings.max_queue_size = 10
mock_settings.max_batch_size = 1
mock_settings.use_queue_per_worker = True
mock_settings.use_memory_queue = False
mock_settings.use_dynamic_batcher = True
mock_settings.new_device_delay_seconds = 0.1
mock_settings.max_worker_restart_count = 3
mock_settings.allow_deep_reset = False
mock_settings.worker_check_sleep_timeout = 0.5
mock_settings.reset_device_command = "echo 'reset'"
mock_settings.reset_device_sleep_time = 0.1
sys.modules["config.settings"] = Mock()
sys.modules["config.settings"].get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"].Settings = Mock()

# Mock decorators and logger
sys.modules["utils.decorators"] = Mock()
sys.modules["utils.decorators"].log_execution_time = (
    lambda *args, **kwargs: lambda func: func
)
mock_logger = Mock()
sys.modules["utils.logger"] = Mock()
sys.modules["utils.logger"].TTLogger = Mock(return_value=mock_logger)

# Import module under test after mocking dependencies
from model_services.scheduler import Scheduler


def create_mock_queue():
    """Helper to create a mock queue with common methods"""
    queue = Mock(spec=Queue)
    queue.put = Mock()
    queue.get = Mock()
    queue.full = Mock(return_value=False)
    queue.close = Mock()
    queue.join_thread = Mock()
    return queue


def create_mock_lock():
    """Helper to create a mock lock with context manager support"""
    lock = MagicMock()
    lock.__enter__ = MagicMock(return_value=lock)
    lock.__exit__ = MagicMock(return_value=None)
    return lock


class TestScheduler:
    """Test cases for the Scheduler class"""

    @pytest.fixture
    def mock_queues(self):
        """Create mock queues for testing"""
        return (
            create_mock_queue(),  # task_queue
            create_mock_queue(),  # result_queue
            create_mock_queue(),  # warmup_signals_queue
            create_mock_queue(),  # error_queue
        )

    @pytest.fixture
    def mock_process(self):
        """Create a mock process for testing"""
        process = Mock(spec=Process)
        process.join = Mock()
        process.is_alive = Mock(return_value=False)
        process.terminate = Mock()
        process.kill = Mock()
        return process

    @pytest.fixture
    def mock_locks(self):
        """Create mock locks for testing"""
        return create_mock_lock(), create_mock_lock()

    @pytest.fixture
    def mock_future(self):
        """Create a mock asyncio future"""
        future = AsyncMock()
        future.set_result = Mock()
        future.set_exception = Mock()
        future.cancelled = Mock(return_value=False)
        future.done = Mock(return_value=False)
        future.cancel = Mock()
        return future

    @pytest.fixture
    def scheduler(self, mock_queues, mock_locks):
        """Create a scheduler instance with mocked components"""
        warmup_signals_queue, task_queue, result_queue, error_queue = mock_queues

        with patch("multiprocessing.Queue") as mock_queue_constructor, patch(
            "threading.Lock"
        ) as mock_lock_constructor:
            mock_queue_constructor.side_effect = [
                warmup_signals_queue,
                task_queue,
                result_queue,
                error_queue,
            ]

            lock_sequence = list(mock_locks)
            mock_lock_constructor.side_effect = (
                lambda: lock_sequence.pop(0) if lock_sequence else create_mock_lock()
            )

            return Scheduler()

    def test_initialization(self, scheduler):
        """Test scheduler initialization"""
        # Verify initial state
        assert not scheduler.is_ready
        # assert scheduler.worker_count == 2  # From the mock_settings.device_ids
        assert scheduler.task_queue is not None
        assert scheduler.result_queues_by_worker is not None
        assert scheduler.warmup_signals_queue is not None
        assert scheduler.error_queue is not None
        assert len(scheduler.workers_to_open) == 2  # Based on device_ids "(0),(1)"
        assert scheduler.worker_info == {}
        assert scheduler.listener_running
        assert scheduler.device_warmup_listener_running
        assert scheduler.monitor_running
        assert scheduler.result_queues == {}

        # Verify logger was initialized
        mock_logger.info.assert_not_called()  # No logs yet

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_check_is_model_ready_when_ready(self, scheduler):
        """Test check_is_model_ready when model is ready"""
        scheduler.is_ready = True
        result = scheduler.check_is_model_ready()
        assert result

    def test_check_is_model_ready_when_not_ready(self, scheduler):
        """Test check_is_model_ready when model is not ready"""
        scheduler.is_ready = False

        with pytest.raises(Exception) as exc_info:
            scheduler.check_is_model_ready()

        assert "405" in str(exc_info.value) or "Model is not ready" in str(
            exc_info.value
        )

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_process_request_success(self, scheduler):
        """Test process_request when successful"""
        # Setup
        scheduler.is_ready = True
        mock_request = Mock()

        # Patch the task_queue.put method
        with patch.object(scheduler.task_queue, "put") as mock_put:
            # Execute
            scheduler.process_request(mock_request)

            # Verify
            mock_put.assert_called_once_with(mock_request, timeout=1.0)

    def test_process_request_queue_full(self, scheduler):
        """Test process_request when queue is full"""
        # Setup
        scheduler.is_ready = True
        mock_request = Mock()

        # Patch the task_queue.full method
        with patch.object(scheduler.task_queue, "full", return_value=True):
            # Execute and verify
            with pytest.raises(Exception) as exc_info:
                scheduler.process_request(mock_request)

            assert "429" in str(exc_info.value) or "Task queue is full" in str(
                exc_info.value
            )

    def test_process_request_queue_put_timeout(self, scheduler):
        """Test process_request when queue.put times out"""
        # Setup
        scheduler.is_ready = True
        mock_request = Mock()

        # Patch the task_queue.put method to raise an exception
        with patch.object(
            scheduler.task_queue, "put", side_effect=Exception("Timeout")
        ):
            # Execute and verify
            with pytest.raises(Exception) as exc_info:
                scheduler.process_request(mock_request)

            assert "429" in str(exc_info.value) or "Unable to queue request" in str(
                exc_info.value
            )

    def test_process_request_not_ready(self, scheduler):
        """Test process_request when model is not ready"""
        # Setup
        scheduler.is_ready = False
        mock_request = Mock()

        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler.process_request(mock_request)

        assert "405" in str(exc_info.value) or "Model is not ready" in str(
            exc_info.value
        )

    @patch("asyncio.create_task")
    @patch("model_services.scheduler.Process")
    def test_start_workers(
        self, mock_process_constructor, mock_create_task, scheduler, mock_process
    ):
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        """Test start_workers method"""
        # Setup
        mock_process_constructor.return_value = mock_process
        scheduler.worker_count = 2  # Should create 2 workers

        # Execute
        scheduler.start_workers()

        # Verify tasks were created
        # result_listener, device_warmup_listener, error_listener, _start_workers_in_sequence
        assert mock_create_task.call_count == 4

        # Verify log message - use scheduler's logger directly
        scheduler.logger.info.assert_any_call("Workers to start: 2")

    @pytest.mark.asyncio
    async def test_result_listener(self, scheduler):
        """Test the result_listener method"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # Setup test data
        test_worker_id = "worker_0"
        test_task_id = "test_task"
        test_data = b"test_data"

        # Create mock result queue that will be returned from result_queues_by_worker
        mock_result_queue_by_worker = Mock()
        mock_result_queue_by_worker.get_nowait = Mock(
            side_effect=[
                (test_worker_id, test_task_id, test_data),  # First call returns result
                None,  # Subsequent calls return None (queue empty)
            ]
        )

        # Create mock response queue where results will be put
        mock_response_queue = AsyncMock()

        # Setup scheduler state
        scheduler.result_queues_by_worker = {0: mock_result_queue_by_worker}
        scheduler.result_queues = {test_task_id: mock_response_queue}

        # Run result_listener for one iteration
        # We'll set listener_running to False after getting the result
        scheduler.listener_running = True

        # Create a task that runs result_listener and stops it after first iteration
        async def run_with_timeout():
            # Give it a short time to process one result
            await asyncio.sleep(0.01)
            scheduler.listener_running = False

        # Run listener and timeout task concurrently
        await asyncio.gather(scheduler.result_listener(), run_with_timeout())

        # Verify result was put into the response queue
        mock_response_queue.put.assert_called_once_with(test_data)

        # Verify listener is stopped
        assert not scheduler.listener_running

        # Verify log message
        mock_logger.info.assert_any_call("Result listener stopped")

    @pytest.mark.asyncio
    async def test_error_listener(self, scheduler):
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        """Test the error_listener method"""

        # Setup test data
        test_worker_id = "worker_0"
        test_task_id = "test_task"
        test_error = "Test error message"

        # Setup scheduler state with worker info and result queues
        scheduler.worker_info = {test_worker_id: {"error_count": 0}}
        mock_result_queue = AsyncMock()
        scheduler.result_queues = {test_task_id: mock_result_queue}

        # Mock the error_queue.get to return error tuples
        mock_error_queue = Mock()
        mock_error_queue.get = Mock(
            side_effect=[
                (test_worker_id, test_task_id, test_error),  # First error
                (test_worker_id, None, None),  # Shutdown signal
            ]
        )
        scheduler.error_queue = mock_error_queue

        # Run error_listener with timeout to prevent infinite loop
        async def run_with_timeout():
            await asyncio.sleep(0.05)
            scheduler.listener_running = False

        scheduler.listener_running = True
        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = [
                (test_worker_id, test_task_id, test_error),
                (test_worker_id, None, None),
            ]
            await asyncio.gather(scheduler.error_listener(), run_with_timeout())

        # Verify error count was incremented twice
        assert scheduler.worker_info[test_worker_id]["error_count"] == 2

        # Verify error was put into the response queue as an Exception
        mock_result_queue.put.assert_called_once()
        put_arg = mock_result_queue.put.call_args[0][0]
        assert isinstance(put_arg, Exception)
        assert test_error in str(put_arg)

        # Verify listener is stopped
        assert not scheduler.listener_running

        # Verify log messages
        mock_logger.error.assert_any_call(
            f"Error in worker {test_task_id}: {test_error}"
        )
        mock_logger.info.assert_any_call("Error listener stopped")

    @pytest.mark.asyncio
    async def test_device_warmup_listener(self, scheduler):
        """Test the device_warmup_listener method"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # Setup test data
        test_device_id = "0"

        # Setup worker_info with the device
        scheduler.worker_info = {test_device_id: {"is_ready": False}}

        # Mock asyncio.to_thread to return device_id then None (shutdown signal)
        # We need to provide multiple returns because the while loop calls it multiple times
        mock_to_thread = AsyncMock(side_effect=[test_device_id, None])

        # Mock asyncio.create_task to avoid creating actual tasks
        mock_create_task = MagicMock()

        scheduler.device_warmup_listener_running = True

        with patch("model_services.scheduler.asyncio.to_thread", mock_to_thread), patch(
            "model_services.scheduler.asyncio.create_task", mock_create_task
        ):
            # Run the listener - it will exit when it gets None
            await scheduler.device_warmup_listener()

        # Verify device is tracked as ready
        assert scheduler.worker_info[test_device_id]["is_ready"]
        assert "ready_time" in scheduler.worker_info[test_device_id]
        assert scheduler.is_ready

        # Verify asyncio.to_thread was called
        assert mock_to_thread.call_count >= 2

        # Verify log messages
        mock_logger.info.assert_any_call(f"Device {test_device_id} is warmed up")
        mock_logger.info.assert_any_call(
            "First device warmed up, starting worker health monitor"
        )
        mock_logger.info.assert_any_call("Device warmup listener is done")

    def test_stop_workers(self, scheduler):
        """Test stop_workers method"""
        # Setup
        mock_process1 = Mock(spec=Process)
        mock_process1.join = Mock()
        mock_process1.is_alive = Mock(
            return_value=True
        )  # Worker is alive, should be joined

        mock_process2 = Mock(spec=Process)
        mock_process2.join = Mock()
        mock_process2.is_alive = Mock(
            return_value=True
        )  # Worker is alive, should be joined

        scheduler.worker_info = {
            "worker_0": {"process": mock_process1},
            "worker_1": {"process": mock_process2},
        }
        scheduler.is_ready = True
        scheduler.monitor_running = True
        scheduler.monitor_task_ref = None

        # Create mock result queues by worker
        mock_result_queue_0 = Mock()
        mock_result_queue_0.put = Mock()
        mock_result_queue_0.close = Mock()
        mock_result_queue_0.join_thread = Mock()

        mock_result_queue_1 = Mock()
        mock_result_queue_1.put = Mock()
        mock_result_queue_1.close = Mock()
        mock_result_queue_1.join_thread = Mock()

        scheduler.result_queues_by_worker = {
            0: mock_result_queue_0,
            1: mock_result_queue_1,
        }

        # Patch queue methods
        with patch.object(scheduler.task_queue, "put") as mock_task_put, patch.object(
            scheduler.task_queue, "close"
        ) as mock_task_close, patch.object(
            scheduler.task_queue, "join_thread"
        ) as mock_task_join, patch.object(
            scheduler.warmup_signals_queue, "put"
        ) as mock_warmup_put, patch.object(
            scheduler.warmup_signals_queue, "close"
        ) as mock_warmup_close, patch.object(
            scheduler.warmup_signals_queue, "join_thread"
        ) as mock_warmup_join, patch.object(
            scheduler.error_queue, "put"
        ) as mock_error_put, patch.object(
            scheduler.error_queue, "close"
        ) as mock_error_close, patch.object(
            scheduler.error_queue, "join_thread"
        ) as mock_error_join:
            # Make is_alive return False after first call so join doesn't hang
            mock_process1.is_alive.side_effect = [True, False]
            mock_process2.is_alive.side_effect = [True, False]

            # Execute
            scheduler.stop_workers()

            # Verify status change
            assert not scheduler.is_ready
            assert not scheduler.monitor_running

            # Verify shutdown signals were sent to workers
            assert mock_task_put.call_count == 2

            # Verify listeners were stopped
            assert not scheduler.listener_running
            assert not scheduler.device_warmup_listener_running

            # Verify workers were joined
            assert mock_process1.join.call_count >= 1
            assert mock_process2.join.call_count >= 1

            # Verify shutdown signals sent to result queues
            mock_result_queue_0.put.assert_called()
            mock_result_queue_1.put.assert_called()
            mock_error_put.assert_called()
            mock_warmup_put.assert_called()

            # Verify queues were closed
            mock_task_close.assert_called_once()
            mock_result_queue_0.close.assert_called_once()
            mock_result_queue_1.close.assert_called_once()
            mock_warmup_close.assert_called_once()
            mock_error_close.assert_called_once()

            mock_task_join.assert_called_once()
            mock_result_queue_0.join_thread.assert_called_once()
            mock_result_queue_1.join_thread.assert_called_once()
            mock_warmup_join.assert_called_once()
            mock_error_join.assert_called_once()

            # Verify worker_info was cleared
            assert len(scheduler.worker_info) == 0

    def test_close_queues(self, scheduler):
        """Test _close_queues method"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # Reset mock_logger to avoid accumulated calls from previous tests
        mock_logger.reset_mock()

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

    @pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
    def test_calculate_worker_count(self, scheduler):
        """Test _calculate_worker_count method with valid settings"""
        # The method uses self.settings which is already mocked
        # Execute
        result = scheduler._calculate_worker_count()

        # Verify - should return 2 based on mock_settings.device_ids = "(0),(1)"
        assert result == 2

    def test_calculate_worker_count_error(self, scheduler):
        """Test _calculate_worker_count method with invalid settings"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # Setup - make device_ids an object without replace method to trigger exception
        original_device_ids = scheduler.settings.device_ids
        scheduler.settings.device_ids = None

        try:
            # Execute and verify
            with pytest.raises(Exception) as exc_info:
                scheduler._calculate_worker_count()

            assert "500" in str(
                exc_info.value
            ) or "Workers cannot be initialized" in str(exc_info.value)
            mock_logger.error.assert_called()
        finally:
            # Restore original value to avoid affecting subsequent tests
            scheduler.settings.device_ids = original_device_ids

    def test_get_max_queue_size(self, scheduler):
        """Test _get_max_queue_size method with valid settings"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # The method uses self.settings which is already mocked with max_queue_size = 10
        # Execute
        result = scheduler._get_max_queue_size()

        # Verify
        assert result == 10

    def test_get_max_queue_size_error(self, scheduler):
        """Test _get_max_queue_size method with invalid settings"""
        pytest.skip("Disabled - causes test isolation issues with module-level mocking")
        # Setup - change settings to have invalid max_queue_size
        scheduler.settings.max_queue_size = 0

        # Execute and verify
        with pytest.raises(Exception) as exc_info:
            scheduler._get_max_queue_size()

        assert "500" in str(exc_info.value) or "Max queue size not provided" in str(
            exc_info.value
        )
        mock_logger.error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
