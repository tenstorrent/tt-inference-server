# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
sys.modules["config.settings"] = Mock()
sys.modules["config.settings"].get_settings = Mock(return_value=mock_settings)
sys.modules["config.settings"].Settings = Mock()

# Mock decorators and logger
sys.modules["utils.decorators"] = Mock()
sys.modules["utils.decorators"].log_execution_time = lambda *args, **kwargs: lambda func: func
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
        assert scheduler.worker_count == 2  # From the mock_settings.device_ids
        assert scheduler.task_queue is not None
        assert scheduler.result_queue is not None
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
        """Test start_workers method"""
        # Setup
        mock_process_constructor.return_value = mock_process
        scheduler.worker_count = 2  # Should create 2 workers

        # Execute
        scheduler.start_workers()

        # Verify tasks were created
        # result_listener, device_warmup_listener, error_listener, _start_workers_in_sequence
        assert mock_create_task.call_count == 4

        # Verify log message
        mock_logger.info.assert_any_call("Workers to start: 2")

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_result_listener(self, mock_to_thread, scheduler):
        """Test the result_listener method"""

        # Setup test data
        test_worker_id = "worker_0"
        test_task_id = "test_task"
        test_image = b"test_image_data"

        # Mock asyncio.to_thread to return values directly (not AsyncMock)
        # to_thread already handles the async part, so we just return the values
        mock_to_thread.side_effect = [
            (test_worker_id, test_task_id, test_image),
            (None, None, None),
        ]

        # Add a queue to the result_queues dictionary and worker_info
        mock_queue = AsyncMock()
        scheduler.result_queues = {test_task_id: mock_queue}
        scheduler.worker_info = {test_worker_id: {"restart_count": 1}}

        # Execute
        await scheduler.result_listener()

        # Verify
        assert mock_to_thread.call_count == 2
        assert mock_to_thread.call_args_list[0][0][0] == scheduler.result_queue.get

        # Verify result was put into the queue
        mock_queue.put.assert_called_once_with(test_image)

        # Verify listener is stopped
        assert not scheduler.listener_running

        # Verify worker restart count was reset
        assert scheduler.worker_info[test_worker_id]["restart_count"] == 0

        # Verify log message
        mock_logger.info.assert_any_call("Result listener stopped")

    @pytest.mark.asyncio
    @patch("asyncio.to_thread")
    async def test_error_listener(self, mock_to_thread, scheduler):
        """Test the error_listener method"""

        # Setup test data
        test_worker_id = "worker_0"
        test_task_id = "test_task"
        test_error = "Test error message"

        # Mock asyncio.to_thread to return values directly
        # Note: error_listener increments error_count BEFORE checking if task_id is None
        # So we need to send a valid worker_id even in the shutdown signal
        mock_to_thread.side_effect = [
            (test_worker_id, test_task_id, test_error),
            (
                test_worker_id,
                None,
                None,
            ),  # worker_id must be valid for error_count increment
        ]

        # Add a queue to the result_queues dictionary and worker_info
        mock_queue = AsyncMock()
        scheduler.result_queues = {test_task_id: mock_queue}
        scheduler.worker_info = {test_worker_id: {"error_count": 0}}

        # Execute
        await scheduler.error_listener()

        # Verify
        assert mock_to_thread.call_count == 2
        assert mock_to_thread.call_args_list[0][0][0] == scheduler.error_queue.get

        # Verify error was put into the queue as an Exception
        mock_queue.put.assert_called_once()
        put_arg = mock_queue.put.call_args[0][0]
        assert isinstance(put_arg, Exception)
        assert test_error in str(put_arg)

        # Verify listener is stopped
        assert not scheduler.listener_running

        # Verify worker error count incremented twice (once for error, once for shutdown signal)
        assert scheduler.worker_info[test_worker_id]["error_count"] == 2

        # Verify log messages
        mock_logger.error.assert_any_call(
            f"Error in worker {test_task_id}: {test_error}"
        )
        mock_logger.info.assert_any_call("Error listener stopped")

    @pytest.mark.asyncio
    @patch("asyncio.create_task")
    @patch("asyncio.to_thread")
    async def test_device_warmup_listener(
        self, mock_to_thread, mock_create_task, scheduler
    ):
        """Test the device_warmup_listener method"""
        # Setup test data
        test_device_id = "0"

        # Mock asyncio.to_thread to return values directly
        mock_to_thread.side_effect = [test_device_id, None]

        # Setup worker_info with the device
        scheduler.worker_info = {test_device_id: {"is_ready": False}}

        # Execute
        await scheduler.device_warmup_listener()

        # Verify
        assert mock_to_thread.call_count == 2
        assert (
            mock_to_thread.call_args_list[0][0][0] == scheduler.warmup_signals_queue.get
        )

        # Verify device is tracked as ready
        assert scheduler.worker_info[test_device_id]["is_ready"]
        assert "ready_time" in scheduler.worker_info[test_device_id]
        assert scheduler.is_ready

        # Verify monitor task was created
        mock_create_task.assert_called_once()

        # Verify log messages
        mock_logger.info.assert_any_call(f"Device {test_device_id} is warmed up")
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

        # Patch queue methods
        with patch.object(scheduler.task_queue, "put") as mock_task_put, patch.object(
            scheduler.task_queue, "close"
        ) as mock_task_close, patch.object(
            scheduler.task_queue, "join_thread"
        ) as mock_task_join, patch.object(
            scheduler.result_queue, "put"
        ) as mock_result_put, patch.object(
            scheduler.result_queue, "close"
        ) as mock_result_close, patch.object(
            scheduler.result_queue, "join_thread"
        ) as mock_result_join, patch.object(
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

            # Verify shutdown signals sent to listener queues
            mock_result_put.assert_called()
            mock_error_put.assert_called()
            mock_warmup_put.assert_called()

            # Verify queues were closed
            mock_task_close.assert_called_once()
            mock_result_close.assert_called_once()
            mock_warmup_close.assert_called_once()
            mock_error_close.assert_called_once()

            mock_task_join.assert_called_once()
            mock_result_join.assert_called_once()
            mock_warmup_join.assert_called_once()
            mock_error_join.assert_called_once()

            # Verify worker_info was cleared
            assert len(scheduler.worker_info) == 0

    def test_close_queues(self, scheduler):
        """Test _close_queues method"""
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
        # The method uses self.settings which is already mocked with max_queue_size = 10
        # Execute
        result = scheduler._get_max_queue_size()

        # Verify
        assert result == 10

    def test_get_max_queue_size_error(self, scheduler):
        """Test _get_max_queue_size method with invalid settings"""
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
