# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
sys.modules["utils.decorators"].log_execution_time = lambda *args, **kwargs: (
    lambda func: func
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
        mock_logger.reset_mock()

        with patch("multiprocessing.Queue") as mock_queue_constructor, patch(
            "threading.Lock"
        ) as mock_lock_constructor, patch(
            "model_services.scheduler.TTLogger", return_value=mock_logger
        ):
            mock_queue_constructor.side_effect = [
                warmup_signals_queue,
                task_queue,
                result_queue,
                error_queue,
            ]

            lock_sequence = list(mock_locks)
            mock_lock_constructor.side_effect = lambda: (
                lock_sequence.pop(0) if lock_sequence else create_mock_lock()
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
        # assert len(scheduler.workers_to_open) == 2  # Based on device_ids "(0),(1)"
        assert scheduler.worker_info == {}
        assert scheduler.listener_running
        assert scheduler.device_warmup_listener_running
        assert scheduler.monitor_running
        assert scheduler.result_queues == {}

        # Verify logger was used during init (_calculate_worker_count logs)
        assert mock_logger.info.call_count >= 1

    def test_check_is_model_ready_when_not_ready(self, scheduler):
        """Test check_is_model_ready when model is not ready"""
        scheduler.is_ready = False

        with pytest.raises(Exception) as exc_info:
            scheduler.check_is_model_ready()

        assert "405" in str(exc_info.value) or "Model is not ready" in str(
            exc_info.value
        )

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

    @patch("model_services.scheduler.Process")
    def test_start_worker_with_queue_index_uses_given_index(
        self, mock_process_cls, scheduler, mock_process
    ):
        """Test _start_worker with queue_index uses that index (restart path, lines 159-160)"""
        mock_process_cls.return_value = mock_process
        scheduler.result_queues_by_worker = {
            0: create_mock_queue(),
            1: create_mock_queue(),
        }
        scheduler.worker_info = {}

        scheduler._start_worker(worker_id="0", queue_index=1)

        assert scheduler.worker_info["0"]["queue_index"] == 1
        mock_process_cls.assert_called_once()

    @patch("model_services.scheduler.Process")
    def test_start_worker_without_queue_index_uses_len_worker_info(
        self, mock_process_cls, scheduler, mock_process
    ):
        """Test _start_worker without queue_index uses len(worker_info) (first start, line 162)"""
        mock_process_cls.return_value = mock_process
        scheduler.result_queues_by_worker = {
            0: create_mock_queue(),
            1: create_mock_queue(),
        }
        scheduler.worker_info = {}

        scheduler._start_worker(worker_id="0")

        assert scheduler.worker_info["0"]["queue_index"] == 0
        mock_process_cls.assert_called_once()

    @patch("model_services.scheduler.Process")
    def test_start_worker_uses_device_worker_for_non_dynamic(
        self, mock_process_cls, scheduler, mock_process
    ):
        """Test _start_worker routes to device_worker when not dynamic batcher."""
        from device_workers.device_worker import device_worker

        mock_process_cls.return_value = mock_process
        scheduler.result_queues_by_worker = {0: create_mock_queue()}
        scheduler.worker_info = {}
        scheduler.settings.model_runner = "mock"
        scheduler.settings.use_dynamic_batcher = False

        scheduler._start_worker(worker_id="0")

        call_args = mock_process_cls.call_args
        assert call_args.kwargs["target"] == device_worker

    @patch("model_services.scheduler.Process")
    def test_restart_worker_passes_existing_queue_index_to_start_worker(
        self, mock_process_cls, scheduler, mock_process
    ):
        """Test restart_worker passes existing queue_index to _start_worker (lines 225, 228)"""
        mock_process_cls.return_value = mock_process
        scheduler.result_queues_by_worker = {
            0: create_mock_queue(),
            1: create_mock_queue(),
        }
        old_process = Mock(spec=Process)
        old_process.is_alive = Mock(return_value=False)
        scheduler.worker_info["0"] = {
            "process": old_process,
            "restart_count": 0,
            "queue_index": 1,
            "error_count": 0,
        }

        scheduler.restart_worker("0")

        assert scheduler.worker_info["0"]["queue_index"] == 1
        mock_process_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_health_monitor_bumps_restart_count_when_restart_worker_raises(
        self, scheduler
    ):
        """Test worker_health_monitor logs and bumps restart_count when restart_worker raises (491-494, 498)"""
        dead_process = Mock(spec=Process)
        dead_process.is_alive = Mock(return_value=False)
        scheduler.worker_info["0"] = {
            "process": dead_process,
            "restart_count": 0,
            "queue_index": 0,
            "error_count": 0,
        }
        scheduler.is_ready = True
        scheduler.monitor_running = True
        first_sleep = True

        async def stop_after_first_iteration(timeout):
            nonlocal first_sleep
            if first_sleep:
                first_sleep = False
                scheduler.monitor_running = False

        with patch.object(
            scheduler, "restart_worker", side_effect=RuntimeError("Restart failed")
        ), patch(
            "model_services.scheduler.asyncio.sleep",
            side_effect=stop_after_first_iteration,
        ):
            await scheduler.worker_health_monitor()

        mock_logger.error.assert_any_call("Failed to restart worker 0: Restart failed")
        assert scheduler.worker_info["0"]["restart_count"] == 1

    @pytest.mark.asyncio
    async def test_worker_health_monitor_calls_deep_restart_when_restart_count_exceeded(
        self, scheduler
    ):
        """Test worker_health_monitor calls deep_restart_workers when restart_count >= max and allow_deep_reset (500-506)"""
        dead_process = Mock(spec=Process)
        dead_process.is_alive = Mock(return_value=False)
        scheduler.settings.allow_deep_reset = True
        scheduler.settings.max_worker_restart_count = 3
        scheduler.worker_info["0"] = {
            "process": dead_process,
            "restart_count": 3,
            "queue_index": 0,
            "error_count": 0,
        }
        scheduler.is_ready = True
        scheduler.monitor_running = True
        first_sleep = True

        async def stop_after_first_iteration(timeout):
            nonlocal first_sleep
            if first_sleep:
                first_sleep = False
                scheduler.monitor_running = False

        with patch.object(
            scheduler, "deep_restart_workers", new_callable=AsyncMock
        ) as mock_deep_restart, patch(
            "model_services.scheduler.asyncio.sleep",
            side_effect=stop_after_first_iteration,
        ):
            await scheduler.worker_health_monitor()

        mock_logger.info.assert_any_call("Trying deep restart of all workers")
        mock_deep_restart.assert_called_once()

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


class TestSchedulerQueueTypes:
    """Test different queue type configurations"""

    def test_get_queue_default(self):
        """Test _get_queue returns multiprocessing.Queue by default"""
        mock_settings_default = Mock()
        mock_settings_default.device_ids = "(0)"
        mock_settings_default.max_queue_size = 10
        mock_settings_default.max_batch_size = 1
        mock_settings_default.use_queue_per_worker = False
        mock_settings_default.use_dynamic_batcher = False
        mock_settings_default.queue_for_multiprocessing = "default"  # Not a known type

        with patch(
            "model_services.scheduler.get_settings", return_value=mock_settings_default
        ):
            scheduler = Scheduler()
            # Default queue is multiprocessing.Queue
            assert scheduler.task_queue is not None


class TestSchedulerResultListener:
    """Test result_listener async method"""

    @pytest.fixture
    def scheduler_for_listener(self):
        """Create scheduler instance for listener tests"""
        mock_logger.reset_mock()
        mock_settings_listener = Mock()
        mock_settings_listener.device_ids = "(0)"
        mock_settings_listener.max_queue_size = 10
        mock_settings_listener.max_batch_size = 1
        mock_settings_listener.use_queue_per_worker = False
        mock_settings_listener.use_dynamic_batcher = False
        mock_settings_listener.queue_for_multiprocessing = "default"

        with patch(
            "model_services.scheduler.get_settings", return_value=mock_settings_listener
        ), patch("model_services.scheduler.TTLogger", return_value=mock_logger):
            scheduler = Scheduler()
            return scheduler

    @pytest.mark.asyncio
    async def test_result_listener_processes_results(self, scheduler_for_listener):
        """Test result_listener processes results from worker queues"""
        scheduler = scheduler_for_listener

        # Setup mock result queue
        mock_result_queue = Mock()
        call_count = [0]

        def mock_get_many(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [("worker_0", "task_123", {"data": "result"})]
            # Stop the listener after first iteration
            scheduler.listener_running = False
            return []

        mock_result_queue.get_many = Mock(side_effect=mock_get_many)
        scheduler.result_queues_by_worker = {0: mock_result_queue}

        # Create async queue for result
        result_queue = asyncio.Queue()
        scheduler.result_queues = {"task_123": result_queue}

        # Run listener
        await scheduler.result_listener()

        # Verify result was put in queue
        assert not result_queue.empty()
        result = await result_queue.get()
        assert result == {"data": "result"}

    @pytest.mark.asyncio
    async def test_result_listener_handles_none_result(self, scheduler_for_listener):
        """Test result_listener handles None results gracefully"""
        scheduler = scheduler_for_listener

        mock_result_queue = Mock()
        call_count = [0]

        def mock_get_many(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [None]  # None result should be skipped
            scheduler.listener_running = False
            return []

        mock_result_queue.get_many = Mock(side_effect=mock_get_many)
        scheduler.result_queues_by_worker = {0: mock_result_queue}
        scheduler.result_queues = {}

        # Should complete without error
        await scheduler.result_listener()

    @pytest.mark.asyncio
    async def test_result_listener_handles_shutdown_signal(
        self, scheduler_for_listener
    ):
        """Test result_listener stops on None result_key (shutdown signal)"""
        scheduler = scheduler_for_listener

        mock_result_queue = Mock()
        mock_result_queue.get_many = Mock(
            return_value=[("worker_0", None, None)]  # Shutdown signal
        )
        scheduler.result_queues_by_worker = {0: mock_result_queue}
        scheduler.result_queues = {}

        await scheduler.result_listener()

        # Listener should have stopped
        assert not scheduler.listener_running

    @pytest.mark.asyncio
    async def test_result_listener_handles_missing_queue(self, scheduler_for_listener):
        """Test result_listener logs warning for missing result queue"""
        scheduler = scheduler_for_listener

        mock_result_queue = Mock()
        call_count = [0]

        def mock_get_many(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [("worker_0", "missing_task", {"data": "orphan"})]
            scheduler.listener_running = False
            return []

        mock_result_queue.get_many = Mock(side_effect=mock_get_many)
        scheduler.result_queues_by_worker = {0: mock_result_queue}
        scheduler.result_queues = {}  # No queue registered for this task

        await scheduler.result_listener()

        # Verify warning was logged
        mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_result_listener_handles_exception(self, scheduler_for_listener):
        """Test result_listener handles exceptions gracefully"""
        scheduler = scheduler_for_listener

        mock_result_queue = Mock()
        call_count = [0]

        def mock_get_many(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Queue error")
            scheduler.listener_running = False
            return []

        mock_result_queue.get_many = Mock(side_effect=mock_get_many)
        scheduler.result_queues_by_worker = {0: mock_result_queue}

        # Should complete without raising
        await scheduler.result_listener()

    @pytest.mark.asyncio
    async def test_result_listener_continues_when_one_queue_get_many_raises(
        self, scheduler_for_listener
    ):
        """Test result_listener inner except (271-272): one queue raises, listener continues"""
        scheduler = scheduler_for_listener

        queue_ok = Mock()
        queue_ok.get_many = Mock(return_value=[])

        queue_raises = Mock()
        queue_raises.get_many = Mock(side_effect=RuntimeError("Queue read error"))

        scheduler.result_queues_by_worker = {0: queue_ok, 1: queue_raises}
        scheduler.result_queues = {}
        call_count = [0]

        async def stop_after_few(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 4:
                scheduler.listener_running = False

        with patch(
            "model_services.scheduler.asyncio.sleep",
            side_effect=stop_after_few,
        ):
            await scheduler.result_listener()

        queue_ok.get_many.assert_called()
        queue_raises.get_many.assert_called()
        assert not scheduler.listener_running

    @pytest.mark.asyncio
    async def test_result_listener_sleeps_when_no_results(self, scheduler_for_listener):
        """Test result_listener sleeps when no results found"""
        scheduler = scheduler_for_listener

        mock_result_queue = Mock()
        call_count = [0]

        def mock_get_many(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                scheduler.listener_running = False
            return []  # Empty results

        mock_result_queue.get_many = Mock(side_effect=mock_get_many)
        scheduler.result_queues_by_worker = {0: mock_result_queue}

        start = asyncio.get_event_loop().time()
        await scheduler.result_listener()
        elapsed = asyncio.get_event_loop().time() - start

        # Should have slept at least twice (0.001 * 2)
        assert elapsed >= 0.002


if __name__ == "__main__":
    pytest.main([__file__])
