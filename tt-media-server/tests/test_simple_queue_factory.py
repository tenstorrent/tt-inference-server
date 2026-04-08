# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from config.constants import QueueType
from model_services.queues.tt_batch_fifo_queue import TTBatchFifoQueue
from model_services.queues.tt_faster_fifo_queue import TTFasterFifoQueue
from model_services.queues.tt_queue import TTQueue
from utils.simple_queue_factory import get_queue, get_task_queue


class TestGetQueue:
    def test_faster_fifo(self):
        queue = get_queue(QueueType.FasterFifo.value, size=10)
        assert isinstance(queue, TTFasterFifoQueue)

    def test_batch_fifo(self):
        queue = get_queue(QueueType.BatchFifo.value, size=10)
        assert isinstance(queue, TTBatchFifoQueue)
        assert queue._batch_size >= 1

    def test_tt_queue(self):
        queue = get_queue(QueueType.TTQueue.value, size=10)
        assert isinstance(queue, TTQueue)

    def test_memory_queue(self):
        from model_services.queues.memory_queue import SharedMemoryChunkQueue

        queue = get_queue(QueueType.MemoryQueue.value, size=10, name="test_factory_q")
        assert isinstance(queue, SharedMemoryChunkQueue)
        queue.close()

    def test_unknown_returns_none(self):
        result = get_queue("unknown_type", size=10)
        assert result is None


class TestGetTaskQueue:
    def test_faster_fifo_returns_faster_fifo_queue(self):
        queue = get_task_queue(QueueType.FasterFifo.value, size=10)
        assert isinstance(queue, TTFasterFifoQueue)

    def test_batch_fifo_returns_batch_queue(self):
        queue = get_task_queue(QueueType.BatchFifo.value, size=10)
        assert isinstance(queue, TTBatchFifoQueue)
        assert queue._batch_size >= 1

    def test_other_type_returns_tt_queue(self):
        queue = get_task_queue(QueueType.TTQueue.value, size=10)
        assert isinstance(queue, TTQueue)
