# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import time
from multiprocessing import get_context

import pytest
from model_services.queues.tt_batch_fifo_queue import TTBatchFifoQueue


class TestTTBatchFifoQueueInit:
    def test_default_init(self):
        queue = TTBatchFifoQueue()
        assert queue.empty()
        assert queue.qsize() == 0

    def test_init_with_max_size(self):
        queue = TTBatchFifoQueue(max_size=5)
        assert queue.empty()

    def test_init_with_batch_size(self):
        queue = TTBatchFifoQueue(batch_size=2)
        assert queue._batch_size == 2

    def test_init_with_custom_context(self):
        ctx = get_context()
        queue = TTBatchFifoQueue(ctx=ctx)
        assert queue.empty()


class TestTTBatchFifoQueuePutGet:
    def test_put_and_get(self):
        queue = TTBatchFifoQueue()
        queue.put("item1")
        result = queue.get(timeout=1.0)
        assert result == "item1"

    def test_put_with_timeout(self):
        queue = TTBatchFifoQueue()
        queue.put("item1", timeout=1.0)
        result = queue.get(timeout=1.0)
        assert result == "item1"

    def test_put_blocking(self):
        queue = TTBatchFifoQueue()
        queue.put("item1", block=True)
        result = queue.get(block=True, timeout=1.0)
        assert result == "item1"

    def test_get_empty_returns_none(self):
        queue = TTBatchFifoQueue()
        result = queue.get(block=False)
        assert result is None

    def test_get_with_timeout_empty_returns_none(self):
        queue = TTBatchFifoQueue()
        result = queue.get(timeout=0.05)
        assert result is None

    def test_put_nowait(self):
        queue = TTBatchFifoQueue()
        queue.put_nowait("fast_item")
        time.sleep(0.05)
        result = queue.get(timeout=1.0)
        assert result == "fast_item"

    def test_get_nowait_with_item(self):
        queue = TTBatchFifoQueue()
        queue.put("item1")
        time.sleep(0.05)
        result = queue.get_nowait()
        assert result == "item1"

    def test_get_nowait_empty_returns_none(self):
        queue = TTBatchFifoQueue()
        result = queue.get_nowait()
        assert result is None


class TestTTBatchFifoQueuePutMany:
    def test_put_many(self):
        queue = TTBatchFifoQueue()
        queue.put_many(["a", "b", "c"])
        time.sleep(0.05)
        assert queue.qsize() == 3

    def test_put_many_with_timeout(self):
        queue = TTBatchFifoQueue()
        queue.put_many(["a", "b"], timeout=1.0)
        time.sleep(0.05)
        assert queue.qsize() == 2

    def test_put_many_without_timeout(self):
        queue = TTBatchFifoQueue()
        queue.put_many(["x", "y"], block=True)
        time.sleep(0.05)
        results = queue.get_many(max_messages_to_get=10, timeout=1.0)
        assert len(results) == 2


class TestTTBatchFifoQueueGetMany:
    def test_get_many_single_item(self):
        queue = TTBatchFifoQueue(batch_size=2)
        queue.put("item1")
        results = queue.get_many(max_messages_to_get=10, timeout=0.2)
        assert "item1" in results

    def test_get_many_fills_batch(self):
        queue = TTBatchFifoQueue(batch_size=2)
        queue.put("item1")
        queue.put("item2")
        time.sleep(0.05)
        results = queue.get_many(max_messages_to_get=2, timeout=1.0)
        assert len(results) == 2

    def test_get_many_stops_at_batch_size(self):
        queue = TTBatchFifoQueue(batch_size=2)
        for i in range(5):
            queue.put(f"item{i}")
        time.sleep(0.05)
        results = queue.get_many(max_messages_to_get=2, timeout=1.0)
        assert len(results) == 2

    def test_get_many_respects_max_messages(self):
        queue = TTBatchFifoQueue(batch_size=10)
        for i in range(5):
            queue.put(f"item{i}")
        time.sleep(0.05)
        results = queue.get_many(max_messages_to_get=3, timeout=1.0)
        assert len(results) <= 3

    def test_get_many_empty_queue_returns_empty(self):
        queue = TTBatchFifoQueue()
        results = queue.get_many(max_messages_to_get=10, block=True, timeout=0.05)
        assert results == []

    def test_get_many_nonblocking_empty(self):
        queue = TTBatchFifoQueue()
        results = queue.get_many(max_messages_to_get=10, block=False, timeout=0.05)
        assert results == []

    def test_get_many_without_timeout(self):
        queue = TTBatchFifoQueue(batch_size=1)
        queue.put("item1")
        time.sleep(0.05)
        results = queue.get_many(max_messages_to_get=1, block=True)
        assert results == ["item1"]

    def test_get_many_waits_for_batch(self):
        """When queue has partial batch, get_many waits briefly for more items."""
        queue = TTBatchFifoQueue(batch_size=2)
        queue.put("item1")
        results = queue.get_many(max_messages_to_get=2, timeout=0.3)
        assert len(results) >= 1
        assert "item1" in results

    def test_get_many_multiple_calls_drain_queue(self):
        queue = TTBatchFifoQueue(batch_size=2)
        for i in range(4):
            queue.put(f"item{i}")
        time.sleep(0.05)

        batch1 = queue.get_many(max_messages_to_get=2, timeout=0.5)
        batch2 = queue.get_many(max_messages_to_get=2, timeout=0.5)
        assert len(batch1) + len(batch2) == 4


class TestTTBatchFifoQueueState:
    def test_qsize(self):
        queue = TTBatchFifoQueue()
        queue.put("a")
        queue.put("b")
        time.sleep(0.05)
        assert queue.qsize() == 2

    def test_empty_true(self):
        queue = TTBatchFifoQueue()
        assert queue.empty() is True

    def test_empty_false(self):
        queue = TTBatchFifoQueue()
        queue.put("a")
        time.sleep(0.05)
        assert queue.empty() is False

    def test_full_false(self):
        queue = TTBatchFifoQueue(max_size=10)
        assert queue.full() is False

    def test_full_true(self):
        queue = TTBatchFifoQueue(max_size=1)
        queue.put("a")
        time.sleep(0.05)
        assert queue.full() is True


class TestTTBatchFifoQueuePeek:
    def test_peek_next_raises(self):
        queue = TTBatchFifoQueue()
        with pytest.raises(NotImplementedError):
            queue.peek_next()

    def test_peek_raises(self):
        queue = TTBatchFifoQueue()
        with pytest.raises(NotImplementedError):
            queue.peek(5)


class TestTTBatchFifoQueueClose:
    def test_close(self):
        queue = TTBatchFifoQueue()
        queue.put("item")
        queue.close()


class TestTTBatchFifoQueuePickle:
    def test_getstate(self):
        queue = TTBatchFifoQueue(max_size=5, batch_size=2)
        state = queue.__getstate__()
        assert "_batch_size" in state
        assert state["_batch_size"] == 2
        assert "_max_size" in state

    def test_setstate(self):
        queue = TTBatchFifoQueue(max_size=5, batch_size=2)
        state = queue.__getstate__()
        new_queue = TTBatchFifoQueue.__new__(TTBatchFifoQueue)
        new_queue.__setstate__(state)
        assert new_queue._batch_size == 2
        assert new_queue._max_size == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
