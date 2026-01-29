# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time

import pytest
from model_services.tt_faster_fifo_queue import TTFasterFifoQueue


class TestTTFasterFifoQueueInit:
    """Test TTFasterFifoQueue initialization"""

    def test_init_default(self):
        """Test default initialization"""
        queue = TTFasterFifoQueue()
        assert queue.empty()
        assert queue._max_size == 0

    def test_init_with_max_size(self):
        """Test initialization with max_size"""
        queue = TTFasterFifoQueue(max_size=100)
        assert queue._max_size == 100
        assert queue.empty()

    def test_init_small_max_size(self):
        """Test initialization with small max_size ensures minimum buffer"""
        queue = TTFasterFifoQueue(max_size=1)
        assert queue._max_size == 1


class TestTTFasterFifoQueuePut:
    """Test put operations"""

    def test_put_single_item(self):
        """Test putting a single item"""
        queue = TTFasterFifoQueue()
        queue.put("test_item")
        assert not queue.empty()

    def test_put_with_block(self):
        """Test put with block parameter"""
        queue = TTFasterFifoQueue()
        queue.put("item", block=True)
        result = queue.get()
        assert result == "item"

    def test_put_with_timeout(self):
        """Test put with timeout parameter"""
        queue = TTFasterFifoQueue()
        queue.put("item", block=True, timeout=1.0)
        result = queue.get()
        assert result == "item"

    def test_put_nowait(self):
        """Test put_nowait"""
        queue = TTFasterFifoQueue()
        queue.put_nowait("nowait_item")
        result = queue.get()
        assert result == "nowait_item"

    def test_put_many(self):
        """Test put_many for batch insertion"""
        queue = TTFasterFifoQueue()
        items = ["item1", "item2", "item3", "item4", "item5"]
        queue.put_many(items)

        # Get all items back
        results = queue.get_many(max_messages_to_get=10)
        assert len(results) == 5
        for item in items:
            assert item in results

    def test_put_many_with_timeout(self):
        """Test put_many with timeout"""
        queue = TTFasterFifoQueue()
        items = ["a", "b", "c"]
        queue.put_many(items, block=True, timeout=1.0)

        results = queue.get_many(max_messages_to_get=10)
        assert len(results) == 3


class TestTTFasterFifoQueueGet:
    """Test get operations"""

    def test_get_single_item(self):
        """Test getting a single item"""
        queue = TTFasterFifoQueue()
        queue.put("test")
        result = queue.get()
        assert result == "test"

    def test_get_with_block(self):
        """Test get with block parameter"""
        queue = TTFasterFifoQueue()
        queue.put("blocked_item")
        result = queue.get(block=True)
        assert result == "blocked_item"

    def test_get_with_timeout(self):
        """Test get with timeout"""
        queue = TTFasterFifoQueue()
        queue.put("timed_item")
        result = queue.get(block=True, timeout=1.0)
        assert result == "timed_item"

    def test_get_nowait(self):
        """Test get_nowait"""
        queue = TTFasterFifoQueue()
        queue.put("nowait")
        result = queue.get_nowait()
        assert result == "nowait"

    def test_get_nowait_empty_raises(self):
        """Test get_nowait on empty queue raises exception"""
        queue = TTFasterFifoQueue()
        with pytest.raises(Exception):
            queue.get_nowait()


class TestTTFasterFifoQueueGetMany:
    """Test get_many batch operations"""

    def test_get_many_basic(self):
        """Test basic get_many operation"""
        queue = TTFasterFifoQueue()
        for i in range(5):
            queue.put(f"item{i}")

        results = queue.get_many(max_messages_to_get=10)
        assert len(results) == 5

    def test_get_many_respects_max(self):
        """Test get_many respects max_messages_to_get"""
        queue = TTFasterFifoQueue()
        for i in range(10):
            queue.put(f"item{i}")

        results = queue.get_many(max_messages_to_get=3)
        assert len(results) <= 3

    def test_get_many_with_block(self):
        """Test get_many with block parameter"""
        queue = TTFasterFifoQueue()
        queue.put("item1")
        queue.put("item2")

        results = queue.get_many(max_messages_to_get=10, block=True)
        assert len(results) == 2

    def test_get_many_with_timeout(self):
        """Test get_many with timeout"""
        queue = TTFasterFifoQueue()
        queue.put("item")

        results = queue.get_many(max_messages_to_get=10, block=True, timeout=1.0)
        assert len(results) == 1

    def test_get_many_empty_nonblocking(self):
        """Test get_many on empty queue with block=False returns empty list"""
        queue = TTFasterFifoQueue()
        results = queue.get_many(max_messages_to_get=10, block=False)
        assert results == []

    def test_get_many_timeout_empty(self):
        """Test get_many with timeout on empty queue"""
        queue = TTFasterFifoQueue()
        start = time.time()
        results = queue.get_many(max_messages_to_get=10, block=True, timeout=0.1)
        elapsed = time.time() - start

        assert results == []
        # Should have waited approximately the timeout
        assert elapsed >= 0.05


class TestTTFasterFifoQueueStatus:
    """Test queue status methods"""

    def test_qsize(self):
        """Test qsize returns approximate size"""
        queue = TTFasterFifoQueue()
        assert queue.qsize() == 0

        queue.put("item1")
        queue.put("item2")
        assert queue.qsize() == 2

    def test_empty_true(self):
        """Test empty returns True for empty queue"""
        queue = TTFasterFifoQueue()
        assert queue.empty() is True

    def test_empty_false(self):
        """Test empty returns False for non-empty queue"""
        queue = TTFasterFifoQueue()
        queue.put("item")
        assert queue.empty() is False

    def test_full_with_no_max(self):
        """Test full returns False when max_size=0"""
        queue = TTFasterFifoQueue(max_size=0)
        for i in range(100):
            queue.put(f"item{i}")
        assert queue.full() is False

    def test_full_when_at_capacity(self):
        """Test full returns True when at capacity"""
        queue = TTFasterFifoQueue(max_size=3)
        queue.put("item1")
        queue.put("item2")
        queue.put("item3")
        assert queue.full() is True

    def test_full_when_below_capacity(self):
        """Test full returns False when below capacity"""
        queue = TTFasterFifoQueue(max_size=10)
        queue.put("item1")
        queue.put("item2")
        assert queue.full() is False


class TestTTFasterFifoQueueDataTypes:
    """Test with various data types"""

    def test_string_data(self):
        """Test with string data"""
        queue = TTFasterFifoQueue()
        queue.put("hello world")
        assert queue.get() == "hello world"

    def test_dict_data(self):
        """Test with dictionary data"""
        queue = TTFasterFifoQueue()
        data = {"key": "value", "number": 42}
        queue.put(data)
        result = queue.get()
        assert result == data

    def test_list_data(self):
        """Test with list data"""
        queue = TTFasterFifoQueue()
        data = [1, 2, 3, "four", 5.0]
        queue.put(data)
        result = queue.get()
        assert result == data

    def test_tuple_data(self):
        """Test with tuple data"""
        queue = TTFasterFifoQueue()
        data = ("worker_id", "task_id", {"result": "data"})
        queue.put(data)
        result = queue.get()
        assert result == data

    def test_none_data(self):
        """Test with None data (shutdown signal)"""
        queue = TTFasterFifoQueue()
        queue.put(None)
        result = queue.get()
        assert result is None

    def test_bytes_data(self):
        """Test with bytes data"""
        queue = TTFasterFifoQueue()
        data = b"binary data here"
        queue.put(data)
        result = queue.get()
        assert result == data


class TestTTFasterFifoQueuePerformance:
    """Performance-related tests"""

    def test_many_items_throughput(self):
        """Test handling many items"""
        queue = TTFasterFifoQueue(max_size=10000)
        num_items = 1000

        # Put many items
        items = [f"item_{i}" for i in range(num_items)]
        queue.put_many(items)

        # Get all items back
        retrieved = []
        while not queue.empty():
            batch = queue.get_many(max_messages_to_get=100, block=False)
            if not batch:
                break
            retrieved.extend(batch)

        assert len(retrieved) == num_items

    def test_concurrent_batch_operations(self):
        """Test batch put followed by batch get"""
        queue = TTFasterFifoQueue()

        # Batch put
        items = [{"id": i, "data": f"payload_{i}"} for i in range(50)]
        queue.put_many(items)

        # Batch get
        results = queue.get_many(max_messages_to_get=100, block=True, timeout=1.0)

        assert len(results) == 50
        ids = [r["id"] for r in results]
        assert sorted(ids) == list(range(50))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
