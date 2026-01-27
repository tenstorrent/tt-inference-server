# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
from multiprocessing import get_context
from queue import Empty

import pytest
from model_services.tt_queue import TTQueue


class TestTTQueueBasics:
    """Test basic TTQueue functionality"""

    def test_init_default(self):
        """Test default initialization"""
        queue = TTQueue()
        assert queue.batch_enabled is False
        assert queue.empty()

    def test_init_with_max_size(self):
        """Test initialization with max_size"""
        queue = TTQueue(max_size=10)
        assert queue.empty()

    def test_init_with_batch_enabled(self):
        """Test initialization with batch_enabled"""
        queue = TTQueue(batch_enabled=True)
        assert queue.batch_enabled is True

    def test_init_with_custom_context(self):
        """Test initialization with custom context"""
        ctx = get_context()
        queue = TTQueue(ctx=ctx)
        assert queue.empty()

    def test_put_and_get(self):
        """Test basic put and get operations"""
        queue = TTQueue()
        queue.put("test_item")
        result = queue.get(timeout=1.0)
        assert result == "test_item"

    def test_put_get_nowait(self):
        """Test put and get_nowait"""
        queue = TTQueue()
        queue.put("item1")
        # Small delay to ensure item is in queue
        time.sleep(0.05)
        result = queue.get_nowait()
        assert result == "item1"

    def test_get_nowait_empty_raises(self):
        """Test get_nowait on empty queue raises Empty"""
        queue = TTQueue()
        with pytest.raises(Empty):
            queue.get_nowait()


class TestTTQueueBatchOperations:
    """Test batch operations"""

    def test_get_many_single_item(self):
        """Test get_many with single item"""
        queue = TTQueue()
        queue.put("item1")
        results = queue.get_many(max_messages_to_get=10, block=True, timeout=1.0)
        assert results == ["item1"]

    def test_get_many_multiple_items(self):
        """Test get_many with multiple items"""
        queue = TTQueue()
        for i in range(5):
            queue.put(f"item{i}")

        # Small delay to ensure items are in queue
        time.sleep(0.01)

        results = queue.get_many(max_messages_to_get=10, block=True, timeout=1.0)
        assert len(results) == 5
        assert "item0" in results
        assert "item4" in results

    def test_get_many_respects_max(self):
        """Test get_many respects max_messages_to_get"""
        queue = TTQueue()
        for i in range(10):
            queue.put(f"item{i}")

        time.sleep(0.01)

        results = queue.get_many(max_messages_to_get=3, block=True, timeout=1.0)
        assert len(results) <= 3

    def test_get_many_empty_queue_nonblocking(self):
        """Test get_many on empty queue with block=False"""
        queue = TTQueue()
        results = queue.get_many(max_messages_to_get=10, block=False, timeout=0.01)
        assert results == []

    def test_get_many_with_none_shutdown_signal(self):
        """Test get_many handles None shutdown signal"""
        queue = TTQueue()
        queue.put("item1")
        queue.put(None)  # Shutdown signal
        queue.put("item2")

        time.sleep(0.01)

        results = queue.get_many(max_messages_to_get=10, block=True, timeout=1.0)
        # Should get item1 and None, then stop
        assert "item1" in results
        assert None in results

    def test_put_many(self):
        """Test put_many adds multiple items"""
        queue = TTQueue()
        items = ["item1", "item2", "item3"]
        queue.put_many(items)

        time.sleep(0.01)

        results = []
        while not queue.empty():
            try:
                results.append(queue.get_nowait())
            except Empty:
                break

        assert len(results) == 3
        assert "item1" in results
        assert "item2" in results
        assert "item3" in results

    def test_put_many_with_timeout(self):
        """Test put_many with timeout parameter"""
        queue = TTQueue()
        items = ["a", "b", "c"]
        queue.put_many(items, block=True, timeout=1.0)

        # Small delay
        time.sleep(0.05)

        results = queue.get_many(max_messages_to_get=10, block=True, timeout=1.0)
        assert len(results) == 3


class TestTTQueueBatchEnabled:
    """Test TTQueue with batch_enabled=True"""

    def test_get_with_leftover_items(self):
        """Test get checks leftover cache first when batch_enabled"""
        queue = TTQueue(batch_enabled=True)

        # Put item in leftover cache
        queue.return_item("leftover_item")
        time.sleep(0.05)

        # Put item in main queue
        queue.put("main_item")
        time.sleep(0.05)

        # Should get leftover item first
        result = queue.get(timeout=1.0)
        assert result == "leftover_item"

        # Then main queue item
        result = queue.get(timeout=1.0)
        assert result == "main_item"

    def test_get_nowait_with_leftover_items(self):
        """Test get_nowait checks leftover cache first"""
        queue = TTQueue(batch_enabled=True)

        queue.return_item("leftover")
        time.sleep(0.05)
        queue.put("main")
        time.sleep(0.05)

        result = queue.get_nowait()
        assert result == "leftover"

    def test_return_item(self):
        """Test return_item puts items in leftover cache"""
        queue = TTQueue(batch_enabled=True)

        queue.return_item("returned_item")
        time.sleep(0.05)

        # Should be able to get it back
        result = queue.get(timeout=1.0)
        assert result == "returned_item"

    def test_peek_next_with_leftover(self):
        """Test peek_next checks leftover cache"""
        queue = TTQueue(batch_enabled=True)

        queue.return_item("peeked")
        time.sleep(0.05)

        result = queue.peek_next()
        assert result == "peeked"

    def test_peek_next_from_main_queue(self):
        """Test peek_next from main queue when leftover empty"""
        queue = TTQueue(batch_enabled=True)

        queue.put("main_item")
        time.sleep(0.05)

        result = queue.peek_next()
        assert result == "main_item"


class TestTTQueuePickling:
    """Test TTQueue serialization for multiprocessing"""

    @pytest.mark.skip(reason="__getstate__ can only be called during spawning context")
    def test_getstate_setstate(self):
        """Test __getstate__ and __setstate__ preserve custom attributes"""
        queue = TTQueue(max_size=5, batch_enabled=True)
        queue.put("test")
        time.sleep(0.05)

        # Simulate pickling
        state = queue.__getstate__()

        # Create new queue and restore state
        new_queue = TTQueue()
        new_queue.__setstate__(state)

        assert new_queue.batch_enabled is True


class TestTTQueueEdgeCases:
    """Test edge cases"""

    def test_get_many_timeout_empty(self):
        """Test get_many with timeout on empty queue"""
        queue = TTQueue()
        start = time.time()
        results = queue.get_many(max_messages_to_get=10, block=True, timeout=0.1)
        elapsed = time.time() - start

        assert results == []
        assert elapsed >= 0.1

    def test_multiple_get_many_calls(self):
        """Test multiple sequential get_many calls"""
        queue = TTQueue()

        for i in range(10):
            queue.put(f"item{i}")

        time.sleep(0.01)

        batch1 = queue.get_many(max_messages_to_get=3, block=False, timeout=0.01)
        batch2 = queue.get_many(max_messages_to_get=3, block=False, timeout=0.01)
        batch3 = queue.get_many(max_messages_to_get=10, block=False, timeout=0.01)

        total_items = len(batch1) + len(batch2) + len(batch3)
        assert total_items == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
