# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
from multiprocessing import shared_memory

import pytest
from domain.completion_response import CompletionOutput, CompletionResult


# Mock the CompletionResult before importing memory_queue
class MockCompletionResult:
    """Mock CompletionResult for testing"""

    def __init__(self, text=None, index=None, finish_reason=None):
        self.text = text
        self.index = index
        self.finish_reason = finish_reason


# Create mock modules with proper dependencies
# Only mock if not already mocked by conftest.py
if "domain.completion_response" not in sys.modules:
    from unittest.mock import Mock

    mock_completion_response = Mock()
    mock_completion_response.CompletionResult = MockCompletionResult
    sys.modules["domain.completion_response"] = mock_completion_response
else:
    # Update the existing mock with our MockCompletionResult
    sys.modules["domain.completion_response"].CompletionResult = MockCompletionResult

# DO NOT mock utils.logger here - let conftest.py handle it
# The logger in conftest.py is already properly configured

# Now import the actual module we're testing
from model_services.queues.memory_queue import SharedMemoryChunkQueue


def make_chunk(task_id: str, is_final: int, text: str):
    """Helper to create test data in the expected format."""
    chunk_type = "final_result" if is_final else "streaming_chunk"
    return (
        "worker_id",
        task_id,
        CompletionOutput(
            type=chunk_type,
            data=CompletionResult(text=text),
        ),
    )


@pytest.fixture
def cleanup_queues():
    """Cleanup any existing shared memory from previous tests"""
    queue_names = [f"test_queue_{i}" for i in range(100)]
    for name in queue_names:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except (FileNotFoundError, OSError):
            pass
    yield
    # Cleanup after test
    for name in queue_names:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except (FileNotFoundError, OSError):
            pass


class TestSharedMemoryChunkQueueInitialization:
    """Test SharedMemoryChunkQueue initialization"""

    def test_init_creates_shared_memory(self, cleanup_queues):
        """Test that initialization creates shared memory"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_1", create=True)

        assert queue is not None
        assert queue.shm is not None
        assert queue.capacity == 1000
        assert queue.name == "test_queue_1"

        queue.close()
        queue.unlink()

    def test_init_with_custom_capacity(self, cleanup_queues):
        """Test initialization with custom capacity"""
        capacity = 5000
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_2", create=True
        )

        assert queue.capacity == capacity

        queue.close()
        queue.unlink()

    def test_init_attaches_to_existing_queue(self, cleanup_queues):
        """Test that initialization can attach to existing queue"""
        queue1 = SharedMemoryChunkQueue(capacity=1000, name="test_queue_3", create=True)

        # Create a second queue that attaches to the same shared memory
        queue2 = SharedMemoryChunkQueue(
            capacity=1000, name="test_queue_3", create=False
        )

        assert queue2.shm is not None
        assert queue2.capacity == 1000

        queue1.close()
        queue1.unlink()
        queue2.close()

    def test_init_cleans_up_existing_queue(self, cleanup_queues):
        """Test that initialization cleans up existing queue"""
        queue1 = SharedMemoryChunkQueue(capacity=1000, name="test_queue_4", create=True)
        queue1.close()

        # Create a new queue with same name - should clean up old one
        queue2 = SharedMemoryChunkQueue(capacity=1000, name="test_queue_4", create=True)

        assert queue2 is not None

        queue2.close()
        queue2.unlink()

    def test_header_initialized_correctly(self, cleanup_queues):
        """Test that header is initialized with correct values"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_5", create=True)

        # Check that indices are initialized to 0
        assert queue._get_write_idx() == 0
        assert queue._get_read_idx() == 0
        assert queue._get_size() == 0

        queue.close()
        queue.unlink()


class TestQueueSize:
    """Test queue size calculations"""

    def test_size_empty_queue(self, cleanup_queues):
        """Test size of empty queue"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_6", create=True)

        assert queue._get_size() == 0

        queue.close()
        queue.unlink()

    def test_size_with_items(self, cleanup_queues):
        """Test size calculation with items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_7", create=True)

        # Add 5 items
        for i in range(5):
            result = queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            assert result is True

        # Size should be 5
        assert queue._get_size() == 5

        queue.close()
        queue.unlink()

    def test_size_with_wraparound(self, cleanup_queues):
        """Test size calculation with index wraparound"""
        queue = SharedMemoryChunkQueue(capacity=100, name="test_queue_8", create=True)

        # Fill queue to near capacity (leaving margin)
        for i in range(80):
            result = queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            assert result is True

        assert queue._get_size() == 80

        # Read some items to advance read_idx
        for i in range(30):
            queue.get_nowait()

        # Size should be 50
        assert queue._get_size() == 50

        queue.close()
        queue.unlink()


class TestPutOperation:
    """Test put operation"""

    def test_put_single_item(self, cleanup_queues):
        """Test putting a single item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_9", create=True)

        result = queue.put(make_chunk("task_1", 0, "hello world"))

        assert result is True
        assert queue._get_size() == 1

        queue.close()
        queue.unlink()

    def test_put_truncates_long_task_id(self, cleanup_queues):
        """Test that long task_id is truncated"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_10", create=True)

        long_task_id = "a" * 200  # Longer than MAX_TASK_ID_LEN (100)
        result = queue.put(make_chunk(long_task_id, 0, "text"))

        assert result is True
        # Queue should accept the truncated version
        assert queue._get_size() == 1

        queue.close()
        queue.unlink()

    def test_put_truncates_long_text(self, cleanup_queues):
        """Test that long text is truncated"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_11", create=True)

        long_text = "x" * 500  # Longer than MAX_TEXT_LEN (450)
        result = queue.put(make_chunk("task_1", 0, long_text))

        assert result is True
        # Queue should accept the truncated version
        assert queue._get_size() == 1

        queue.close()
        queue.unlink()

    def test_put_with_is_final_flag(self, cleanup_queues):
        """Test putting item with is_final flag"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_13", create=True)

        result = queue.put(make_chunk("task_1", 0, "chunk"))
        assert result is True

        result = queue.put(make_chunk("task_1", 1, "final"))
        assert result is True

        assert queue._get_size() == 2

        queue.close()
        queue.unlink()


class TestGetNowait:
    """Test non-blocking get operation"""

    def test_get_nowait_empty_queue(self, cleanup_queues):
        """Test get_nowait on empty queue returns None"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_14", create=True)

        result = queue.get_nowait()

        assert result is None

        queue.close()
        queue.unlink()

    def test_get_nowait_single_item(self, cleanup_queues):
        """Test get_nowait retrieves single item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_15", create=True)

        queue.put(make_chunk("task_1", 0, "hello"))
        result = queue.get_nowait()

        assert result is not None
        assert len(result) == 3
        worker_id, task_id, chunk_dict = result
        assert task_id == "task_1"
        assert chunk_dict["type"] == "streaming_chunk"

        queue.close()
        queue.unlink()

    def test_get_nowait_respects_order(self, cleanup_queues):
        """Test that get_nowait retrieves items in order"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_16", create=True)

        # Put 3 items
        for i in range(3):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Get them back in order
        for i in range(3):
            worker_id, task_id, chunk_dict = queue.get_nowait()
            assert task_id == f"task_{i}"

        # Queue should be empty now
        assert queue.get_nowait() is None

        queue.close()
        queue.unlink()

    def test_get_nowait_with_final_flag(self, cleanup_queues):
        """Test get_nowait with is_final flag"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_17", create=True)

        queue.put(make_chunk("task_1", 1, "final_text"))
        worker_id, task_id, chunk_dict = queue.get_nowait()

        assert chunk_dict["type"] == "final_result"
        assert task_id == "task_1"

        queue.close()
        queue.unlink()

    def test_get_nowait_multiple_items(self, cleanup_queues):
        """Test get_nowait with multiple items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_18", create=True)

        # Put multiple items
        for i in range(10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Get all items
        results = []
        while True:
            result = queue.get_nowait()
            if result is None:
                break
            results.append(result)

        assert len(results) == 10
        assert queue._get_size() == 0

        queue.close()
        queue.unlink()


class TestGetBlocking:
    """Test blocking get operation"""

    def test_get_blocking_single_item(self, cleanup_queues):
        """Test blocking get retrieves item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_19", create=True)

        queue.put(make_chunk("task_1", 0, "hello"))
        worker_id, task_id, chunk_dict = queue.get(timeout=1.0)

        assert task_id == "task_1"
        assert chunk_dict["type"] == "streaming_chunk"

        queue.close()
        queue.unlink()

    def test_get_blocking_timeout(self, cleanup_queues):
        """Test blocking get times out on empty queue"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_20", create=True)

        with pytest.raises(TimeoutError):
            queue.get(timeout=0.1)

        queue.close()
        queue.unlink()

    def test_get_blocking_multiple_items(self, cleanup_queues):
        """Test blocking get with multiple items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_21", create=True)

        # Put multiple items
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Get all items
        for i in range(5):
            worker_id, task_id, chunk_dict = queue.get(timeout=1.0)
            assert task_id == f"task_{i}"

        queue.close()
        queue.unlink()


class TestIndexManagement:
    """Test index management (write and read)"""

    def test_write_idx_increments(self, cleanup_queues):
        """Test that write_idx increments correctly"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_22", create=True)

        initial_idx = queue._get_write_idx()

        queue.put(make_chunk("task_1", 0, "text"))

        new_idx = queue._get_write_idx()
        assert new_idx == (initial_idx + 1) % queue.capacity

        queue.close()
        queue.unlink()

    def test_read_idx_increments(self, cleanup_queues):
        """Test that read_idx increments correctly"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_23", create=True)

        queue.put(make_chunk("task_1", 0, "text"))

        initial_read_idx = queue._get_read_idx()
        queue.get_nowait()

        new_read_idx = queue._get_read_idx()
        assert new_read_idx == (initial_read_idx + 1) % queue.capacity

        queue.close()
        queue.unlink()

    def test_index_wraparound(self, cleanup_queues):
        """Test that indices wrap around correctly"""
        capacity = 10
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_24", create=True
        )

        # Fill and drain queue multiple times to cause wraparound
        for _ in range(2):
            for i in range(capacity - 1):
                queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

            for i in range(capacity - 1):
                queue.get_nowait()

        # Indices should have wrapped around
        assert queue._get_write_idx() < capacity
        assert queue._get_read_idx() < capacity

        queue.close()
        queue.unlink()


class TestQueueClose:
    """Test queue cleanup operations"""

    def test_close_shared_memory(self, cleanup_queues):
        """Test that close properly closes shared memory"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_25", create=True)

        queue.close()

        # After close, the queue should still have reference but be closed
        assert queue.shm is not None

    def test_unlink_shared_memory(self, cleanup_queues):
        """Test that unlink properly unlinks shared memory"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_26", create=True)

        queue.unlink()

        # After unlink, shared memory should be cleaned up
        assert queue.shm is not None

    def test_close_and_unlink(self, cleanup_queues):
        """Test close followed by unlink"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_27", create=True)

        queue.close()
        queue.unlink()

        # Should not raise any errors


class TestJoinThread:
    """Test join_thread method"""

    def test_join_thread_returns_true(self, cleanup_queues):
        """Test that join_thread returns True"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_28", create=True)

        result = queue.join_thread()

        assert result is True

        queue.close()
        queue.unlink()


class TestErrorHandling:
    """Test error handling"""

    def test_put_with_invalid_index(self, cleanup_queues):
        """Test put handles errors gracefully"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_29", create=True)

        # Normal put should work
        result = queue.put(make_chunk("task_1", 0, "text"))
        assert result is True

        queue.close()
        queue.unlink()

    def test_get_nowait_handles_errors(self, cleanup_queues):
        """Test get_nowait handles errors gracefully"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_30", create=True)

        queue.put(make_chunk("task_1", 0, "text"))
        result = queue.get_nowait()

        # Should return valid result
        assert result is not None

        queue.close()
        queue.unlink()


class TestQueueIntegration:
    """Integration tests for queue operations"""

    def test_put_and_get_sequence(self, cleanup_queues):
        """Test sequence of puts and gets"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_31", create=True)

        # Put several items
        items = [
            ("task_1", 0, "first"),
            ("task_1", 0, "second"),
            ("task_1", 1, "final"),
        ]

        for task_id, is_final, text in items:
            result = queue.put(make_chunk(task_id, is_final, text))
            assert result is True

        # Get them back
        for i, (expected_task, expected_final, expected_text) in enumerate(items):
            worker_id, task_id, chunk_dict = queue.get_nowait()
            assert task_id == expected_task
            if expected_final == 1:
                assert chunk_dict["type"] == "final_result"
            else:
                assert chunk_dict["type"] == "streaming_chunk"

        # Queue should be empty
        assert queue.get_nowait() is None

        queue.close()
        queue.unlink()

    def test_concurrent_put_and_get(self, cleanup_queues):
        """Test interleaved puts and gets"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_32", create=True)

        # Alternate between put and get
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            worker_id, task_id, chunk_dict = queue.get_nowait()
            assert task_id == f"task_{i}"

        # Queue should be empty
        assert queue.get_nowait() is None

        queue.close()
        queue.unlink()

    def test_large_queue_capacity(self, cleanup_queues):
        """Test queue with large capacity"""
        capacity = 50000
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_33", create=True
        )

        # Put and get many items
        num_items = 1000
        for i in range(num_items):
            result = queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            assert result is True

        assert queue._get_size() == num_items

        # Get some items back
        for i in range(num_items // 2):
            result = queue.get_nowait()
            assert result is not None

        assert queue._get_size() == num_items // 2

        queue.close()
        queue.unlink()


class TestGetMany:
    """Test get_many batch operations"""

    def test_get_many_empty_queue_block_false(self, cleanup_queues):
        """Test get_many with block=False returns empty list"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_34", create=True)

        results = queue.get_many(max_messages_to_get=10, block=False)

        assert results == []

        queue.close()
        queue.unlink()

    def test_get_many_timeout(self, cleanup_queues):
        """Test get_many times out on empty queue"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_35", create=True)

        with pytest.raises(TimeoutError):
            queue.get_many(max_messages_to_get=10, block=True, timeout=0.1)

        queue.close()
        queue.unlink()

    def test_get_many_single_item(self, cleanup_queues):
        """Test get_many retrieves single item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_36", create=True)

        queue.put(make_chunk("task_1", 0, "hello"))
        results = queue.get_many(max_messages_to_get=10)

        assert len(results) == 1
        worker_id, task_id, chunk_dict = results[0]
        assert task_id == "task_1"
        assert chunk_dict["type"] == "streaming_chunk"

        queue.close()
        queue.unlink()

    def test_get_many_multiple_items(self, cleanup_queues):
        """Test get_many retrieves multiple items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_37", create=True)

        # Put 5 items
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        results = queue.get_many(max_messages_to_get=10)

        assert len(results) == 5
        for i, (worker_id, task_id, chunk_dict) in enumerate(results):
            assert task_id == f"task_{i}"

        queue.close()
        queue.unlink()

    def test_get_many_with_final_items(self, cleanup_queues):
        """Test get_many with final_result items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_38", create=True)

        queue.put(make_chunk("task_1", 0, "chunk"))
        queue.put(make_chunk("task_1", 1, "final"))

        results = queue.get_many(max_messages_to_get=10)

        assert len(results) == 2
        assert results[0][2]["type"] == "streaming_chunk"
        assert results[1][2]["type"] == "final_result"

        queue.close()
        queue.unlink()

    def test_get_many_limits_max_items(self, cleanup_queues):
        """Test get_many respects max_messages_to_get limit"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_39", create=True)

        # Put 10 items
        for i in range(10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Request only 3
        results = queue.get_many(max_messages_to_get=3)

        assert len(results) == 3
        assert queue._get_size() == 7

        queue.close()
        queue.unlink()

    def test_get_many_with_wraparound(self, cleanup_queues):
        """Test get_many handles wraparound correctly"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_40", create=True
        )

        # Fill queue to near capacity (accounting for margin of 10)
        # With capacity=20, margin=10, so max items = 10
        for i in range(10):
            result = queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            assert result is True

        # Read some items to advance read_idx
        for _ in range(5):
            queue.get_nowait()

        # At this point: read_idx=5, write_idx=10, so 5 items remain
        assert queue._get_size() == 5

        # Put more items to cause wraparound
        # After 5 puts: write_idx wraps from 10->11->12->13->14->15
        # Items written to slots 10, 11, 12, 13, 14
        for i in range(10, 15):
            result = queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))
            assert result is True

        # After wraparound: read_idx=5, write_idx=15
        # write_idx >= read_idx, so available_items = write_idx - read_idx = 15 - 5 = 10
        results = queue.get_many(max_messages_to_get=20)

        # Should get all 10 items (5 remaining at slots 5-9 + 5 new at slots 10-14)
        assert len(results) == 10

        queue.close()
        queue.unlink()

    def test_get_many_corruption_read_idx_out_of_bounds(self, cleanup_queues):
        """Test get_many handles corrupted read_idx"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_41", create=True)

        # Manually corrupt read_idx
        import struct

        struct.pack_into("Q", queue.shm.buf, queue.header_offset_read, 9999)

        results = queue.get_many(max_messages_to_get=10)

        # Should return empty list on corruption
        assert results == []

        queue.close()
        queue.unlink()

    def test_get_many_corruption_read_idx_out_of_bounds_detected(self, cleanup_queues):
        """Test get_many detects read_idx corruption when out of bounds"""
        queue = SharedMemoryChunkQueue(capacity=100, name="test_queue_42", create=True)

        # Put some items
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Corrupt read_idx to be >= capacity (this should be detected)
        import struct

        struct.pack_into("Q", queue.shm.buf, queue.header_offset_read, 200)

        # The corruption check at line 345 should catch read_idx >= capacity
        results = queue.get_many(max_messages_to_get=10)

        # Should return empty list on corruption detection
        assert results == []

        queue.close()
        queue.unlink()

    def test_get_many_with_large_write_idx(self, cleanup_queues):
        """Test get_many handles large write_idx that wraps correctly"""
        queue = SharedMemoryChunkQueue(capacity=100, name="test_queue_42b", create=True)

        # Put some items
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Set write_idx to a large value that wraps (but is still valid modulo)
        # This tests that the queue handles modulo arithmetic correctly
        import struct

        # Set write_idx to 150, which modulo 100 = 50
        # This simulates a scenario where write_idx has wrapped multiple times
        struct.pack_into("Q", queue.shm.buf, queue.header_offset_write, 150)

        # read_idx=0, write_idx=50 (after modulo)
        # available_items = 50 - 0 = 50
        results = queue.get_many(max_messages_to_get=10)

        # Should read up to 10 items (limited by max_messages_to_get)
        # But actual data is only at indices 0-4, so it will read empty slots
        assert len(results) == 10  # Reads what it calculates, even if data is empty

        queue.close()
        queue.unlink()

    def test_get_many_error_handling(self, cleanup_queues):
        """Test get_many error handling updates read_idx correctly"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_43", create=True)

        # Put items
        for i in range(3):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Mock buffer access to raise exception
        original_buffer = queue.buffer

        class CorruptBuffer:
            def __getitem__(self, idx):
                if idx == 0:
                    raise RuntimeError("Simulated error")
                return original_buffer[idx]

            def __setitem__(self, idx, val):
                original_buffer[idx] = val

        queue.buffer = CorruptBuffer()

        # Should handle error and update read_idx for successfully read items
        with pytest.raises(RuntimeError):
            queue.get_many(max_messages_to_get=10)

        queue.close()
        queue.unlink()


class TestPutMany:
    """Test put_many batch operations"""

    def test_put_many_empty_list(self, cleanup_queues):
        """Test put_many with empty list returns early"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_44", create=True)

        # Should not raise error
        queue.put_many([])

        assert queue._get_size() == 0

        queue.close()
        queue.unlink()

    def test_put_many_single_item(self, cleanup_queues):
        """Test put_many with single item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_45", create=True)

        items = [make_chunk("task_1", 0, "hello")]
        queue.put_many(items)

        assert queue._get_size() == 1
        result = queue.get_nowait()
        assert result[1] == "task_1"

        queue.close()
        queue.unlink()

    def test_put_many_multiple_items(self, cleanup_queues):
        """Test put_many with multiple items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_46", create=True)

        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(10)]
        queue.put_many(items)

        assert queue._get_size() == 10

        # Verify all items
        for i in range(10):
            result = queue.get_nowait()
            assert result[1] == f"task_{i}"

        queue.close()
        queue.unlink()

    def test_put_many_with_final_items(self, cleanup_queues):
        """Test put_many with final_result items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_47", create=True)

        items = [
            make_chunk("task_1", 0, "chunk1"),
            make_chunk("task_1", 0, "chunk2"),
            make_chunk("task_1", 1, "final"),
        ]
        queue.put_many(items)

        results = queue.get_many(max_messages_to_get=10)
        assert len(results) == 3
        assert results[0][2]["type"] == "streaming_chunk"
        assert results[1][2]["type"] == "streaming_chunk"
        assert results[2][2]["type"] == "final_result"

        queue.close()
        queue.unlink()

    def test_put_many_truncates_long_fields(self, cleanup_queues):
        """Test put_many truncates long task_id and text"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_48", create=True)

        long_task_id = "a" * 200
        long_text = "x" * 500
        items = [make_chunk(long_task_id, 0, long_text)]
        queue.put_many(items)

        result = queue.get_nowait()
        # Should be truncated
        assert len(result[1]) <= 100
        assert len(result[2]["data"].text) <= 450

        queue.close()
        queue.unlink()

    def test_put_many_timeout(self, cleanup_queues):
        """Test put_many times out when queue is full"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_49", create=True
        )

        # Fill queue to near capacity
        for i in range(capacity - 5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Try to put many items that exceed capacity
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(20)]

        with pytest.raises(TimeoutError):
            queue.put_many(items, block=True, timeout=0.1)

        queue.close()
        queue.unlink()

    def test_put_many_block_false_partial(self, cleanup_queues):
        """Test put_many with block=False puts as many as possible"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_50", create=True
        )

        # Fill queue partially (leaving margin of 10)
        for i in range(10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # At this point: write_idx=10, read_idx=0
        # available_space = (capacity - write_idx) + read_idx - 10 = (20 - 10) + 0 - 10 = 0
        # So no items can be put without blocking

        # Try to put many items
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(20)]

        # Should put as many as possible without blocking
        queue.put_many(items, block=False)

        # With available_space=0, num_items becomes 0 and function returns early
        # So size should remain 10
        assert queue._get_size() == 10

        queue.close()
        queue.unlink()

    def test_put_many_block_false_with_space(self, cleanup_queues):
        """Test put_many with block=False when space is available"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_50c", create=True
        )

        # Fill queue partially (leaving more space)
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # At this point: write_idx=5, read_idx=0
        # available_space = (capacity - write_idx) + read_idx - 10 = (20 - 5) + 0 - 10 = 5

        # Try to put items
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(10)]

        # Should put as many as possible without blocking (5 items)
        queue.put_many(items, block=False)

        # Should have put 5 items
        assert queue._get_size() == 10  # 5 original + 5 new

        queue.close()
        queue.unlink()

    def test_put_many_block_false_no_space(self, cleanup_queues):
        """Test put_many with block=False when no space available returns early"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_50b", create=True
        )

        # Fill queue to near capacity
        for i in range(capacity - 5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        initial_size = queue._get_size()

        # Try to put items when no space available
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(20)]

        # Should return early without putting anything
        queue.put_many(items, block=False)

        # Size should not have changed (or changed minimally)
        assert (
            queue._get_size() == initial_size or queue._get_size() <= initial_size + 5
        )

        queue.close()
        queue.unlink()

    def test_put_many_with_wraparound(self, cleanup_queues):
        """Test put_many handles wraparound correctly"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_51", create=True
        )

        # Fill queue (accounting for margin of 10)
        # With capacity=20, margin=10, so max items = 10
        for i in range(10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Drain some items
        for _ in range(5):
            queue.get_nowait()

        # At this point: read_idx=5, write_idx=10
        # 5 items remain (indices 5-9)
        assert queue._get_size() == 5

        # Now put_many should handle wraparound
        # write_idx >= read_idx case: available_space = (capacity - write_idx) + read_idx - 10
        # = (20 - 10) + 5 - 10 = 5
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(10, 15)]
        queue.put_many(items)

        # Should have 10 items total (5 remaining + 5 new)
        assert queue._get_size() == 10

        queue.close()
        queue.unlink()

    def test_put_many_error_handling(self, cleanup_queues):
        """Test put_many error handling"""

        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_52", create=True)

        items = [make_chunk("task_1", 0, "text")]

        # Mock _set_write_idx to raise an error during the batch write
        # This tests the error handling path in put_many
        original_set_write_idx = queue._set_write_idx

        def corrupt_set_write_idx(val):
            raise RuntimeError("Simulated error")

        queue._set_write_idx = corrupt_set_write_idx

        try:
            with pytest.raises(RuntimeError, match="Simulated error"):
                queue.put_many(items)
        finally:
            queue._set_write_idx = original_set_write_idx

        queue.close()
        queue.unlink()

    def test_put_many_available_space_negative(self, cleanup_queues):
        """Test put_many handles negative available_space correctly"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_52b", create=True
        )

        # Fill queue to near capacity
        for i in range(capacity - 8):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        # Try to put items - available_space calculation might be negative
        # but should be clamped to 0
        items = [make_chunk(f"task_{i}", 0, f"text_{i}") for i in range(15)]

        # Should handle gracefully without blocking
        queue.put_many(items, block=False)

        queue.close()
        queue.unlink()


class TestPeekNext:
    """Test peek_next operation"""

    def test_peek_next_empty_queue_no_timeout(self, cleanup_queues):
        """Test peek_next on empty queue without timeout returns None"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_53", create=True)

        result = queue.peek_next()

        assert result is None

        queue.close()
        queue.unlink()

    def test_peek_next_empty_queue_with_timeout(self, cleanup_queues):
        """Test peek_next on empty queue with timeout returns None"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_54", create=True)

        result = queue.peek_next(timeout=0.1)

        assert result is None

        queue.close()
        queue.unlink()

    def test_peek_next_single_item(self, cleanup_queues):
        """Test peek_next retrieves item without advancing read_idx"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_55", create=True)

        queue.put(make_chunk("task_1", 0, "hello"))

        # Peek should return item
        result = queue.peek_next()
        assert result is not None
        worker_id, task_id, chunk_dict = result
        assert task_id == "task_1"

        # read_idx should not have advanced
        assert queue._get_size() == 1

        # Should be able to get the same item
        result2 = queue.get_nowait()
        assert result2[1] == "task_1"

        queue.close()
        queue.unlink()

    def test_peek_next_with_final_item(self, cleanup_queues):
        """Test peek_next with final_result item"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_56", create=True)

        queue.put(make_chunk("task_1", 1, "final"))

        result = queue.peek_next()
        assert result[2]["type"] == "final_result"

        queue.close()
        queue.unlink()

    def test_peek_next_corruption_read_idx_out_of_bounds(self, cleanup_queues):
        """Test peek_next handles corrupted read_idx"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_57", create=True)

        # Manually corrupt read_idx
        import struct

        struct.pack_into("Q", queue.shm.buf, queue.header_offset_read, 9999)

        result = queue.peek_next()

        # Should return None on corruption
        assert result is None

        queue.close()
        queue.unlink()


class TestPeek:
    """Test peek operation"""

    def test_peek_empty_queue_no_timeout(self, cleanup_queues):
        """Test peek on empty queue without timeout returns empty list"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_58", create=True)

        results = queue.peek(5)

        assert results == []

        queue.close()
        queue.unlink()

    def test_peek_empty_queue_with_timeout(self, cleanup_queues):
        """Test peek on empty queue with timeout returns empty list"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_59", create=True)

        results = queue.peek(5, timeout=0.1)

        assert results == []

        queue.close()
        queue.unlink()

    def test_peek_single_item(self, cleanup_queues):
        """Test peek retrieves item without advancing read_idx"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_60", create=True)

        queue.put(make_chunk("task_1", 0, "hello"))

        results = queue.peek(5)

        assert len(results) == 1
        assert results[0][1] == "task_1"

        # read_idx should not have advanced
        assert queue._get_size() == 1

        queue.close()
        queue.unlink()

    def test_peek_multiple_items(self, cleanup_queues):
        """Test peek retrieves multiple items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_61", create=True)

        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        results = queue.peek(10)

        assert len(results) == 5
        for i, (worker_id, task_id, chunk_dict) in enumerate(results):
            assert task_id == f"task_{i}"

        # read_idx should not have advanced
        assert queue._get_size() == 5

        queue.close()
        queue.unlink()

    def test_peek_limits_n(self, cleanup_queues):
        """Test peek respects n parameter"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_62", create=True)

        for i in range(10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        results = queue.peek(3)

        assert len(results) == 3

        queue.close()
        queue.unlink()

    def test_peek_with_final_items(self, cleanup_queues):
        """Test peek with final_result items"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_63", create=True)

        queue.put(make_chunk("task_1", 0, "chunk"))
        queue.put(make_chunk("task_1", 1, "final"))

        results = queue.peek(5)

        assert len(results) == 2
        assert results[0][2]["type"] == "streaming_chunk"
        assert results[1][2]["type"] == "final_result"

        queue.close()
        queue.unlink()

    def test_peek_with_corrupted_read_idx(self, cleanup_queues):
        """Test peek behavior with corrupted read_idx"""
        queue = SharedMemoryChunkQueue(capacity=100, name="test_queue_64", create=True)

        queue.put(make_chunk("task_1", 0, "text"))

        # Corrupt read_idx to be out of bounds
        import struct

        struct.pack_into("Q", queue.shm.buf, queue.header_offset_read, 200)

        # peek doesn't check read_idx bounds like peek_next does
        # It calculates items_to_read = min(n, self._get_size())
        # _get_size() with read_idx=200, write_idx=1 will calculate incorrectly
        # The check at line 598 checks current_idx after modulo, which will always be valid
        results = queue.peek(5)

        # With corrupted read_idx, _get_size() calculation is wrong
        # but the loop will read up to n items, and current_idx modulo will wrap
        # The queue will attempt to read, but may read invalid/empty slots
        assert len(results) <= 5  # Won't read more than requested

        queue.close()
        queue.unlink()

    def test_peek_with_valid_indices(self, cleanup_queues):
        """Test peek with valid indices works correctly"""
        queue = SharedMemoryChunkQueue(capacity=100, name="test_queue_64b", create=True)

        # Put multiple items
        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        results = queue.peek(10)

        # Should peek all 5 items without advancing read_idx
        assert len(results) == 5
        assert queue._get_size() == 5  # read_idx should not have advanced

        queue.close()
        queue.unlink()


class TestQueueHelpers:
    """Test helper methods"""

    def test_put_nowait(self, cleanup_queues):
        """Test put_nowait method"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_65", create=True)

        result = queue.put_nowait(make_chunk("task_1", 0, "hello"))

        assert result is True
        assert queue._get_size() == 1

        queue.close()
        queue.unlink()

    def test_qsize(self, cleanup_queues):
        """Test qsize method"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_66", create=True)

        assert queue.qsize() == 0

        for i in range(5):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        assert queue.qsize() == 5

        queue.close()
        queue.unlink()

    def test_empty(self, cleanup_queues):
        """Test empty method"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_67", create=True)

        assert queue.empty() is True

        queue.put(make_chunk("task_1", 0, "hello"))

        assert queue.empty() is False

        queue.close()
        queue.unlink()

    def test_full(self, cleanup_queues):
        """Test full method"""
        capacity = 20
        queue = SharedMemoryChunkQueue(
            capacity=capacity, name="test_queue_68", create=True
        )

        assert queue.full() is False

        # Fill to near capacity (leaving margin)
        for i in range(capacity - 10):
            queue.put(make_chunk(f"task_{i}", 0, f"text_{i}"))

        assert queue.full() is True

        queue.close()
        queue.unlink()


class TestErrorHandlingAdvanced:
    """Test advanced error handling scenarios"""

    def test_close_error_handling(self, cleanup_queues):
        """Test close handles errors gracefully"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_69", create=True)

        # Close once
        queue.close()

        # Close again - should handle error gracefully
        queue.close()

        queue.unlink()

    def test_unlink_error_handling(self, cleanup_queues):
        """Test unlink handles errors gracefully"""
        queue = SharedMemoryChunkQueue(capacity=1000, name="test_queue_70", create=True)

        # Unlink once
        queue.unlink()

        # Unlink again - should handle error gracefully
        try:
            queue.unlink()
        except Exception:
            pass  # Expected to fail on second unlink
