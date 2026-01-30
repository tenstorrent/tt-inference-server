import struct
import time
from multiprocessing import shared_memory
from typing import Any, List, Optional

import numpy as np
from domain.completion_response import CompletionResult
from utils.logger import TTLogger
from model_services.tt_queue_interface import TTQueueInterface

MAX_TEXT_LEN = 450
MAX_TASK_ID_LEN = 100

chunk_dtype = np.dtype(
    [
        ("task_id", f"U{MAX_TASK_ID_LEN}"),
        ("is_final", "i4"),
        ("text", f"U{MAX_TEXT_LEN}"),
        ("item_available", "i4"),
    ]
)


class SharedMemoryChunkQueue(TTQueueInterface):
    def __init__(self, capacity=200000, name="chunk_queue", create=True):
        self.capacity = capacity
        self.chunk_size = chunk_dtype.itemsize
        self.name = name
        self.logger = TTLogger()

        # Header layout:
        # [write_idx(8)] [write_lock(4)] [pad(52)]
        # [read_idx(8)] [read_lock(4)] [pad(52)]
        # = 128 bytes total, with locks on separate cache lines
        self.header_offset_write = 0
        self.header_offset_write_lock = 8
        self.header_offset_read = 64
        self.header_offset_read_lock = 72
        self.header_size = 128

        total_size = self.header_size + (self.chunk_size * capacity)

        if create:
            try:
                existing_shm = shared_memory.SharedMemory(name=name)
                existing_shm.close()
                existing_shm.unlink()
                print(f"[MemoryQueue] Cleaned up existing: {name}")
            except FileNotFoundError:
                pass
            except Exception as e:
                self.logger.error(f"[MemoryQueue] Cleanup warning: {e}")

            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )

            self._initialize_header()

            self.buffer = np.ndarray(
                (capacity,),
                dtype=chunk_dtype,
                buffer=self.shm.buf,
                offset=self.header_size,
            )

            self.logger.info(
                f"[MemoryQueue] Created: {name}, capacity={capacity}, "
                f"size={total_size / 1024 / 1024:.2f} MB, header_size={self.header_size}"
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name)

            # Calculate capacity from existing shared memory size
            actual_capacity = (self.shm.size - self.header_size) // self.chunk_size
            if capacity != 200000 and capacity != actual_capacity:
                self.logger.warning(
                    f"[MemoryQueue] Capacity mismatch: requested={capacity}, "
                    f"actual={actual_capacity}. Using actual capacity."
                )
            self.capacity = actual_capacity

            # Create buffer reference
            self.buffer = np.ndarray(
                (self.capacity,),
                dtype=chunk_dtype,
                buffer=self.shm.buf,
                offset=self.header_size,
            )

            self.logger.info(
                f"[MemoryQueue] Attached: {name}, capacity={self.capacity}"
            )

    def _initialize_header(self):
        """Initialize header indices and locks"""
        struct.pack_into("Q", self.shm.buf, self.header_offset_write, 0)  # write_idx
        struct.pack_into(
            "I", self.shm.buf, self.header_offset_write_lock, 0
        )  # write_lock
        struct.pack_into("Q", self.shm.buf, self.header_offset_read, 0)  # read_idx
        struct.pack_into(
            "I", self.shm.buf, self.header_offset_read_lock, 0
        )  # read_lock

    def _get_write_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, self.header_offset_write)[0]

    def _get_read_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, self.header_offset_read)[0]

    def _set_write_idx(self, val: int):
        if val < 0:
            raise ValueError(f"write_idx cannot be negative: {val}")

        next_write_idx = val % self.capacity

        current = struct.unpack_from("Q", self.shm.buf, self.header_offset_write)[0]
        if abs(next_write_idx - current) > 1000:
            self.logger.warning(
                f"Large jump in write_idx: {current} → {next_write_idx} "
                f"(from unbounded val={val})"
            )

        struct.pack_into("Q", self.shm.buf, self.header_offset_write, next_write_idx)

    def _set_read_idx(self, val: int):
        if val < 0:
            raise ValueError(f"read_idx cannot be negative: {val}")

        next_read_idx = val % self.capacity

        current = struct.unpack_from("Q", self.shm.buf, self.header_offset_read)[0]
        if abs(next_read_idx - current) > 1000:
            self.logger.warning(
                f"Large jump in read_idx: {current} → {next_read_idx} "
                f"(from unbounded val={val})"
            )

        struct.pack_into("Q", self.shm.buf, self.header_offset_read, next_read_idx)

    def _get_size(self) -> int:
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        if write_idx >= read_idx:
            return write_idx - read_idx
        else:
            return (self.capacity - read_idx) + write_idx

    def full(self) -> bool:
        """Check if the queue is full (or nearly full)."""
        return (
            self._get_size() >= self.capacity - 10
        )  # Same margin as _get_next_write_slot

    def _get_next_write_slot(self) -> int:
        # Rough size check without lock (acceptable race)
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        if write_idx >= read_idx:
            size = write_idx - read_idx
        else:
            size = (self.capacity - read_idx) + write_idx

        if size >= self.capacity - 10:  # Leave margin
            return -1

        write_idx = self._get_write_idx()
        next_write_idx = (write_idx + 1) % self.capacity
        self._set_write_idx(next_write_idx)
        return write_idx

    def put(self, obj, block=True, timeout=None) -> bool:
        """Match multiprocessing.Queue interface.

        Accepts (worker_id, task_id, data) tuple where data is a dict with:
        - type: "streaming_chunk" or "final_result"
        - chunk/result: CompletionResult with .text attribute
        - task_id: str
        """
        # Extract from tuple: (worker_id, task_id, data_dict)
        _, task_id, data = obj

        # Extract fields from the dict
        chunk_type = data.get("type", "streaming_chunk")
        is_final = 1 if chunk_type == "final_result" else 0

        # Get text from chunk or result
        chunk_obj = data.get("data")
        text = chunk_obj.text if chunk_obj else ""

        slot_idx = self._get_next_write_slot()

        if slot_idx == -1:
            return False

        if len(task_id) > MAX_TASK_ID_LEN:
            task_id = task_id[:MAX_TASK_ID_LEN]
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN]

        try:
            self.buffer[slot_idx]["task_id"] = task_id
            self.buffer[slot_idx]["text"] = text
            self.buffer[slot_idx]["is_final"] = is_final
            self.buffer[slot_idx]["item_available"] = 2  # ✅ Mark ready
        except Exception as e:
            self.logger.error(f"Error writing to slot {slot_idx}: {e}")
            raise

        return True

    def get_nowait(self):
        """NON-BLOCKING GET - Returns None if queue is empty"""
        try:
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            if read_idx == write_idx:
                return None

            current_state = int(self.buffer[read_idx]["item_available"])

            if current_state == 0:
                return None

            self.buffer[read_idx]["item_available"] = 0
            self._set_read_idx(read_idx + 1)

            data = self.buffer[read_idx].copy()

            task_id = str(data["task_id"]).rstrip("\x00")
            text = str(data["text"]).rstrip("\x00")

            is_final = int(data["is_final"])

            chunk = CompletionResult(
                text=text,
                index=None,
                finish_reason=None,
            )

            if is_final == 1:
                chunk_dict = {
                    "type": "final_result",
                    "data": chunk,
                }
            else:
                chunk_dict = {
                    "type": "streaming_chunk",
                    "data": chunk,
                }

            return ("1", task_id, chunk_dict)
        except Exception as e:
            self.logger.error(f"Error in get_nowait: {e}")
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def get(self, timeout: float = None):
        start_time = time.time() if timeout else None

        while True:
            if timeout and (time.time() - start_time) > timeout:
                size = self._get_size()
                print(f"[MemoryQueue] GET TIMEOUT - size={size}")
                raise TimeoutError(f"Queue get timed out after {timeout}s")

            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            if read_idx == write_idx:
                time.sleep(0.0001)
                continue

            if read_idx < 0 or read_idx >= self.capacity:
                self.logger.error(f"CORRUPTION: read_idx {read_idx} out of bounds!")
                raise RuntimeError("read_idx corrupted")

            slot_idx = read_idx
            data = self.buffer[slot_idx].copy()
            self.buffer[slot_idx]["item_available"] = 0

            self._set_read_idx(read_idx + 1)
            break

        task_id = str(data["task_id"]).rstrip("\x00")
        text = str(data["text"]).rstrip("\x00")
        is_final = int(data["is_final"])

        chunk = CompletionResult(
            text=text,
            index=None,
            finish_reason=None,
        )

        if is_final == 1:
            chunk_dict = {
                "type": "final_result",
                "data": chunk,
            }
        else:
            chunk_dict = {
                "type": "streaming_chunk",
                "data": chunk,
            }

        return ("1", task_id, chunk_dict)

    def get_many(
        self,
        max_messages_to_get: int = 100,
        block: bool = True,
        timeout: float = None,
    ) -> list:
        """
        Get multiple items at once - more efficient than individual gets.

        Args:
            max_messages_to_get: Maximum number of items to retrieve
            block: If True, wait for at least one item
            timeout: Maximum time to wait (only used if block=True)

        Returns:
            List of items (may be empty if block=False and queue is empty)
        """
        results = []
        start_time = time.time() if timeout else None

        # Get first item (blocking if requested)
        while True:
            if timeout and (time.time() - start_time) > timeout:
                if not results:
                    raise TimeoutError(f"Queue get_many timed out after {timeout}s")
                return results

            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            if read_idx == write_idx:
                if not block:
                    return results
                time.sleep(0.0001)
                continue

            # We have at least one item, break out of the waiting loop
            break

        # Now read as many items as available, up to max_messages_to_get
        items_read = 0
        while items_read < max_messages_to_get:
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            if read_idx == write_idx:
                break

            if read_idx < 0 or read_idx >= self.capacity:
                self.logger.error(f"CORRUPTION: read_idx {read_idx} out of bounds!")
                break

            slot_idx = read_idx
            data = self.buffer[slot_idx].copy()
            self.buffer[slot_idx]["item_available"] = 0
            self._set_read_idx(read_idx + 1)

            task_id = str(data["task_id"]).rstrip("\x00")
            text = str(data["text"]).rstrip("\x00")
            is_final = int(data["is_final"])

            chunk = CompletionResult(
                text=text,
                index=None,
                finish_reason=None,
            )

            if is_final == 1:
                chunk_dict = {
                    "type": "final_result",
                    "data": chunk,
                }
            else:
                chunk_dict = {
                    "type": "streaming_chunk",
                    "data": chunk,
                }

            results.append(("1", task_id, chunk_dict))
            items_read += 1

        return results

    def close(self):
        try:
            self.shm.close()
            print(f"[MemoryQueue] Closed: {self.name}")
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error closing: {e}")

    def join_thread(self):
        """Placeholder method to mimic a multiprocessing queue"""
        return True

    def unlink(self):
        try:
            self.shm.unlink()
            print(f"[MemoryQueue] Unlinked: {self.name}")
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error unlinking: {e}")

    def put_nowait(self, item: Any):
        """Equivalent to put(obj, False)."""
        raise NotImplementedError("put_nowait is not implemented for SharedMemoryChunkQueue")

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        raise NotImplementedError("qsize is not implemented for SharedMemoryChunkQueue")

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise (approximate)."""
        raise NotImplementedError("empty is not implemented for SharedMemoryChunkQueue")

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise (approximate)."""
        return self._get_size() >= self.capacity - 10  # Same margin as _get_next_write_slot

    def put_many(
        self, items: List[Any], block: bool = True, timeout: Optional[float] = None
    ):
        """Put multiple items into the queue."""
        for item in items:
            self.put(item, block=block, timeout=timeout)

    def get_many(
        self,
        max_messages_to_get: int = 100,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Get multiple items from the queue."""
        batch = []

        # Get first item (blocking or non-blocking based on params)
        if block:
            try:
                first_item = self.get(timeout=timeout)
                if first_item is not None:
                    batch.append(first_item)
            except TimeoutError:
                return batch
        else:
            first_item = self.get_nowait()
            if first_item is not None:
                batch.append(first_item)
            else:
                return batch  # Return empty if non-blocking and empty

        # Try to get more items non-blocking
        for _ in range(max_messages_to_get - 1):
            item = self.get_nowait()
            if item is None:
                break
            batch.append(item)

        return batch

    def peek_next(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Peek at next item for conditional processing."""
        raise NotImplementedError("peek_next is not implemented for SharedMemoryChunkQueue")

    def peek(self, n: int, timeout: Optional[float] = None) -> List[Any]:
        """Peek at next n items for conditional processing."""
        raise NotImplementedError("peek is not implemented for SharedMemoryChunkQueue")
