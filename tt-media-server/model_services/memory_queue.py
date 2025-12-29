import struct
import time
from multiprocessing import Lock, shared_memory

import numpy as np
from domain.completion_response import CompletionStreamChunk
from utils.logger import TTLogger

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


class SharedMemoryChunkQueue:
    def __init__(self, capacity=200000, name="chunk_queue", create=True):
        self.capacity = capacity
        self.chunk_size = chunk_dtype.itemsize
        self.name = name
        self.logger = TTLogger()

        self.write_lock = Lock()
        self.read_lock = Lock()

        # ✅ ROBUST: Add guards around header
        # Header layout: [GUARD(8)] [write_idx(8)] [read_idx(8)] [GUARD(8)] = 32 bytes
        self.guard_value = 0xDEADBEEFDEADBEEF
        self.header_offset_write = 8  # After first guard
        self.header_offset_read = 16  # After write_idx
        self.header_size = 32  # With guards

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

            # ✅ Initialize guards and indices
            self._initialize_header()

            # ✅ Create buffer reference AFTER header with guards
            self.buffer = np.ndarray(
                (capacity,),
                dtype=chunk_dtype,
                buffer=self.shm.buf,
                offset=self.header_size,
            )

            self.logger.error(
                f"[MemoryQueue] Created: {name}, capacity={capacity}, "
                f"size={total_size / 1024 / 1024:.2f} MB, header_size={self.header_size}"
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name)

            # ✅ Create buffer reference
            self.buffer = np.ndarray(
                (capacity,),
                dtype=chunk_dtype,
                buffer=self.shm.buf,
                offset=self.header_size,
            )

            # ✅ Verify guards on attach
            self._verify_guards()
            self.logger.info(f"[MemoryQueue] Attached: {name}")

    def _initialize_header(self):
        """✅ Initialize header with guards"""
        struct.pack_into("Q", self.shm.buf, 0, self.guard_value)  # First guard
        struct.pack_into("Q", self.shm.buf, self.header_offset_write, 0)  # write_idx
        struct.pack_into("Q", self.shm.buf, self.header_offset_read, 0)  # read_idx
        struct.pack_into("Q", self.shm.buf, 24, self.guard_value)  # Second guard

    def _verify_guards(self):
        """✅ Verify guards haven't been corrupted"""
        guard1 = struct.unpack_from("Q", self.shm.buf, 0)[0]
        guard2 = struct.unpack_from("Q", self.shm.buf, 24)[0]

        if guard1 != self.guard_value or guard2 != self.guard_value:
            self.logger.error(
                f"CORRUPTION DETECTED: Guards corrupted! "
                f"guard1={hex(guard1)}, guard2={hex(guard2)}, expected={hex(self.guard_value)}"
            )
            raise RuntimeError("Header guards corrupted - buffer overflow detected!")

    def _get_write_idx(self) -> int:
        """✅ Get write index with guard verification"""
        self._verify_guards()
        return struct.unpack_from("Q", self.shm.buf, self.header_offset_write)[0]

    def _get_read_idx(self) -> int:
        """✅ Get read index with guard verification"""
        self._verify_guards()
        read_idx = struct.unpack_from("Q", self.shm.buf, self.header_offset_read)[0]
        if read_idx > 0:
            self.logger.error(f"Read read_idx: {read_idx}")
        return read_idx

    def _set_write_idx(self, val: int):
        """✅ Set bounded write index with wraparound and guard check"""
        # ✅ Validate input BEFORE modulo
        if val < 0:
            raise ValueError(f"write_idx cannot be negative: {val}")

        # ✅ Verify guards BEFORE writing
        self._verify_guards()

        next_write_idx = val % self.capacity

        # ✅ Debug: log suspicious jumps
        current = struct.unpack_from("Q", self.shm.buf, self.header_offset_write)[0]
        if abs(next_write_idx - current) > 1000:
            self.logger.warning(
                f"Large jump in write_idx: {current} → {next_write_idx} "
                f"(from unbounded val={val})"
            )

        struct.pack_into("Q", self.shm.buf, self.header_offset_write, next_write_idx)

        # ✅ Verify guards AFTER writing
        self._verify_guards()
        self.logger.error(f"Set write_idx to {next_write_idx}")

    def _set_read_idx(self, val: int):
        """✅ Set bounded read index with wraparound and guard check"""
        # ✅ Validate input BEFORE modulo
        if val < 0:
            raise ValueError(f"read_idx cannot be negative: {val}")

        # ✅ Verify guards BEFORE writing
        self._verify_guards()

        next_read_idx = val % self.capacity

        # ✅ Debug: log suspicious jumps
        current = struct.unpack_from("Q", self.shm.buf, self.header_offset_read)[0]
        if abs(next_read_idx - current) > 1000:
            self.logger.error(
                f"Large jump in read_idx: {current} → {next_read_idx} "
                f"(from unbounded val={val})"
            )

        struct.pack_into("Q", self.shm.buf, self.header_offset_read, next_read_idx)

        # ✅ Verify guards AFTER writing
        self._verify_guards()
        self.logger.error(f"Set read_idx to {next_read_idx}")

    def _get_size(self) -> int:
        """✅ Calculate size for bounded circular buffer"""
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        # ✅ For bounded indices, calculate distance accounting for wraparound
        if write_idx >= read_idx:
            size = write_idx - read_idx
        else:
            # ✅ write_idx wrapped around, reader is behind
            size = (self.capacity - read_idx) + write_idx

        return size

    def _get_next_write_slot(self) -> int:
        """✅ Atomically get next write slot using a lock"""
        with self.write_lock:
            write_idx = self._get_write_idx()
            read_idx = self._get_read_idx()

            # ✅ Check if queue is full for bounded indices
            if write_idx >= read_idx:
                size = write_idx - read_idx
            else:
                size = (self.capacity - read_idx) + write_idx

            if size >= self.capacity:
                self.logger.warning(
                    f"Queue full: write={write_idx}, read={read_idx}, size={size}"
                )
                return -1

            # ✅ Reserve this slot and increment write_idx atomically
            slot_idx = write_idx
            next_write_idx = (write_idx + 1) % self.capacity
            if (write_idx + 1) % self.capacity == 0:
                self.logger.error("Write index wrapped around to 0")
            if abs(next_write_idx - slot_idx) > 500:
                self.logger.error(
                    f"CORRUPTION: next_write_idx {next_write_idx} equals read_idx {read_idx}!"
                )
            if next_write_idx == 1:
                self.logger.error("Next write_idx is 1")
            self._set_write_idx(next_write_idx)

            return slot_idx

    def put(self, task_id: str, is_final: int, text: str) -> bool:
        """✅ Thread-safe put with locked write_idx increment"""

        # ✅ Use lock for slots to make sure it's a unique slot
        slot_idx = self._get_next_write_slot()

        if slot_idx == -1:
            self.logger.warning("Queue is full, dropping item")
            return False

        # ✅ Truncate strings to max length STRICTLY
        if len(task_id) > MAX_TASK_ID_LEN:
            task_id = task_id[:MAX_TASK_ID_LEN]

        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN]

        # ✅ Validate before writing to avoid overflow
        assert len(task_id) <= MAX_TASK_ID_LEN, f"task_id too long: {len(task_id)}"
        assert len(text) <= MAX_TEXT_LEN, f"text too long: {len(text)}"

        try:
            # ✅ Write data to the slot (exclusive - only we have this slot_idx)
            self.buffer[slot_idx]["task_id"] = task_id
            self.buffer[slot_idx]["text"] = text
            self.buffer[slot_idx]["is_final"] = is_final
            self.buffer[slot_idx]["item_available"] = 1

            # ✅ Verify guards after writing
            self._verify_guards()
        except Exception as e:
            self.logger.error(f"Error writing to slot {slot_idx}: {e}")
            raise

        return True

    async def get_nowait(self):
        """NON-BLOCKING GET - Returns None if queue is empty"""
        try:
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            # ✅ Check if empty - for bounded indices, compare directly
            if read_idx == write_idx:
                return None

            # ✅ Slot index is already the bounded index
            slot_idx = read_idx

            # ✅ Verify slot_idx is within bounds
            if slot_idx < 0 or slot_idx >= self.capacity:
                self.logger.error(f"CORRUPTION: slot_idx {slot_idx} out of bounds!")
                return None

            # Copy data before updating index
            data = self.buffer[slot_idx].copy()

            if not data:
                return None

            if int(data["item_available"]) == 0:
                return None

            if int(data["item_available"]) == 2:
                self._set_read_idx(read_idx + 1)
                return None

            # Mark read so others do not read it again
            self.buffer[slot_idx]["item_available"] = 2

            # ✅ Update read index with wraparound
            self._set_read_idx(read_idx + 1)

            # Convert to strings
            task_id = str(data["task_id"]).rstrip("\x00")
            text = str(data["text"]).rstrip("\x00")

            # Validate task_id
            if not task_id or len(task_id) == 0:
                self.logger.error(f"Empty task_id at slot {slot_idx}")
                return None

            is_final = int(data["is_final"])

            chunk = CompletionStreamChunk(
                text=text,
                index=None,
                finish_reason=None,
            )

            if is_final == 1:
                chunk_dict = {
                    "type": "final_result",
                    "result": chunk,
                    "task_id": task_id,
                    "return_result": False,
                }
            else:
                chunk_dict = {
                    "type": "streaming_chunk",
                    "chunk": chunk,
                    "task_id": task_id,
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

            # ✅ Check if empty for bounded indices
            if read_idx == write_idx:
                time.sleep(0.0001)
                continue

            # ✅ Verify bounds
            if read_idx < 0 or read_idx >= self.capacity:
                self.logger.error(f"CORRUPTION: read_idx {read_idx} out of bounds!")
                raise RuntimeError("read_idx corrupted")

            slot_idx = read_idx
            data = self.buffer[slot_idx].copy()
            self.buffer[slot_idx]["item_available"] = 0

            # ✅ Update read index with wraparound
            self._set_read_idx(read_idx + 1)
            break

        task_id = str(data["task_id"]).rstrip("\x00")
        text = str(data["text"]).rstrip("\x00")
        is_final = int(data["is_final"])

        chunk = CompletionStreamChunk(
            text=text,
            index=None,
            finish_reason=None,
        )

        if is_final == 1:
            chunk_dict = {
                "type": "final_result",
                "result": chunk,
                "task_id": task_id,
                "return_result": False,
            }
        else:
            chunk_dict = {
                "type": "streaming_chunk",
                "chunk": chunk,
                "task_id": task_id,
            }

        return ("1", task_id, chunk_dict)

    def close(self):
        try:
            self.shm.close()
            print(f"[MemoryQueue] Closed: {self.name}")
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error closing: {e}")

    def unlink(self):
        try:
            self.shm.unlink()
            print(f"[MemoryQueue] Unlinked: {self.name}")
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error unlinking: {e}")
