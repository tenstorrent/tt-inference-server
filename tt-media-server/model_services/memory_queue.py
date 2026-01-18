import struct
import time
from multiprocessing import shared_memory

import numpy as np
from utils.logger import TTLogger

MAX_TEXT_LEN = 450

chunk_dtype = np.dtype(
    [
        ("is_final", "i4"),
        ("text", f"U{MAX_TEXT_LEN}"),
        ("item_available", "i4"),
    ]
)


class SharedMemoryChunkQueue:
    def __init__(self, capacity=2000, name="chunk_queue", create=True):
        self.capacity = capacity
        self.chunk_size = chunk_dtype.itemsize
        self.name = name
        self.logger = TTLogger()

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
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self.buffer = np.ndarray(
                (capacity,),
                dtype=chunk_dtype,
                buffer=self.shm.buf,
                offset=self.header_size,
            )

    def _initialize_header(self):
        struct.pack_into("Q", self.shm.buf, self.header_offset_write, 0)
        struct.pack_into("I", self.shm.buf, self.header_offset_write_lock, 0)
        struct.pack_into("Q", self.shm.buf, self.header_offset_read, 0)
        struct.pack_into("I", self.shm.buf, self.header_offset_read_lock, 0)

    def _get_write_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, self.header_offset_write)[0]

    def _get_read_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, self.header_offset_read)[0]

    def _set_write_idx(self, val: int):
        next_write_idx = val % self.capacity
        struct.pack_into("Q", self.shm.buf, self.header_offset_write, next_write_idx)

    def _set_read_idx(self, val: int):
        next_read_idx = val % self.capacity
        struct.pack_into("Q", self.shm.buf, self.header_offset_read, next_read_idx)

    def _get_next_write_slot(self) -> int:
        write_idx = self._get_write_idx()
        read_idx = self._get_read_idx()

        if write_idx >= read_idx:
            size = write_idx - read_idx
        else:
            size = (self.capacity - read_idx) + write_idx

        if size >= self.capacity - 10:
            return -1

        next_write_idx = (write_idx + 1) % self.capacity
        self._set_write_idx(next_write_idx)
        return write_idx

    def put(self, is_final: int, text: str) -> bool:
        slot_idx = self._get_next_write_slot()
        if slot_idx == -1:
            return False

        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN]

        self.buffer[slot_idx]["text"] = text
        self.buffer[slot_idx]["is_final"] = is_final
        self.buffer[slot_idx]["item_available"] = 1
        return True

    def get_nowait_raw(self):
        """
        Non-blocking get. Returns (is_final, text) or None.
        """
        read_idx = self._get_read_idx()
        write_idx = self._get_write_idx()

        if read_idx == write_idx:
            return None

        if self.buffer[read_idx]["item_available"] == 0:
            return None

        self.buffer[read_idx]["item_available"] = 0
        self._set_read_idx(read_idx + 1)

        # Read directly without copy
        is_final = int(self.buffer[read_idx]["is_final"])
        text = str(self.buffer[read_idx]["text"]).rstrip("\x00")

        return (is_final, text)

    def get_blocking(self, timeout: float = 0.001):
        """
        ✅ Blocking get with short timeout.
        More efficient than polling with sleep.

        Returns (is_final, text) or None on timeout.
        """
        deadline = time.perf_counter() + timeout
        spin_count = 0
        max_spins = 100  # Spin briefly before sleeping

        while True:
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            # Check if data available
            if read_idx != write_idx and self.buffer[read_idx]["item_available"] != 0:
                self.buffer[read_idx]["item_available"] = 0
                self._set_read_idx(read_idx + 1)

                is_final = int(self.buffer[read_idx]["is_final"])
                text = str(self.buffer[read_idx]["text"]).rstrip("\x00")
                return (is_final, text)

            # Check timeout
            if time.perf_counter() >= deadline:
                return None

            # Spin briefly, then sleep
            spin_count += 1
            if spin_count > max_spins:
                time.sleep(0.00001)  # 10μs - much shorter than asyncio.sleep minimum
                spin_count = 0

    async def get_async(self, timeout: float = 0.1):
        """
        ✅ Async-friendly blocking get.
        Runs blocking get in thread pool to not block event loop.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Default thread pool
            self.get_blocking,
            timeout,
        )

    def close(self):
        try:
            self.shm.close()
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error closing: {e}")

    def unlink(self):
        try:
            self.shm.unlink()
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error unlinking: {e}")
