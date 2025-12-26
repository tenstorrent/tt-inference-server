import struct
import time
from multiprocessing import shared_memory

import numpy as np
from domain.completion_response import CompletionStreamChunk

MAX_TEXT_LEN = 256
MAX_TASK_ID_LEN = 36  # UUID length

chunk_dtype = np.dtype(
    [
        ("task_id", f"S{MAX_TASK_ID_LEN}"),
        ("is_final", "i4"),
        ("text", f"S{MAX_TEXT_LEN}"),  # ✅ FIX: Use MAX_TEXT_LEN not MAX_TASK_ID_LEN
    ]
)


class SharedMemoryChunkQueue:
    def __init__(self, capacity=50000, name="chunk_queue", create=True):
        self.capacity = capacity
        self.chunk_size = chunk_dtype.itemsize
        self.name = name

        self.header_size = 16
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
                print(f"[MemoryQueue] Cleanup warning: {e}")

            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )
            struct.pack_into("QQ", self.shm.buf, 0, 0, 0)
            print(
                f"[MemoryQueue] Created: {name}, capacity={capacity}, size={total_size} bytes"
            )
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"[MemoryQueue] Attached: {name}")

        self.buffer = np.ndarray(
            (capacity,),
            dtype=chunk_dtype,
            buffer=self.shm.buf,
            offset=self.header_size,
        )

    def _get_write_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, 0)[0]

    def _get_read_idx(self) -> int:
        return struct.unpack_from("Q", self.shm.buf, 8)[0]

    def _set_write_idx(self, val: int):
        struct.pack_into("Q", self.shm.buf, 0, val)

    def _set_read_idx(self, val: int):
        struct.pack_into("Q", self.shm.buf, 8, val)

    def put(self, task_id: str, is_final: int, text: str):
        while True:
            write_idx = self._get_write_idx()
            read_idx = self._get_read_idx()

            if write_idx - read_idx >= self.capacity:
                time.sleep(0.0001)
                continue

            slot_idx = write_idx % self.capacity

            # TRUNCATE STRING FIRST, then encode
            # This ensures we don't split multi-byte UTF-8 characters
            if len(text) > MAX_TEXT_LEN:
                text = text[:MAX_TEXT_LEN]

            task_id_bytes = task_id.encode("utf-8")[:MAX_TASK_ID_LEN]
            text_bytes = text.encode("utf-8")

            # Double-check byte length (shouldn't exceed MAX_TEXT_LEN * 4 for UTF-8)
            if len(text_bytes) > MAX_TEXT_LEN:
                # Rare case: multi-byte chars made it too long, truncate safely
                text = text[: MAX_TEXT_LEN // 4]  # Conservative truncation
                text_bytes = text.encode("utf-8")

            # Write data
            self.buffer[slot_idx]["task_id"] = task_id_bytes
            self.buffer[slot_idx]["text"] = text_bytes
            self.buffer[slot_idx]["is_final"] = is_final

            # MEMORY BARRIER - Force write to be visible to other processes
            self._set_write_idx(write_idx + 1)
            return

    def get(self, timeout: float = None):
        while True:
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            if read_idx >= write_idx:
                continue  # ✅ SPIN - no sleep!

            slot_idx = read_idx % self.capacity
            data = self.buffer[slot_idx]

            task_id = data["task_id"].tobytes().decode("utf-8").rstrip("\x00")
            text = data["text"].tobytes().decode("utf-8").rstrip("\x00")
            is_final = int(data["is_final"])

            self._set_read_idx(read_idx + 1)

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

    def get_nowait(self):
        """NON-BLOCKING GET - Returns None if queue is empty"""
        read_idx = self._get_read_idx()
        write_idx = self._get_write_idx()

        if read_idx >= write_idx:
            return None  # Queue is empty

        slot_idx = read_idx % self.capacity
        data = self.buffer[slot_idx]

        # SAFE DECODE - handle truncated UTF-8
        try:
            task_id = data["task_id"].tobytes().decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            task_id = (
                data["task_id"]
                .tobytes()
                .decode("utf-8", errors="ignore")
                .rstrip("\x00")
            )

        try:
            text = data["text"].tobytes().decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            # FALLBACK: Decode with error handling
            text = (
                data["text"].tobytes().decode("utf-8", errors="ignore").rstrip("\x00")
            )
            print(
                f"[MemoryQueue] Warning: Truncated UTF-8 in text, recovered: {text[:50]}..."
            )

        is_final = int(data["is_final"])

        self._set_read_idx(read_idx + 1)

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
            print(f"[MemoryQueue] Error closing: {e}")

    def unlink(self):
        try:
            self.shm.unlink()
            print(f"[MemoryQueue] Unlinked: {self.name}")
        except Exception as e:
            print(f"[MemoryQueue] Error unlinking: {e}")
