import struct
import time
from multiprocessing import shared_memory


class SharedRingQueue:
    """Zero-copy ring buffer for streaming text chunks"""

    def __init__(
        self,
        name: str,
        capacity: int = 50000,
        max_text_size: int = 512,
        max_task_id_size: int = 100,
        max_worker_id_size: int = 50,
        create: bool = True,
    ):
        self.capacity = capacity
        self.max_text_size = max_text_size
        self.max_task_id_size = max_task_id_size
        self.max_worker_id_size = max_worker_id_size
        self.name = name

        # Slot structure: [text_len:4][text:max_text_size][index:4][finish_reason:1][task_id_len:4][task_id:max_task_id_size][worker_id_len:4][worker_id:max_worker_id_size]
        self.slot_size = (
            4
            + max_text_size  # text
            + 4
            + 1  # index + finish_reason
            + 4
            + max_task_id_size  # task_id
            + 4
            + max_worker_id_size  # worker_id
        )

        # ✅ ADD HEADER FOR SHARED COUNTERS: [write_idx:8][read_idx:8]
        self.header_size = 16
        self.data_offset = self.header_size
        total_size = self.header_size + (capacity * self.slot_size)

        if create:
            # Clean up existing shared memory first
            try:
                existing_shm = shared_memory.SharedMemory(name=name)
                existing_shm.close()
                existing_shm.unlink()
                print(f"[SharedRingQueue] Cleaned up existing shared memory: {name}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[SharedRingQueue] Cleanup error (ignore): {e}")

            # Create fresh shared memory
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )
            print(f"[SharedRingQueue] Created shared memory: {name}, size={total_size}")

            # ✅ INITIALIZE COUNTERS IN SHARED MEMORY
            struct.pack_into("QQ", self.shm.buf, 0, 0, 0)  # write_idx=0, read_idx=0
            print("[SharedRingQueue] Initialized counters to 0")
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            print(f"[SharedRingQueue] Attached to existing shared memory: {name}")

    def _get_write_idx(self) -> int:
        """Read write index from shared memory"""
        return struct.unpack_from("Q", self.shm.buf, 0)[0]

    def _get_read_idx(self) -> int:
        """Read read index from shared memory"""
        return struct.unpack_from("Q", self.shm.buf, 8)[0]

    def _set_write_idx(self, value: int):
        """Write write index to shared memory"""
        struct.pack_into("Q", self.shm.buf, 0, value)

    def _set_read_idx(self, value: int):
        """Write read index to shared memory"""
        struct.pack_into("Q", self.shm.buf, 8, value)

    def put(self, worker_id: str, task_id: str, chunk) -> bool:
        """Put a chunk object directly into shared memory - NO SERIALIZATION"""
        try:
            # Extract data from chunk object
            if hasattr(chunk, "chunk"):
                text = chunk.chunk.text
                index = chunk.chunk.index
                finish_reason = getattr(chunk.chunk, "finish_reason", None)
            elif hasattr(chunk, "text"):
                text = chunk.text
                index = getattr(chunk, "index", 0)
                finish_reason = getattr(chunk, "finish_reason", None)
            else:
                text = str(chunk)
                index = 0
                finish_reason = None

            text_bytes = text.encode("utf-8")
            if len(text_bytes) > self.max_text_size:
                text_bytes = text_bytes[: self.max_text_size]

            task_id_bytes = task_id.encode("utf-8")
            if len(task_id_bytes) > self.max_task_id_size:
                task_id_bytes = task_id_bytes[: self.max_task_id_size]

            worker_id_bytes = worker_id.encode("utf-8")
            if len(worker_id_bytes) > self.max_worker_id_size:
                worker_id_bytes = worker_id_bytes[: self.max_worker_id_size]

            # ✅ READ FROM SHARED MEMORY
            write_idx = self._get_write_idx()
            read_idx = self._get_read_idx()

            print(
                f"[SharedRingQueue.put] BEFORE: write_idx={write_idx}, read_idx={read_idx}, task_id={task_id[:20]}"
            )

            # Check if buffer is full
            if write_idx - read_idx >= self.capacity:
                print(
                    f"[SharedRingQueue.put] BUFFER FULL! write_idx={write_idx}, read_idx={read_idx}"
                )
                return False  # Buffer full

            slot_idx = write_idx % self.capacity
            offset = self.data_offset + (slot_idx * self.slot_size)

            pos = offset

            # Write text
            struct.pack_into("I", self.shm.buf, pos, len(text_bytes))
            pos += 4
            self.shm.buf[pos : pos + len(text_bytes)] = text_bytes
            pos += self.max_text_size

            # Write index
            struct.pack_into("I", self.shm.buf, pos, index)
            pos += 4

            # Write finish_reason
            reason_byte = 0
            if finish_reason == "stop":
                reason_byte = 1
            elif finish_reason == "length":
                reason_byte = 2
            struct.pack_into("B", self.shm.buf, pos, reason_byte)
            pos += 1

            # Write task_id
            struct.pack_into("I", self.shm.buf, pos, len(task_id_bytes))
            pos += 4
            self.shm.buf[pos : pos + len(task_id_bytes)] = task_id_bytes
            pos += self.max_task_id_size

            # Write worker_id
            struct.pack_into("I", self.shm.buf, pos, len(worker_id_bytes))
            pos += 4
            self.shm.buf[pos : pos + len(worker_id_bytes)] = worker_id_bytes

            # ✅ UPDATE WRITE INDEX IN SHARED MEMORY
            self._set_write_idx(write_idx + 1)

            print(
                f"[SharedRingQueue.put] AFTER: write_idx={write_idx + 1}, text_len={len(text)}"
            )

            return True
        except Exception as e:
            print(f"[SharedRingQueue.put] EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get(self, timeout: float = None):
        """Get a chunk from shared memory - BLOCKING until data available or timeout
        Returns the reconstructed chunk object, not a tuple"""
        start = time.time()

        print(f"[SharedRingQueue.get] CALLED (BLOCKING mode, timeout={timeout})")

        # ✅ BLOCK INDEFINITELY UNTIL DATA AVAILABLE (or timeout)
        while True:
            # ✅ READ FROM SHARED MEMORY
            read_idx = self._get_read_idx()
            write_idx = self._get_write_idx()

            # Check for timeout
            if timeout is not None and (time.time() - start) >= timeout:
                print(f"[SharedRingQueue.get] TIMEOUT after {timeout}s")
                return None

            if read_idx >= write_idx:
                # No data available, sleep briefly and retry
                time.sleep(0.0001)  # 0.1ms
                continue

            # Data is available!
            print(
                f"[SharedRingQueue.get] DATA AVAILABLE! Reading slot {read_idx}, write_idx={write_idx}"
            )

            slot_idx = read_idx % self.capacity
            offset = self.data_offset + (slot_idx * self.slot_size)

            pos = offset

            try:
                # Read text
                text_len = struct.unpack_from("I", self.shm.buf, pos)[0]
                pos += 4
                text_bytes = bytes(self.shm.buf[pos : pos + text_len])
                text = text_bytes.decode("utf-8")
                pos += self.max_text_size

                # Read index
                index = struct.unpack_from("I", self.shm.buf, pos)[0]
                pos += 4

                # Read finish_reason
                reason_byte = struct.unpack_from("B", self.shm.buf, pos)[0]
                finish_reason = None
                if reason_byte == 1:
                    finish_reason = "stop"
                elif reason_byte == 2:
                    finish_reason = "length"
                pos += 1

                # Read task_id
                task_id_len = struct.unpack_from("I", self.shm.buf, pos)[0]
                pos += 4
                task_id_bytes = bytes(self.shm.buf[pos : pos + task_id_len])
                task_id = task_id_bytes.decode("utf-8")
                pos += self.max_task_id_size

                # Read worker_id
                worker_id_len = struct.unpack_from("I", self.shm.buf, pos)[0]
                pos += 4
                worker_id_bytes = bytes(self.shm.buf[pos : pos + worker_id_len])
                worker_id = worker_id_bytes.decode("utf-8")

                # ✅ UPDATE READ INDEX IN SHARED MEMORY
                self._set_read_idx(read_idx + 1)

                print(
                    f"[SharedRingQueue.get] SUCCESS: task_id={task_id[:20]}, text_len={len(text)}"
                )

                # ✅ RECONSTRUCT THE CHUNK OBJECT HERE
                from domain.completion_response import (
                    CompletionStreamChunk,
                    StreamingChunkOutput,
                )

                chunk = StreamingChunkOutput(
                    type="streaming_chunk",
                    chunk=CompletionStreamChunk(
                        text=text,
                        index=index,
                        finish_reason=finish_reason,
                    ),
                    task_id=task_id,
                )

                # Return tuple format that scheduler expects: (worker_id, result_key, chunk_object)
                return (worker_id, task_id, chunk)

            except Exception as e:
                print(f"[SharedRingQueue.get] PARSE ERROR: {e}")
                import traceback

                traceback.print_exc()
                self._set_read_idx(read_idx + 1)  # Skip corrupted slot
                continue

    def close(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except:
            pass
