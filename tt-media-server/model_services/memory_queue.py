# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import struct
import threading
import time
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
from utils.logger import TTLogger

MAX_TEXT_LEN = 450
MAX_SLOTS = 1000  # Maximum concurrent requests
CHUNKS_PER_SLOT = 4000  # Maximum chunks per request/slot

# Chunk data structure
chunk_dtype = np.dtype(
    [
        ("is_final", "i4"),
        ("text", f"U{MAX_TEXT_LEN}"),
        ("item_available", "i4"),
    ]
)

# Slot header structure (per-slot read/write indices)
# Each slot has: write_idx (8 bytes), read_idx (8 bytes), in_use flag (4 bytes), padding (12 bytes) = 32 bytes
SLOT_HEADER_SIZE = 32

# Global header structure (128 bytes):
# - bytes 0-7: capacity_per_slot (Q)
# - bytes 8-15: max_slots (Q)
# - bytes 16-23: chunk_size (Q)
# - bytes 24-127: reserved
GLOBAL_HEADER_SIZE = 128


class SharedMemoryChunkQueue:
    """
    Shared memory queue with support for multiple concurrent request slots.

    Memory Layout:
    - Global header (128 bytes): metadata including capacity_per_slot, max_slots
    - Slot headers (MAX_SLOTS * SLOT_HEADER_SIZE): per-slot read/write indices
    - Chunk data (MAX_SLOTS * CHUNKS_PER_SLOT * chunk_size): actual data

    Each slot has its own circular buffer of CHUNKS_PER_SLOT chunks.
    Writers attach with a specific offset (slot_id) to write to their reserved slot.
    """

    def __init__(
        self,
        capacity_per_slot: int = CHUNKS_PER_SLOT,
        max_slots: int = MAX_SLOTS,
        name: str = "chunk_queue",
        create: bool = True,
        slot_id: Optional[int] = None,
    ):
        """
        Initialize the shared memory queue.

        Args:
            capacity_per_slot: Number of chunks per slot (default: 4000) - ignored when create=False
            max_slots: Maximum number of concurrent slots/requests (default: 1000) - ignored when create=False
            name: Shared memory name
            create: If True, create new shared memory; if False, attach to existing
            slot_id: Slot ID for writers to attach to (None for creator/manager)
        """
        self.name = name
        self.slot_id = slot_id
        self.chunk_size = chunk_dtype.itemsize
        self.logger = TTLogger()

        # Global header layout
        self.global_header_size = GLOBAL_HEADER_SIZE
        self.slot_manager = SlotManager(self)

        if create:
            # Use provided parameters for creation
            self.capacity_per_slot = capacity_per_slot
            self.max_slots = max_slots
            self._calculate_layout()
            total_size = self.data_offset + (self.chunk_size * self.total_chunks)
            self._create_shared_memory(name, total_size)
        else:
            # Attach first, then read parameters from header
            self._attach_shared_memory(name)

    def _calculate_layout(self):
        """Calculate memory layout based on capacity_per_slot and max_slots."""
        # Slot headers layout
        self.slot_headers_offset = self.global_header_size
        self.slot_headers_size = self.max_slots * SLOT_HEADER_SIZE

        # Chunk data layout
        self.data_offset = self.slot_headers_offset + self.slot_headers_size
        self.total_chunks = self.max_slots * self.capacity_per_slot

    def _create_shared_memory(self, name: str, total_size: int):
        """Create new shared memory and initialize headers."""
        try:
            existing_shm = shared_memory.SharedMemory(name=name)
            existing_shm.close()
            existing_shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Cleanup warning: {e}")

        self.shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
        self._write_global_header()
        self._initialize_all_headers()
        self._create_buffer()

    def _attach_shared_memory(self, name: str):
        """Attach to existing shared memory and read configuration from header."""
        self.shm = shared_memory.SharedMemory(name=name)
        self._read_global_header()
        self._calculate_layout()
        self._create_buffer()

    def _write_global_header(self):
        """Write configuration to global header."""
        struct.pack_into("Q", self.shm.buf, 0, self.capacity_per_slot)
        struct.pack_into("Q", self.shm.buf, 8, self.max_slots)
        struct.pack_into("Q", self.shm.buf, 16, self.chunk_size)

    def _read_global_header(self):
        """Read configuration from global header."""
        self.capacity_per_slot = struct.unpack_from("Q", self.shm.buf, 0)[0]
        self.max_slots = struct.unpack_from("Q", self.shm.buf, 8)[0]
        stored_chunk_size = struct.unpack_from("Q", self.shm.buf, 16)[0]

        # Validate chunk size matches
        if stored_chunk_size != 0 and stored_chunk_size != self.chunk_size:
            self.logger.warning(
                f"[MemoryQueue] Chunk size mismatch: stored={stored_chunk_size}, current={self.chunk_size}"
            )

        self.logger.debug(
            f"[MemoryQueue] Attached with capacity_per_slot={self.capacity_per_slot}, "
            f"max_slots={self.max_slots}"
        )

    def _create_buffer(self):
        """Create numpy buffer view over shared memory."""
        self.buffer = np.ndarray(
            (self.total_chunks,),
            dtype=chunk_dtype,
            buffer=self.shm.buf,
            offset=self.data_offset,
        )

    def _initialize_all_headers(self):
        """Initialize global header and all slot headers."""
        # Initialize all slot headers
        for slot in range(self.max_slots):
            self._initialize_slot_header(slot)

    def _initialize_slot_header(self, slot_id: int):
        """Initialize a single slot's header."""
        base = self._get_slot_header_offset(slot_id)
        struct.pack_into("Q", self.shm.buf, base, 0)  # write_idx
        struct.pack_into("Q", self.shm.buf, base + 8, 0)  # read_idx
        struct.pack_into("I", self.shm.buf, base + 16, 0)  # in_use flag (0 = free)
        struct.pack_into("I", self.shm.buf, base + 20, 0)  # reserved

    def _get_slot_header_offset(self, slot_id: int) -> int:
        """Get the byte offset for a slot's header."""
        return self.slot_headers_offset + (slot_id * SLOT_HEADER_SIZE)

    def _get_slot_data_offset(self, slot_id: int) -> int:
        """Get the starting chunk index for a slot's data."""
        return slot_id * self.capacity_per_slot

    # Slot header accessors
    def _get_write_idx(self, slot_id: int) -> int:
        base = self._get_slot_header_offset(slot_id)
        return struct.unpack_from("Q", self.shm.buf, base)[0]

    def _set_write_idx(self, slot_id: int, val: int):
        base = self._get_slot_header_offset(slot_id)
        # Store raw value, NOT modulo - we need to track total writes
        struct.pack_into("Q", self.shm.buf, base, val)

    def _get_read_idx(self, slot_id: int) -> int:
        base = self._get_slot_header_offset(slot_id)
        return struct.unpack_from("Q", self.shm.buf, base + 8)[0]

    def _set_read_idx(self, slot_id: int, val: int):
        base = self._get_slot_header_offset(slot_id)
        # Store raw value, NOT modulo - we need to track total reads
        struct.pack_into("Q", self.shm.buf, base + 8, val)

    def _get_slot_in_use(self, slot_id: int) -> bool:
        base = self._get_slot_header_offset(slot_id)
        return struct.unpack_from("I", self.shm.buf, base + 16)[0] == 1

    def _set_slot_in_use(self, slot_id: int, in_use: bool):
        base = self._get_slot_header_offset(slot_id)
        struct.pack_into("I", self.shm.buf, base + 16, 1 if in_use else 0)

    def _get_available_to_read(self, slot_id: int) -> int:
        """Get number of items available to read."""
        write_idx = self._get_write_idx(slot_id)
        read_idx = self._get_read_idx(slot_id)
        # Simple subtraction works because we store raw indices
        return write_idx - read_idx

    def _get_available_to_write(self, slot_id: int) -> int:
        """Get number of slots available for writing."""
        # Leave 1 slot empty to distinguish full from empty
        available_to_read = self._get_available_to_read(slot_id)
        return self.capacity_per_slot - 1 - available_to_read

    def _get_chunk_index(self, slot_id: int, raw_idx: int) -> int:
        """Convert raw index to global buffer index (applies modulo here)."""
        local_idx = raw_idx % self.capacity_per_slot
        return self._get_slot_data_offset(slot_id) + local_idx

    # Write operations (for workers)
    def put(self, is_final: int, text: str, slot_id: Optional[int] = None) -> bool:
        """
        Write a chunk to a slot.

        Args:
            is_final: 1 if this is the final chunk, 0 otherwise
            text: Text content to write
            slot_id: Slot to write to (uses self.slot_id if not provided)

        Returns:
            True if successful, False if slot is full
        """
        slot = slot_id if slot_id is not None else self.slot_id
        if slot is None:
            raise ValueError("No slot_id specified for write operation")

        # Check if there's room to write
        if self._get_available_to_write(slot) < 1:
            return False

        write_idx = self._get_write_idx(slot)

        # Truncate text if needed
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN]

        # Write to buffer - modulo applied in _get_chunk_index
        chunk_idx = self._get_chunk_index(slot, write_idx)
        self.buffer[chunk_idx]["text"] = text
        self.buffer[chunk_idx]["is_final"] = is_final
        self.buffer[chunk_idx]["item_available"] = 1

        # Advance write index (raw, no modulo)
        self._set_write_idx(slot, write_idx + 1)
        return True

    def put_blocking(
        self,
        is_final: int,
        text: str,
        slot_id: Optional[int] = None,
        timeout: float = 1.0,
    ) -> bool:
        """
        Write a chunk to a slot, blocking until space is available.

        Args:
            is_final: 1 if this is the final chunk, 0 otherwise
            text: Text content to write
            slot_id: Slot to write to (uses self.slot_id if not provided)
            timeout: Maximum time to wait in seconds

        Returns:
            True if successful, False on timeout
        """
        deadline = time.perf_counter() + timeout
        spin_count = 0

        while True:
            if self.put(is_final, text, slot_id):
                return True

            if time.perf_counter() >= deadline:
                return False

            spin_count += 1
            if spin_count > 100:
                time.sleep(0.0001)  # 100μs backoff
                spin_count = 0

    # Read operations (for consumers)
    def get_nowait(self, slot_id: int) -> Optional[tuple]:
        """
        Non-blocking read from a slot.

        Args:
            slot_id: Slot to read from

        Returns:
            (is_final, text) tuple or None if no data available
        """
        if self._get_available_to_read(slot_id) == 0:
            return None

        read_idx = self._get_read_idx(slot_id)

        # Modulo applied in _get_chunk_index
        chunk_idx = self._get_chunk_index(slot_id, read_idx)

        if self.buffer[chunk_idx]["item_available"] == 0:
            return None

        # Mark as read
        self.buffer[chunk_idx]["item_available"] = 0

        # Read data
        is_final = int(self.buffer[chunk_idx]["is_final"])
        text = str(self.buffer[chunk_idx]["text"]).rstrip("\x00")

        # Advance read index (raw, no modulo)
        self._set_read_idx(slot_id, read_idx + 1)

        return (is_final, text)

    def read_batch(self, slot_id: int, max_items: int = 1000) -> list:
        """
        Read all available data from a slot (batch read).
        """
        available = self._get_available_to_read(slot_id)

        # Nothing to read
        if available == 0:
            return []

        results = []
        read_idx = self._get_read_idx(slot_id)
        items_to_read = min(max_items, available)
        slot_offset = self._get_slot_data_offset(slot_id)
        capacity = self.capacity_per_slot
        buffer = self.buffer  # Local reference for speed

        for _ in range(items_to_read):
            # Calculate chunk index with wraparound (modulo applied here)
            local_idx = read_idx % capacity
            chunk_idx = slot_offset + local_idx

            if buffer[chunk_idx]["item_available"] == 0:
                break

            # Mark as read
            buffer[chunk_idx]["item_available"] = 0

            # Read data
            is_final = int(buffer[chunk_idx]["is_final"])
            text = str(buffer[chunk_idx]["text"]).rstrip("\x00")

            # Advance read index (raw, no modulo)
            read_idx += 1

            # Add to results
            results.append((is_final, text))

            # Stop after final chunk
            if is_final == 1:
                break

        # Update read index (raw value)
        self._set_read_idx(slot_id, read_idx)

        return results

    def read_all_slots_batch(
        self, slot_ids: list[int], max_items_per_slot: int = 100
    ) -> dict[int, list[tuple[int, str]]]:
        """
        Batch read from multiple slots at once.

        Reads all available data from each specified slot and returns
        a dictionary mapping slot IDs to their data.

        Args:
            slot_ids: List of slot IDs to read from
            max_items_per_slot: Maximum number of items to read in one batch

        Returns:
            Dictionary mapping slot_id -> list of (is_final, text) tuples.
            Only slots with data are included in the result.

        Example:
            {
                0: [(0, "token1"), (0, "token2"), (1, "[DONE]")],
                3: [(0, "hello"), (0, "world")],
            }
        """
        results = {}

        for slot_id in slot_ids:
            batch = self.read_batch(slot_id, max_items_per_slot)
            if batch:  # Only include slots that have data
                results[slot_id] = batch

        return results

    def read_all_active_slots_batch(
        self, max_items_per_slot: int = 15000
    ) -> dict[int, list[tuple[int, str]]]:
        """
        Batch read from ALL active (in-use) slots.

        Uses SlotManager to get only allocated slots instead of scanning all slots.

        Args:
            max_items_per_slot: Maximum items to read per slot

        Returns:
            Dictionary mapping slot_id -> list of (is_final, text) tuples.
            Only slots with data are included in the result.

        Example:
            {
                0: [(0, "token1"), (0, "token2")],
                5: [(0, "data"), (1, "[DONE]")],
                12: [(0, "more"), (0, "tokens")],
            }
        """
        results = {}

        # Get only the allocated slots from slot_manager
        allocated_slots = self.slot_manager.get_all_allocated_slots()

        for slot_id in allocated_slots:
            batch = self.read_batch(slot_id, max_items_per_slot)
            if batch:  # Only include slots that have data
                results[slot_id] = batch

        return results

    def get_blocking(self, slot_id: int, timeout: float = 0.001) -> Optional[tuple]:
        """
        Blocking read with timeout.

        Args:
            slot_id: Slot to read from
            timeout: Maximum time to wait in seconds

        Returns:
            (is_final, text) tuple or None on timeout
        """
        deadline = time.perf_counter() + timeout
        spin_count = 0
        max_spins = 100

        while True:
            result = self.get_nowait(slot_id)
            if result is not None:
                return result

            if time.perf_counter() >= deadline:
                return None

            spin_count += 1
            if spin_count > max_spins:
                time.sleep(0.00001)  # 10μs
                spin_count = 0

    async def get_async(self, slot_id: int, timeout: float = 0.1) -> Optional[tuple]:
        """
        Async-friendly blocking read.

        Args:
            slot_id: Slot to read from
            timeout: Maximum time to wait in seconds

        Returns:
            (is_final, text) tuple or None on timeout
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.get_blocking,
            slot_id,
            timeout,
        )

    def get_slot_size(self, slot_id: int) -> int:
        """Get the number of unread items in a slot."""
        return self._get_available_to_read(slot_id)

    def is_slot_empty(self, slot_id: int) -> bool:
        """Check if a slot has no unread data."""
        return self._get_available_to_read(slot_id) == 0

    def close(self):
        """Close the shared memory (does not unlink)."""
        try:
            self.shm.close()
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error closing: {e}")

    def unlink(self):
        """Unlink (delete) the shared memory."""
        try:
            self.shm.unlink()
        except Exception as e:
            self.logger.error(f"[MemoryQueue] Error unlinking: {e}")


class SlotManager:
    """
    Thread-safe manager for allocating and freeing slots in SharedMemoryChunkQueue.

    This manager handles:
    - Reserving slots for new requests with thread-safe locking
    - Freeing slots when requests complete
    - Tracking which slots are in use and which request IDs own them
    - Circular buffer allocation pattern for efficient slot reuse

    Usage:
        manager = SlotManager(queue)
        slot_id = manager.reserve_slot(request_id="abc-123")  # Get a free slot
        # ... use slot_id for reading/writing ...
        manager.free_slot(slot_id)  # Release the slot
    """

    def __init__(self, queue: SharedMemoryChunkQueue):
        """
        Initialize the slot manager.

        Args:
            queue: The SharedMemoryChunkQueue to manage slots for
        """
        self.queue = queue
        self.lock = threading.Lock()
        self.logger = TTLogger()

        # Local tracking of allocated slots (for faster lookups)
        self._allocated_slots: set = set()

        # Mapping: slot_id -> request_id
        self._slot_to_request: dict[int, str] = {}

        # Mapping: request_id -> slot_id (reverse lookup)
        self._request_to_slot: dict[str, int] = {}

        # Circular buffer pointer - next slot to check for allocation
        self._next_slot_pointer: int = 0

    def reserve_slot(self, request_id: str) -> Optional[int]:
        """
        Reserve a free slot for a new request.

        Args:
            request_id: UUID4 string identifying the request

        Returns:
            slot_id if successful, None if no slots available

        Thread-safe: Uses locking to prevent race conditions.
        """
        with self.lock:
            # Check if request already has a slot
            if request_id in self._request_to_slot:
                existing_slot = self._request_to_slot[request_id]
                self.logger.warning(
                    f"[SlotManager] Request {request_id} already has slot {existing_slot}"
                )
                return existing_slot

            slot_id = self._find_next_free_slot_unlocked()
            if slot_id is None:
                self.logger.warning("[SlotManager] No free slots available")
                return None

            # Reserve the slot
            self.queue._set_slot_in_use(slot_id, True)
            self.queue._initialize_slot_header(slot_id)
            self.queue._set_slot_in_use(slot_id, True)  # Re-set after init

            # Track allocation
            self._allocated_slots.add(slot_id)
            self._slot_to_request[slot_id] = request_id
            self._request_to_slot[request_id] = slot_id

            self.logger.debug(
                f"[SlotManager] Reserved slot {slot_id} for request {request_id}"
            )
            return slot_id

    def _find_next_free_slot_unlocked(self) -> Optional[int]:
        """
        Find the next free slot using circular buffer pattern.
        Must be called with lock held.

        Returns:
            slot_id if found, None if all slots are full
        """
        max_slots = self.queue.max_slots

        # Start from current pointer and wrap around
        for _ in range(max_slots):
            slot_id = self._next_slot_pointer

            # Advance pointer (circular)
            self._next_slot_pointer = (self._next_slot_pointer + 1) % max_slots

            # Check if slot is free
            if slot_id not in self._allocated_slots and not self.queue._get_slot_in_use(
                slot_id
            ):
                return slot_id

        return None

    def get_next_free_slot(self) -> Optional[int]:
        """
        Get the next free slot ID without reserving it.
        Useful for checking availability before committing.

        Returns:
            slot_id if a free slot exists, None if all slots are full

        Thread-safe: Uses locking.
        """
        with self.lock:
            max_slots = self.queue.max_slots
            pointer = self._next_slot_pointer

            for _ in range(max_slots):
                slot_id = pointer
                pointer = (pointer + 1) % max_slots

                if (
                    slot_id not in self._allocated_slots
                    and not self.queue._get_slot_in_use(slot_id)
                ):
                    return slot_id

            return None

    def release_slot(self, slot_id: int) -> bool:
        """
        Free a previously reserved slot.

        Args:
            slot_id: The slot to free

        Returns:
            True if successful, False if slot was not allocated

        Thread-safe: Uses locking to prevent race conditions.
        """
        with self.lock:
            if slot_id < 0 or slot_id >= self.queue.max_slots:
                self.logger.error(f"[SlotManager] Invalid slot_id: {slot_id}")
                return False

            if slot_id not in self._allocated_slots:
                self.logger.warning(
                    f"[SlotManager] Slot {slot_id} was not allocated by this manager"
                )
                return False

            # Get request_id for logging
            request_id = self._slot_to_request.get(slot_id, "unknown")

            # Clear the slot data
            self._clear_slot_data(slot_id)

            # Mark as free
            self.queue._set_slot_in_use(slot_id, False)
            self._allocated_slots.discard(slot_id)

            # Remove from mappings
            if slot_id in self._slot_to_request:
                del self._slot_to_request[slot_id]
            if request_id in self._request_to_slot:
                del self._request_to_slot[request_id]

            self.logger.debug(
                f"[SlotManager] Freed slot {slot_id} (was request {request_id})"
            )
            return True

    def free_slot_by_request(self, request_id: str) -> bool:
        """
        Free a slot by request ID.

        Args:
            request_id: The request ID whose slot should be freed

        Returns:
            True if successful, False if request not found

        Thread-safe: Uses locking.
        """
        with self.lock:
            if request_id not in self._request_to_slot:
                self.logger.warning(
                    f"[SlotManager] Request {request_id} has no allocated slot"
                )
                return False

            slot_id = self._request_to_slot[request_id]

            # Clear the slot data
            self._clear_slot_data(slot_id)

            # Mark as free
            self.queue._set_slot_in_use(slot_id, False)
            self._allocated_slots.discard(slot_id)

            # Remove from mappings
            del self._slot_to_request[slot_id]
            del self._request_to_slot[request_id]

            self.logger.debug(
                f"[SlotManager] Freed slot {slot_id} for request {request_id}"
            )
            return True

    def get_slot_by_request(self, request_id: str) -> Optional[int]:
        """
        Get the slot ID for a given request ID.

        Args:
            request_id: The request ID to look up

        Returns:
            slot_id if found, None otherwise

        Thread-safe: Uses locking.
        """
        with self.lock:
            return self._request_to_slot.get(request_id)

    def get_request_by_slot(self, slot_id: int) -> Optional[str]:
        """
        Get the request ID for a given slot ID.

        Args:
            slot_id: The slot ID to look up

        Returns:
            request_id if found, None otherwise

        Thread-safe: Uses locking.
        """
        with self.lock:
            return self._slot_to_request.get(slot_id)

    def _clear_slot_data(self, slot_id: int):
        """Clear all data in a slot's buffer."""
        start_idx = self.queue._get_slot_data_offset(slot_id)
        end_idx = start_idx + self.queue.capacity_per_slot

        for idx in range(start_idx, end_idx):
            self.queue.buffer[idx]["item_available"] = 0
            self.queue.buffer[idx]["is_final"] = 0
            self.queue.buffer[idx]["text"] = ""

    def get_allocated_count(self) -> int:
        """Get the number of currently allocated slots."""
        with self.lock:
            return len(self._allocated_slots)

    def get_free_count(self) -> int:
        """Get the number of available slots."""
        with self.lock:
            return self.queue.max_slots - len(self._allocated_slots)

    def is_slot_allocated(self, slot_id: int) -> bool:
        """Check if a slot is currently allocated."""
        with self.lock:
            return slot_id in self._allocated_slots

    def get_all_allocated_slots(self) -> list:
        """Get a list of all currently allocated slot IDs."""
        with self.lock:
            return list(self._allocated_slots)

    def get_all_request_mappings(self) -> dict[str, int]:
        """
        Get a copy of all request_id -> slot_id mappings.

        Returns:
            Dictionary mapping request IDs to slot IDs

        Thread-safe: Uses locking.
        """
        with self.lock:
            return dict(self._request_to_slot)

    def force_free_all(self):
        """
        Force free all slots (use with caution, for cleanup/reset).

        This will free all slots regardless of their state.
        """
        with self.lock:
            for slot_id in list(self._allocated_slots):
                self._clear_slot_data(slot_id)
                self.queue._set_slot_in_use(slot_id, False)

            self._allocated_slots.clear()
            self._slot_to_request.clear()
            self._request_to_slot.clear()
            self._next_slot_pointer = 0
            self.logger.info("[SlotManager] Force freed all slots")
