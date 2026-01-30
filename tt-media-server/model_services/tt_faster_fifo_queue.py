# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from queue import Empty
from typing import Any, List, Optional

from faster_fifo import Queue as FasterFifoQueue
from model_services.tt_queue_interface import TTQueueInterface


class TTFasterFifoQueue(TTQueueInterface):
    """
    Uses faster-fifo for high-performance queuing.
    """

    def __init__(self, max_size: int = 0, *, ctx=None):
        # faster-fifo uses max_size_bytes, estimate ~20KB per item
        # If max_size=0, use a large default
        max_bytes = max(max_size * 20240, 10 * 1024 * 1024)  # At least 10MB
        self._queue = FasterFifoQueue(max_size_bytes=max_bytes)
        self._max_size = max_size

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """Put a single item into the queue."""
        if timeout is not None:
            self._queue.put(item, block=block, timeout=timeout)
        else:
            self._queue.put(item, block=block)

    def put_nowait(self, item):
        """Non-blocking put."""
        self._queue.put(item, block=False)

    def put_many(
        self, items: List, block: bool = True, timeout: Optional[float] = None
    ):
        """
        Put multiple items at once - faster than individual puts.

        Note: The timeout applies to the entire batch, not to each item individually.
        """
        if timeout is not None:
            self._queue.put_many(items, block=block, timeout=timeout)
        else:
            self._queue.put_many(items, block=block)

    def get(self, block: bool = True, timeout: Optional[float] = None):
        """Get a single item from the queue."""
        try:
            if timeout is not None:
                return self._queue.get(block=block, timeout=timeout)
            else:
                return self._queue.get(block=block)
        except Empty:
            return None

    def get_nowait(self):
        """Non-blocking get."""
        return self._queue.get(block=False)

    def get_many(
        self,
        max_messages_to_get: int = 100,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> List:
        """
        Get multiple items at once - much faster than individual gets.

        Note: The timeout applies to the entire batch, not to each item individually.

        Args:
            max_messages_to_get: Maximum number of items to retrieve
            block: If True, wait for at least one item
            timeout: Maximum time to wait (only used if block=True)

        Returns:
            List of items (may be empty if block=False and queue is empty)
        """
        try:
            if timeout is not None:
                return self._queue.get_many(
                    max_messages_to_get=max_messages_to_get,
                    block=block,
                    timeout=timeout,
                )
            else:
                return self._queue.get_many(
                    max_messages_to_get=max_messages_to_get, block=block
                )
        except Empty:
            return []

    def qsize(self) -> int:
        """Approximate queue size."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def full(self) -> bool:
        """Check if queue is full (approximate)."""
        if self._max_size <= 0:
            return False
        return self.qsize() >= self._max_size

    def peek_next(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Peek at next item for conditional processing."""
        raise NotImplementedError("peek_next is not implemented for TTFasterFifoQueue")

    def peek(self, n: int, timeout: Optional[float] = None) -> List[Any]:
        """Peek at next n items for conditional processing."""
        raise NotImplementedError("peek is not implemented for TTFasterFifoQueue")


from asyncio import Semaphore


class SlotManager:
    """
    Manages a pool of N slots (each slot is a TTFasterFifoQueue).

    Thread-safety: The claim/release operations are NOT thread-safe.
    If multiple threads need to claim/release slots concurrently,
    external synchronization is required.
    """

    def __init__(
        self, num_slots: int = 64, slot_size_bytes: int = 16 * 1024, **queue_kwargs
    ):
        """
        Create a slot manager with a fixed number of pre-allocated slots.

        Args:
            num_slots: Number of slots to create.
            slot_size_bytes: Size of each slot's circular buffer in bytes.
            **queue_kwargs: Additional arguments passed to Queue constructor
                           (e.g., maxsize, loads, dumps).
        """
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")

        self._slots: list = [
            TTFasterFifoQueue(slot_size_bytes, **queue_kwargs) for _ in range(num_slots)
        ]
        self._free_slots: set = set(range(num_slots))
        self._claimed_slots: set = set()
        self._semaphore = Semaphore(num_slots)

    async def claim_slot(self) -> tuple[int, TTFasterFifoQueue]:
        """
        Claim a free slot from the pool.

        Returns:
            Tuple of (slot_id, queue) where slot_id can be sent to workers
            and queue can be used to read results.

        Raises:
            RuntimeError: If no free slots are available.
        """
        await self._semaphore.acquire()

        slot_id = self._free_slots.pop()
        self._claimed_slots.add(slot_id)
        return slot_id, self._slots[slot_id]

    def release_slot(self, slot_id: int) -> None:
        """
        Release a slot back to the pool.

        Note: This does NOT clear the queue. If there are unread messages,
        they will still be there when the slot is claimed again. Call
        slot.get_many_nowait() to drain it if needed before releasing.

        Args:
            slot_id: The slot ID returned from claim_slot().

        Raises:
            ValueError: If the slot_id is not currently claimed.
        """
        if slot_id not in self._claimed_slots:
            raise ValueError(f"Slot {slot_id} is not claimed")

        self._claimed_slots.remove(slot_id)
        self._free_slots.add(slot_id)
        self._semaphore.release()

    def get_slot(self, slot_id: int) -> TTFasterFifoQueue:
        """
        Get a slot by ID.

        This is typically used by workers who receive the slot_id via a work queue
        and need to write results to it.

        Args:
            slot_id: The slot ID.

        Returns:
            The Queue instance for this slot.

        Raises:
            IndexError: If slot_id is out of range.
        """
        return self._slots[slot_id]

    @property
    def num_free(self) -> int:
        """Number of slots currently available for claiming."""
        return len(self._free_slots)

    @property
    def num_claimed(self) -> int:
        """Number of slots currently in use."""
        return len(self._claimed_slots)

    @property
    def num_slots(self) -> int:
        """Total number of slots in the pool."""
        return len(self._slots)
