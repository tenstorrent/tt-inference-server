# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        """Put multiple items at once - faster than individual puts."""
        if timeout is not None:
            self._queue.put_many(items, block=block, timeout=timeout)
        else:
            self._queue.put_many(items, block=block)

    def get(self, block: bool = True, timeout: Optional[float] = None):
        """Get a single item from the queue."""
        if timeout is not None:
            return self._queue.get(block=block, timeout=timeout)
        else:
            return self._queue.get(block=block)

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
        except Exception:
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
