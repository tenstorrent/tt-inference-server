# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import multiprocessing
from queue import Empty
from typing import Any, List, Optional
import time

from model_services.queues.tt_queue_interface import TTQueueInterface


class TTBatchFifoQueue(TTQueueInterface):
    """
    Batch-aware queue using only multiprocessing.Queue for better pickling.
    Uses smart batching logic to improve worker utilization.
    """

    def __init__(self, max_size: int = 0, batch_size: int = 1, *, ctx=None):
        if ctx is None:
            ctx = multiprocessing.get_context()

        # Use simple multiprocessing Queue - fully picklable
        self._queue = ctx.Queue(maxsize=max_size if max_size > 0 else 0)
        self._batch_size = batch_size
        self._max_size = max_size
        self._ctx = ctx

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
        """Put multiple items at once."""
        for item in items:
            if timeout is not None:
                self._queue.put(item, block=block, timeout=timeout)
            else:
                self._queue.put(item, block=block)

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
        try:
            return self._queue.get(block=False)
        except Empty:
            return None

    def get_many(
        self,
        max_messages_to_get: int = 100,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> List:
        """
        Smart batching get_many that tries to get full batches.
        Uses exponential backoff to wait for more items when queue has some but not enough.
        """
        items = []

        # First, try to get at least one item
        try:
            if timeout is not None:
                first_item = self._queue.get(block=block, timeout=timeout)
            else:
                first_item = self._queue.get(block=block)
            items.append(first_item)
        except Empty:
            return []

        # Now try to get more items to fill the batch
        batch_timeout = min(timeout or 1.0, 1.0)  # Max 1 second for batch filling
        start_time = time.time()

        while (
            len(items) < max_messages_to_get
            and (time.time() - start_time) < batch_timeout
        ):
            try:
                # Try to get more items with increasingly shorter timeouts
                remaining_time = batch_timeout - (time.time() - start_time)
                item_timeout = min(remaining_time, 0.1)  # Max 100ms per item

                if item_timeout <= 0:
                    break

                item = self._queue.get(block=True, timeout=item_timeout)
                items.append(item)

                # If we got a full batch, return immediately
                if len(items) >= self._batch_size:
                    break

            except Empty:
                # No more items available, check if we should wait longer
                if (
                    len(items) < self._batch_size
                    and (time.time() - start_time) < batch_timeout * 0.5
                ):
                    # Wait a bit longer for more items to arrive
                    time.sleep(0.01)
                else:
                    # Give up waiting
                    break

        return items

    def qsize(self) -> int:
        """Approximate queue size."""
        try:
            return self._queue.qsize()
        except Exception:
            return 0

    def empty(self) -> bool:
        """Check if queue is empty."""
        try:
            return self._queue.empty()
        except Exception:
            return True

    def full(self) -> bool:
        """Check if queue is full."""
        try:
            return self._queue.full()
        except Exception:
            return False

    def peek_next(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Peek at next item for conditional processing."""
        raise NotImplementedError("peek_next is not implemented for TTBatchFifoQueue")

    def peek(self, n: int, timeout: Optional[float] = None) -> List[Any]:
        """Peek at next n items for conditional processing."""
        raise NotImplementedError("peek is not implemented for TTBatchFifoQueue")

    def join_thread(self):
        """Join the background feeder thread (if applicable)."""
        if hasattr(self._queue, "join_thread"):
            self._queue.join_thread()

    def close(self):
        """Close the queue."""
        try:
            self._queue.close()
        except Exception:
            pass

    def __getstate__(self):
        """Support for pickling."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Support for unpickling."""
        self.__dict__.update(state)
