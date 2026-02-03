# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Request Coalescer - Batches incoming requests before putting them in the task queue.

This solves the PUT starvation problem where many individual put() calls compete
with 32 workers doing get_many() calls. By batching puts, we:
1. Reduce lock contention (1 put_many vs N puts)
2. Allow batches to form in the queue
3. Improve overall throughput
"""

import asyncio
import threading
import time
from typing import Optional

from utils.logger import TTLogger


class RequestCoalescer:
    """
    Accumulates requests and flushes them in batches to reduce lock contention.

    Thread-safe for use with multiple async request handlers.
    """

    def __init__(
        self,
        task_queue,
        max_batch_size: int = 64,
        max_delay_ms: float = 5.0,
    ):
        """
        Args:
            task_queue: The underlying queue that supports put_many()
            max_batch_size: Maximum requests to accumulate before flushing
            max_delay_ms: Maximum time to wait before flushing (milliseconds)
        """
        self.task_queue = task_queue
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms

        self._pending_requests: list = []
        self._lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._last_flush_time = time.time()

        self.logger = TTLogger()

        # Stats for monitoring
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0

    def put(self, request, timeout: Optional[float] = None) -> None:
        """
        Add a request to the pending batch.

        If batch is full, flushes immediately.
        Otherwise, starts/resets a timer to flush after max_delay_ms.
        """
        flush_now = False
        batch_to_flush = None

        with self._lock:
            self._pending_requests.append(request)
            self.total_requests += 1

            if len(self._pending_requests) >= self.max_batch_size:
                # Batch is full, flush immediately
                batch_to_flush = self._pending_requests
                self._pending_requests = []
                flush_now = True
                self._cancel_timer()
            else:
                # Start timer if not already running
                if self._flush_timer is None:
                    self._start_timer()

        # Flush outside the lock to avoid blocking other puts
        if flush_now and batch_to_flush:
            self._flush_batch(batch_to_flush, timeout)

    def _start_timer(self):
        """Start the flush timer."""
        self._flush_timer = threading.Timer(
            self.max_delay_ms / 1000.0, self._timer_flush
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _cancel_timer(self):
        """Cancel the flush timer if running."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _timer_flush(self):
        """Called by timer to flush pending requests."""
        batch_to_flush = None

        with self._lock:
            self._flush_timer = None
            if self._pending_requests:
                batch_to_flush = self._pending_requests
                self._pending_requests = []

        if batch_to_flush:
            self._flush_batch(batch_to_flush, timeout=1.0)

    def _flush_batch(self, batch: list, timeout: Optional[float] = None):
        """Flush a batch of requests to the task queue."""
        if not batch:
            return

        batch_size = len(batch)
        self.total_batches += 1
        self.total_batch_size += batch_size

        try:
            if hasattr(self.task_queue, "put_many"):
                # Use put_many for efficiency - single lock acquisition
                if timeout is not None:
                    self.task_queue.put_many(batch, block=True, timeout=timeout)
                else:
                    self.task_queue.put_many(batch, block=True)
            else:
                # Fallback to individual puts
                for request in batch:
                    self.task_queue.put(request, timeout=timeout)

            avg_batch = self.total_batch_size / max(self.total_batches, 1)
            self.logger.debug(
                f"Coalescer flushed batch_size={batch_size}, "
                f"avg={avg_batch:.1f}, total_batches={self.total_batches}"
            )
        except Exception as e:
            self.logger.error(f"Failed to flush batch: {e}")
            raise

    def flush(self, timeout: Optional[float] = None):
        """Force flush any pending requests."""
        batch_to_flush = None

        with self._lock:
            self._cancel_timer()
            if self._pending_requests:
                batch_to_flush = self._pending_requests
                self._pending_requests = []

        if batch_to_flush:
            self._flush_batch(batch_to_flush, timeout)

    def full(self) -> bool:
        """Check if the underlying queue is full."""
        return self.task_queue.full()

    def qsize(self) -> int:
        """Get approximate queue size including pending."""
        pending = len(self._pending_requests)
        queue_size = (
            self.task_queue.qsize() if hasattr(self.task_queue, "qsize") else 0
        )
        return pending + queue_size

    def get_stats(self) -> dict:
        """Get coalescer statistics."""
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": self.total_batch_size / max(self.total_batches, 1),
            "pending_requests": len(self._pending_requests),
        }

    def shutdown(self):
        """Flush remaining requests and cleanup."""
        self.flush(timeout=2.0)
        self._cancel_timer()
