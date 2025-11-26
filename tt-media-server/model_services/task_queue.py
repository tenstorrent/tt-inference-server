# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
from collections import deque
from multiprocessing import Lock, Semaphore
from multiprocessing.managers import SyncManager
from utils.logger import TTLogger


class TaskQueueManager(SyncManager):
    pass

def make_deque():
    return deque()

TaskQueueManager.register(
    'deque',
    callable=make_deque,
    exposed=['append', 'pop', 'popleft', '__len__', '__getitem__', 'clear']
)


def make_managed_task_queue(manager, max_size=0):
    dequeue_proxy = manager.deque()
    sem = Semaphore(0)
    lock = Lock()
    return TaskQueue(dequeue_proxy, sem, lock, max_size=max_size)

class TaskQueue:
    def __init__(self, dequeue_proxy, sem, lock, max_size=0):
        """
        Initialize TaskQueue with managed deque, semaphore, lock, and max size.
        Args:
            dequeue_proxy: Manager-backed deque proxy.
            sem: Semaphore for item counting.
            lock: Lock for synchronization.
            max_size: Maximum queue size (0 for infinite).
        Returns:
            None
        Raises:
            Nothing
        """
        self._dequeue = dequeue_proxy
        self._sem = sem
        self._lock = lock
        self._max_size = max_size
        self._closed = False

    def put(self, item, timeout=None):
        """
        Put item in queue with optional timeout.
        Args:
            item: Item to add.
            timeout: Timeout in milliseconds (None for infinite).
        Returns:
            None
        Raises:
            Exception: If queue is closed.
            TimeoutError: If operation times out.
        """
        start = time.time()
        timeout = timeout / 1000 if timeout is not None else None
        while True:
            with self._lock:
                if self._closed:
                    raise Exception("TaskQueue is closed")
                if self._max_size == 0 or len(self._dequeue) < self._max_size:
                    self._dequeue.append(item)
                    self._sem.release()
                    return
            if timeout is None or (time.time() - start) >= timeout:
                raise TimeoutError("TaskQueue put timed out")
            time.sleep(seconds=0.001)

    def get(self):
        """
        Get item from queue, block if empty.
        Returns:
            Item from front of queue
        Raises:
            ValueError: If queue is closed.
            Exception: If queue is empty.
        """
        self._sem.acquire()
        with self._lock:
            if self._closed:
                raise ValueError("TaskQueue is closed")
            if not self._dequeue:
                raise Exception("TaskQueue empty")
            return self._dequeue.popleft()

    def get_nowait(self):
        """
        Get item from queue, no block.
        Returns:
            Item from front of queue
        Raises:
            ValueError: If queue is closed.
            Exception: If queue is empty.
        """
        with self._lock:
            if self._closed:
                raise ValueError("TaskQueue is closed")
            if not self._sem.acquire(blocking=False):
                raise Exception("TaskQueue empty")
            return self._dequeue.popleft()

    def get_if_top(self, predicate, timeout=None, **kwargs):
        """
        Get front item if predicate matches, with optional timeout.
        Args:
            predicate: Function to test front item.
            timeout: Timeout in milliseconds (None for infinite).
            **kwargs: Additional arguments for predicate.
        Returns:
            Item from front of queue if predicate matches
        Raises:
            ValueError: If queue is closed.
            Exception: If queue is empty or timeout occurs.
        """
        start = time.time()
        timeout = timeout / 1000 if timeout is not None else None
        while True:
            with self._lock:
                if self._closed:
                    raise ValueError("TaskQueue is closed")
                if self._sem.acquire(False):
                    if self._dequeue and predicate(self._dequeue[0], **kwargs):
                        return self._dequeue.popleft()
                    else:
                        self._sem.release()
            if timeout is None or (time.time() - start) >= timeout:
                raise Exception("TaskQueue empty")
            time.sleep(seconds=0.001)

    def full(self):
        """
        Check if queue is full.
        Returns:
            bool
        Raises:
            Nothing
        """
        with self._lock:
            return self._max_size > 0 and len(self._dequeue) >= self._max_size

    def qsize(self):
        """
        Get current queue size.
        Returns:
            int
        Raises:
            Nothing
        """
        with self._lock:
            return len(self._dequeue)

    def close(self):
        """
        Close queue, clear items, disallow puts.
        Returns:
            None
        Raises:
            Nothing
        """
        with self._lock:
            self._closed = True
            self._dequeue.clear()
            while self._sem.acquire(False):
                pass

    def join_thread(self):
        """
        Block until queue is empty.
        Returns:
            None
        Raises:
            Nothing
        """
        while True:
            with self._lock:
                if not self._dequeue:
                    return
            time.sleep(seconds=0.001)
            
    @staticmethod
    def get_greedy_batch(task_queue, max_batch_size, max_batch_delay_time_ms, batching_predicate):
        """
        Collects a batch of items from the queue, starting with a blocking get, then greedily
        attempts to add more items using get_if_top with a timeout. Handles shutdown signals and errors gracefully.
        Args:
            task_queue: The queue to get items from.
            max_batch_size: Maximum number of items in the batch.
            max_batch_delay_time_ms: Timeout for greedy batching (ms).
            batching_predicate: Predicate to determine batch eligibility.
        Returns:
            List of items (batch), may contain None as shutdown signal.
        """
        logger = TTLogger()
        batch = []

        # Get first item (blocking)
        try:
            first_item = task_queue.get()
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received - shutting down gracefully")
            return [None]
        except Exception as e:
            logger.error(f"Error getting first item from queue: {e}")
            return [None]

        if first_item is None:
            return [None]
        batch.append(first_item)

        # Greedily try to get more items
        timeout = max_batch_delay_time_ms
        for _ in range(max_batch_size - 1):
            try:
                item = task_queue.get_if_top(
                    batching_predicate,
                    timeout=timeout,
                    batch=batch
                )
            except Exception as e:
                logger.debug(f"Stopped greedy batching: {e}")
                break
            timeout = None  # Only use timeout for the first greedy get
            if item is None:
                batch.append(None)
                break
            batch.append(item)

        return batch
 