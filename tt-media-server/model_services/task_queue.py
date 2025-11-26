# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
from collections import deque
from multiprocessing import Lock, Semaphore
from multiprocessing.managers import SyncManager


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
 