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
        self._dequeue = dequeue_proxy
        self._sem = sem
        self._lock = lock
        self._max_size = max_size
        self._closed = False

    def put(self, item, timeout=None):
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
        self._sem.acquire()
        with self._lock:
            if self._closed:
                raise ValueError("TaskQueue is closed")
            if not self._dequeue:
                raise Exception("TaskQueue empty")
            return self._dequeue.popleft()

    def get_nowait(self):
        with self._lock:
            if self._closed:
                raise ValueError("TaskQueue is closed")
            if not self._sem.acquire(blocking=False):
                raise Exception("TaskQueue empty")
            return self._dequeue.popleft()

    def get_if_top(self, predicate, timeout=None, **kwargs):
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
        with self._lock:
            return self._max_size > 0 and len(self._dequeue) >= self._max_size

    def qsize(self):
        with self._lock:
            return len(self._dequeue)

    def empty(self):
        with self._lock:
            return len(self._dequeue) == 0

    def close(self):
        with self._lock:
            self._closed = True
            self._dequeue.clear()
            while self._sem.acquire(False):
                pass

    def join_thread(self):
        while True:
            with self._lock:
                if not self._dequeue:
                    return
            time.sleep(seconds=0.001)
 