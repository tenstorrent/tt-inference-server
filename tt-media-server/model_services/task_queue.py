import threading
from collections import deque

class TaskQueue:
    def __init__(self, max_size: int = 0):
        self._queue = deque()
        self._lock = threading.Lock()
        self._max_size = max_size

    def put(self, item):
        with self._lock:
            if self._max_size > 0 and len(self._queue) >= self._max_size:
                raise Exception("TaskQueue full")
            self._queue.append(item)

    def get(self):
        with self._lock:
            if not self._queue:
                raise Exception("TaskQueue empty")
            return self._queue.popleft()

    def get_nowait(self):
        return self.get()

    def get_if_top(self, predicate):
        with self._lock:
            if not self._queue:
                return None
            head = self._queue[0]
            if predicate(head):
                return self._queue.popleft()
            return None

    def full(self):
        with self._lock:
            return self._max_size > 0 and len(self._queue) >= self._max_size

    def qsize(self):
        with self._lock:
            return len(self._queue)

    def empty(self):
        with self._lock:
            return len(self._queue) == 0