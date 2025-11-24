from multiprocessing import Condition
from multiprocessing.managers import SyncManager
from collections import deque

class TaskQueue:
    def __init__(self, max_size: int = 0, condition=None):
        self._queue = deque()
        self._max_size = max_size
        self._condition = condition if condition is not None else Condition()

    def put(self, item):
        with self._condition:
            if self._max_size > 0 and len(self._queue) >= self._max_size:
                raise Exception("TaskQueue full")
            self._queue.append(item)
            self._condition.notify()  # Wake up one waiting get()

    def get(self):
        with self._condition:
            while not self._queue:
                self._condition.wait()  # Block until an item is available
            return self._queue.popleft()

    def get_nowait(self):
        with self._condition:
            if not self._queue:
                raise Exception("TaskQueue empty")
            return self._queue.popleft()

    def get_if_top(self, predicate):
        with self._condition:
            if not self._queue:
                return None
            head = self._queue[0]
            if predicate(head):
                return self._queue.popleft()
            return None

    def full(self):
        with self._condition:
            return self._max_size > 0 and len(self._queue) >= self._max_size

    def qsize(self):
        with self._condition:
            return len(self._queue)

    def empty(self):
        with self._condition:
            return len(self._queue) == 0

def make_task_queue(max_size=0):
    # Create a Condition for cross-process blocking
    cond = Condition()
    return TaskQueue(max_size=max_size, condition=cond)

class TaskQueueManager(SyncManager):
    pass

TaskQueueManager.register('TaskQueue', callable=make_task_queue)