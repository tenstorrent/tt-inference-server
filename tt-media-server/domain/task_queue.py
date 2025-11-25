from multiprocessing import Condition
from multiprocessing.managers import SyncManager
from collections import deque

def make_managed_task_queue(manager, max_size=0):
    """
    Factory to create a manager-backed TaskQueue with shared queue and condition.
    Usage:
        manager = TaskQueueManager()
        manager.start()
        tq = make_managed_task_queue(manager, max_size=100)
    """
    queue_proxy = manager.deque()
    condition_proxy = manager.Condition()
    return TaskQueue(queue_proxy, condition_proxy, max_size=max_size)

def make_deque():
    return deque()

def make_condition():
    return Condition()
class TaskQueueManager(SyncManager):
    pass

TaskQueueManager.register('deque', callable=make_deque)
TaskQueueManager.register('Condition', callable=make_condition)

class TaskQueue:
    def __init__(self, queue_proxy, condition_proxy, max_size=0):
        self._queue = queue_proxy
        self._condition = condition_proxy
        self._max_size = max_size
        self._closed = False

    def put(self, item):
        with self._condition:
            if self._closed:
                raise Exception("TaskQueue is closed")
            if self._max_size > 0 and len(self._queue) >= self._max_size:
                raise Exception("TaskQueue full")
            self._queue.append(item)
            self._condition.notify()

    def get(self):
        with self._condition:
            if not self._queue:
                raise Exception("TaskQueue empty")
            return self._queue.popleft()

    def get_nowait(self):
        with self._condition:
            if not self._queue:
                raise Exception("TaskQueue empty")
            return self._queue.popleft()

    def get_if_top(self, predicate, **kwargs):
        with self._condition:
            if self._queue and predicate(self._queue[0], **kwargs):
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
        
    def close(self):
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
            
    def join_thread(self):
        with self._condition:
            while not self.empty():
                self._condition.wait(timeout=0.1)