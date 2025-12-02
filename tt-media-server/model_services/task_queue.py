from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
from queue import Empty

class TaskQueue(Queue):
    def __init__(self, max_size=0, batch_enabled=False, *, ctx=None):
        if ctx is None:
            ctx = get_context()
        self.batch_enabled = batch_enabled
        super().__init__(maxsize=max_size, ctx=ctx)
        self._leftover_items = Queue(ctx=ctx)

    def get(self, block=True, timeout=None):
        if self.batch_enabled:
            try:
                item = self._leftover_items.get_nowait()
                return item
            except Empty:
                return super().get(block=block, timeout=timeout)
        else:
            return super().get(block=block, timeout=timeout)

    def get_nowait(self):
        if self.batch_enabled:
            try:
                item = self._leftover_items.get_nowait()
                return item
            except Empty:
                return super().get_nowait()
        else:
            return super().get_nowait()

    def get_if_top(self, timeout=None):
        if self.batch_enabled:
            try:
                item = self._leftover_items.get_nowait()
                return item
            except Empty:
                item = super().get(block=False, timeout=timeout)
            return item
        else:
            return super().get(block=False, timeout=timeout)
        
    def put_if_not_top(self, item):
        self._leftover_items.put(item)

class TaskQueueManager(SyncManager):
    pass

def make_task_queue(max_size=0):
    return TaskQueue(max_size=max_size, ctx=get_context())

TaskQueueManager.register('TaskQueue', callable=make_task_queue)

def make_managed_task_queue(manager, max_size=0):
    return manager.TaskQueue(max_size=max_size)
