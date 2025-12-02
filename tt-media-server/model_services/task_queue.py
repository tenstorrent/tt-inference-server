from multiprocessing import get_context
from multiprocessing.queues import Queue
from queue import Empty

class TaskQueue(Queue):
    def __init__(self, max_size=0, batch_enabled: bool=False, *, ctx=None):
        if ctx is None:
            ctx = get_context()
        super().__init__(maxsize=max_size, ctx=ctx)
        
        self.batch_enabled = batch_enabled
        self._leftover_items = Queue(ctx=ctx)

    def __getstate__(self):
        parent_state = super().__getstate__()
        
        return (parent_state, self.batch_enabled, self._leftover_items)

    def __setstate__(self, state):
        parent_state, batch_enabled, leftover_items = state
        super().__setstate__(parent_state)
        
        self.batch_enabled = batch_enabled
        self._leftover_items = leftover_items

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
