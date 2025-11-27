# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from multiprocessing import Lock
from multiprocessing.managers import SyncManager
from multiprocessing import get_context
from multiprocessing.queues import Queue

from queue import Empty

class TaskQueue(Queue):
    def __init__(self, max_size=0, *, ctx=None, lock=None):
        super().__init__(maxsize=max_size, ctx=ctx)
        self._lock = lock
        self._leftover_item = None

    def get(self, block=True, timeout=None):
        with self._lock:
            if self._leftover_item is not None:
                item = self._leftover_item
                self._leftover_item = None
                return item
        return super().get(block, timeout)

    def get_nowait(self):
        with self._lock:
            if self._leftover_item is not None:
                item = self._leftover_item
                self._leftover_item = None
                return item
        return super().get_nowait()

    def get_if_top(self, predicate, timeout=None, **kwargs):
        with self._lock:
            item = self._leftover_item
            self._leftover_item = None
                
        if item is None:
            item = super().get(block=True, timeout=timeout)
                
        if predicate(item, **kwargs):
            return item
        
        with self._lock:
            self._leftover_item = item
            
        raise Empty
        

class TaskQueueManager(SyncManager):
    pass

def make_task_queue(max_size=0):
    lock = Lock()
    return TaskQueue(max_size=max_size, ctx=get_context(), lock=lock)

TaskQueueManager.register('TaskQueue', callable=make_task_queue)

def make_managed_task_queue(manager, max_size=0):
    return manager.TaskQueue(max_size=max_size)
