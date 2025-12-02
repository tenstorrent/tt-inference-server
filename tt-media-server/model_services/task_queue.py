# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
from queue import Empty


class TaskQueue(Queue):
    def __init__(self, max_size=0, *, ctx=None):
        if ctx is None:
            ctx = get_context()
        super().__init__(maxsize=max_size, ctx=ctx)
        self._leftover_items = Queue(ctx=ctx)

    def get(self, block=True, timeout=None):
        try:
            item = self._leftover_items.get_nowait()
            return item
        except Empty:
            return super().get(block=block, timeout=timeout)

    def get_nowait(self):
        try:
            item = self._leftover_items.get_nowait()
            return item
        except Empty:
            return super().get_nowait()

    def get_if_top(self, timeout=None):
        try:
            item = self._leftover_items.get_nowait()
            return item
        except Empty:
            item = super().get(block=False, timeout=timeout)
        return item
        
    def put_if_not_top(self, item):
        self._leftover_items.put(item)


class TaskQueueManager(SyncManager):
    pass


TaskQueueManager.register("TaskQueue", TaskQueue)


def make_managed_task_queue(manager, max_size=0):
    return manager.TaskQueue(max_size=max_size)
