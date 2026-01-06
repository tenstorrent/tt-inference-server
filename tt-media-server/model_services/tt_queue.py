# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from multiprocessing import get_context
from multiprocessing.queues import Queue
from queue import Empty


class TTQueue(Queue):
    def __init__(self, max_size=0, batch_enabled=False, *, ctx=None):
        if ctx is None:
            ctx = get_context()
        super().__init__(maxsize=max_size, ctx=ctx)

        self.batch_enabled = batch_enabled
        self._leftover_items = Queue(ctx=ctx)

    def __getstate__(self):
        # Preserve custom attributes during pickling
        parent_state = super().__getstate__()
        return (parent_state, self.batch_enabled, self._leftover_items)

    def __setstate__(self, state):
        # Restore custom attributes after unpickling
        parent_state, batch_enabled, leftover_items = state
        super().__setstate__(parent_state)
        self.batch_enabled = batch_enabled
        self._leftover_items = leftover_items

    def get(self, block=True, timeout=None):
        """Get item, checking leftover cache first if batching enabled"""
        if self.batch_enabled:
            try:
                return self._leftover_items.get_nowait()
            except Empty:
                pass
        return super().get(block=block, timeout=timeout)

    def get_nowait(self):
        """Non-blocking get, checking leftover cache first if batching enabled"""
        if self.batch_enabled:
            try:
                return self._leftover_items.get_nowait()
            except Empty:
                pass
        return super().get_nowait()

    def peek_next(self, timeout=None):
        """Peek at next item for conditional processing"""
        if self.batch_enabled:
            try:
                return self._leftover_items.get_nowait()
            except Empty:
                pass
        return super().get(block=False, timeout=timeout)

    def return_item(self, item):
        """Return item to leftover cache for later processing"""
        self._leftover_items.put(item)
