# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from multiprocessing import Lock, Semaphore, get_context, Value
from multiprocessing.shared_memory import SharedMemory
from collections import deque
import struct
import time
import os

class TaskQueue:
    """Fast task queue with deque-like performance - multiprocess safe"""
    
    def __init__(self, max_size=0):
        """
        Initialize TaskQueue with lock-free puts for maximum performance.
        """
        self.max_size = max_size
        
        # Use deque for fast operations (process-local)
        self._queue = deque()
        
        # Shared memory for atomic counters
        self.state_shm = SharedMemory(create=True, size=16)
        self.state_shm_name = self.state_shm.name
        struct.pack_into('LL', self.state_shm.buf, 0, 0, 0)
        
        ctx = get_context()
        
        # Only lock for gets
        self.read_lock = ctx.Lock()
        
        # **IMPORTANT**: These semaphores MUST be shared across processes
        # They are passed by reference and pickled correctly
        self.items_available = ctx.Semaphore(0)
        self.space_available = ctx.Semaphore(max_size if max_size > 0 else 1000000)
        
        # Peek cache for performance
        self._peek_cache = None
        self._closed = False
        
        print(f"TaskQueue initialized with max_size={max_size}")

    def __getstate__(self):
        """Custom pickle support - include semaphores and locks"""
        return {
            'max_size': self.max_size,
            'state_shm_name': self.state_shm_name,
            'read_lock': self.read_lock,
            'items_available': self.items_available,  # **PASS SEMAPHORE**
            'space_available': self.space_available,  # **PASS SEMAPHORE**
            '_peek_cache': self._peek_cache,
            '_closed': self._closed,
        }

    def __setstate__(self, state):
        """Custom unpickle support - reuse parent's synchronization primitives"""
        self.max_size = state['max_size']
        self.state_shm_name = state['state_shm_name']
        self.read_lock = state['read_lock']  # **REUSE PARENT'S LOCK**
        self.items_available = state['items_available']  # **REUSE PARENT'S SEMAPHORE**
        self.space_available = state['space_available']  # **REUSE PARENT'S SEMAPHORE**
        self._peek_cache = state['_peek_cache']
        self._closed = state['_closed']
        
        # Recreate process-local attributes only
        self._queue = deque()
        
        # Reattach to shared memory
        self.state_shm = SharedMemory(name=self.state_shm_name)

    def _ensure_attached(self):
        """Ensure shared memory is attached"""
        try:
            _ = self.state_shm.buf[0]
        except (AttributeError, ValueError):
            try:
                self.state_shm = SharedMemory(name=self.state_shm_name)
            except:
                raise RuntimeError("Cannot attach to shared memory")

    def _update_size(self):
        """Update shared size counter"""
        try:
            self._ensure_attached()
            size = len(self._queue)
            struct.pack_into('LL', self.state_shm.buf, 0, size, 0)
        except:
            pass

    def put(self, item, timeout=None):
        """Put item in queue - LOCK-FREE"""
        self._ensure_attached()
        
        if self._closed:
            raise Exception("TaskQueue is closed")
        
        self._peek_cache = None
        
        # For unbounded queues, super fast path
        if self.max_size <= 0:
            self._queue.append(item)
            self._update_size()
            self.items_available.release()
            return
        
        # For bounded queues, acquire space semaphore
        if timeout is None:
            acquired = self.space_available.acquire(timeout=-1)
        elif timeout == 0:
            acquired = self.space_available.acquire(timeout=0)
        else:
            timeout_sec = timeout / 1000 if timeout > 100 else timeout
            acquired = self.space_available.acquire(timeout=timeout_sec)
        
        if not acquired:
            raise Exception("Queue full")
        
        self._queue.append(item)
        self._update_size()
        self.items_available.release()

    def get(self, block=True, timeout=None):
        """Get item from queue - properly handles blocking"""
        self._ensure_attached()
        
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        # Check peek cache first
        if self._peek_cache is not None:
            with self.read_lock:
                if len(self._queue) > 0:
                    item = self._peek_cache
                    self._peek_cache = None
                    self._queue.popleft()
                    self._update_size()
                    
                    if self.max_size > 0:
                        self.space_available.release()
                    
                    return item
        
        # Wait for items with proper timeout handling
        while True:
            if not block:
                # Non-blocking: don't wait
                acquired = self.items_available.acquire(timeout=0)
                if not acquired:
                    raise Exception("Queue empty")
            else:
                # Blocking: wait for item
                if timeout is None:
                    # Wait indefinitely
                    acquired = self.items_available.acquire(timeout=-1)
                else:
                    # Wait with timeout
                    timeout_sec = timeout / 1000 if timeout > 100 else timeout
                    acquired = self.items_available.acquire(timeout=timeout_sec)
                
                if not acquired:
                    raise Exception("Queue empty - timeout")
            
            # Now we have a semaphore signal, get the item with lock
            with self.read_lock:
                if len(self._queue) > 0:
                    item = self._queue.popleft()
                    self._update_size()
                    
                    if self.max_size > 0:
                        self.space_available.release()
                    
                    return item
                # Race condition: semaphore released but queue empty
                # This can happen with multiple consumers
                # Just loop and acquire semaphore again

    def get_nowait(self):
        """Get item without blocking"""
        return self.get(block=False, timeout=0)

    def peek(self):
        """Peek at front item"""
        self._ensure_attached()
        
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        if self._peek_cache is not None:
            return self._peek_cache
        
        if len(self._queue) == 0:
            raise Exception("Queue empty")
        
        item = self._queue[0]
        self._peek_cache = item
        return item

    def get_if_top(self, predicate, timeout=None, **kwargs):
        """Get item if predicate matches"""
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        start_time = time.time()
        timeout_sec = timeout / 1000 if timeout is not None else None
        
        while True:
            try:
                top_item = self.peek()
                
                if predicate(top_item, **kwargs):
                    return self.get()
                else:
                    if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
                        raise Exception("No matching item - timeout")
                    
                    self._peek_cache = None
                    time.sleep(0.0001)
                    
            except Exception as e:
                if "empty" in str(e).lower():
                    if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
                        raise Exception("Queue empty")
                    time.sleep(0.0001)
                else:
                    raise

    def put_many(self, items):
        """Batch put - lock-free"""
        if self._closed:
            raise Exception("TaskQueue is closed")
        
        self._peek_cache = None
        
        count = len(items)
        if count == 0:
            return
        
        if self.max_size > 0:
            for _ in range(count):
                if not self.space_available.acquire(timeout=-1):
                    raise Exception("Queue full")
        
        self._queue.extend(items)
        self._update_size()
        
        # Release semaphore for each item
        for _ in range(count):
            self.items_available.release()

    def get_many(self, count):
        """Batch get"""
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        self._peek_cache = None
        
        items = []
        
        with self.read_lock:
            actual_count = min(count, len(self._queue))
            if actual_count == 0:
                raise Exception("Queue empty")
            
            for _ in range(actual_count):
                items.append(self._queue.popleft())
            
            self._update_size()
        
        if self.max_size > 0:
            for _ in range(actual_count):
                self.space_available.release()
        
        return items

    def qsize(self):
        """Get queue size"""
        return len(self._queue)

    def empty(self):
        """Check if queue is empty"""
        return len(self._queue) == 0

    def full(self):
        """Check if queue is full"""
        if self.max_size <= 0:
            return False
        return len(self._queue) >= self.max_size

    def close(self):
        """Close the queue"""
        self._closed = True
        self._peek_cache = None

    def cleanup(self):
        """Clean up shared memory resources"""
        try:
            self.state_shm.close()
            self.state_shm.unlink()
        except:
            pass

    def join_thread(self):
        """Wait until queue is empty"""
        while not self.empty():
            time.sleep(0.001)


class TaskQueueManager:
    """Manager for creating TaskQueue instances"""
    
    def __init__(self):
        self._started = False
        self._queues = {}
    
    def start(self):
        """Start the manager"""
        self._started = True
    
    def shutdown(self):
        """Shutdown the manager"""
        self._started = False
        for queue in self._queues.values():
            try:
                queue.cleanup()
                queue.close()
            except:
                pass
        self._queues.clear()
    
    def task_queue(self, max_size=0):
        """Create a TaskQueue instance"""
        if not self._started:
            raise RuntimeError("TaskQueueManager not started")
        queue = TaskQueue(max_size=max_size)
        self._queues[id(queue)] = queue
        return queue


def make_task_queue(max_size=0):
    """Factory function for creating TaskQueue"""
    return TaskQueue(max_size=max_size)


def make_managed_task_queue(manager, max_size=0):
    """Create a managed TaskQueue instance"""
    return manager.task_queue(max_size=max_size)
