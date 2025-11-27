# SPDX-License-Identifier: Apache-2.0

import time
import pickle
from faster_fifo import Queue as FasterQueue

class FifoQueue:
    def __init__(self, max_size=0):
        """
        Initialize TaskQueue with faster-fifo optimized for write performance.
        """
        # Calculate appropriate byte limit
        if max_size > 0:
            estimated_bytes_per_item = 10240  # 10KB
            byte_limit = max_size * estimated_bytes_per_item
        else:
            byte_limit = 50_000_000_000  # 50GB
        
        byte_limit = max(byte_limit, 5_000_000_000)
        
        print(f"Creating TaskQueue: {byte_limit/(1024**3):.2f} GB")
        
        self._queue = FasterQueue(max_size_bytes=byte_limit)
        self._max_size = max_size
        self._closed = False

    def put(self, item, timeout=None):
        """
        Put item in queue - optimized for speed.
        Pre-serialize outside any locks for maximum performance.
        """
        if self._closed:
            raise Exception("TaskQueue is closed")
        
        try:
            if timeout is None:
                self._queue.put(item)
            else:
                timeout_sec = timeout / 1000 if timeout > 0 else 0.0
                self._queue.put(item, timeout=timeout_sec)
                
        except Exception as e:
            if "timeout" in str(e).lower() or "full" in str(e).lower():
                raise TimeoutError("TaskQueue put timed out")
            raise

    def put_many(self, items):
        """
        **BATCH PUT** - This is MUCH faster for write operations!
        Use this instead of individual puts.
        """
        if self._closed:
            raise Exception("TaskQueue is closed")
        
        try:
            # faster-fifo has put_many which is significantly faster
            self._queue.put_many(items)
        except Exception as e:
            if "timeout" in str(e).lower() or "full" in str(e).lower():
                raise TimeoutError("TaskQueue put_many timed out")
            raise

    def get(self):
        """
        Get item from queue.
        """
        if self._closed:
            raise ValueError("TaskQueue is closed")

        try:
            return self._queue.get()
        except Exception as e:
            if "timeout" in str(e).lower() or "empty" in str(e).lower():
                raise Exception("TaskQueue empty")
            raise

    def get_many(self, count):
        """
        **BATCH GET** - Much faster for read operations!
        """
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        items = []
        try:
            for _ in range(count):
                try:
                    item = self._queue.get(timeout=0.001)
                    items.append(item)
                except:
                    break  # No more items
            return items
        except Exception as e:
            if "timeout" in str(e).lower() or "empty" in str(e).lower():
                return items  # Return partial results
            raise

    def get_nowait(self):
        """Get item without blocking."""
        if self._closed:
            raise ValueError("TaskQueue is closed")

        try:
            return self._queue.get(timeout=0.0)
        except Exception as e:
            raise Exception("TaskQueue empty")

    def peek(self):
        """Peek at next item without removing it."""
        if self._closed:
            raise ValueError("TaskQueue is closed")
        
        try:
            item = self._queue.get(timeout=0.001)
            return item
        except:
            raise Exception("TaskQueue empty")

    def full(self):
        """Check if queue is full."""
        if self._max_size <= 0:
            return False
        try:
            self._queue.put(None, timeout=0.0)
            self._queue.get(timeout=0.0)
            return False
        except:
            return True

    def empty(self):
        """Check if queue is empty."""
        try:
            item = self._queue.get(timeout=0.0)
            self._queue.put(item)
            return False
        except:
            return True

    def qsize(self):
        """Get queue size (approximate)."""
        return 0

    def close(self):
        """Close queue."""
        self._closed = True
        try:
            while True:
                self._queue.get(timeout=0.0)
        except:
            pass

    def join_thread(self):
        """Block until queue is empty."""
        while not self.empty():
            time.sleep(0.001)

    def get_if_top(self, predicate, timeout=None, **kwargs):
        """Get item if predicate matches."""
        start = time.time()
        timeout_sec = timeout / 1000 if timeout is not None else None
        
        while True:
            if self._closed:
                raise ValueError("TaskQueue is closed")
            
            try:
                top_item = self.peek()
                
                if predicate(top_item, **kwargs):
                    return self.get()
                else:
                    if timeout_sec is not None and (time.time() - start) >= timeout_sec:
                        raise Exception("TaskQueue empty")
                    time.sleep(0.001)
                    
            except Exception as e:
                if "empty" in str(e).lower():
                    if timeout_sec is not None and (time.time() - start) >= timeout_sec:
                        raise Exception("TaskQueue empty")
                    time.sleep(0.001)
                else:
                    raise
