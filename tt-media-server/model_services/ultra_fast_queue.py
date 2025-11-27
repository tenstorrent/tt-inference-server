import pickle
import struct
from multiprocessing import shared_memory, Value, Lock
import time
import threading

class UltraFastQueue:
    """Optimized shared memory queue with minimal lock overhead"""
    
    def __init__(self, max_items=1000000, avg_item_size=2048):
        self.max_items = max_items
        self.avg_item_size = avg_item_size
        
        # Realistic buffer size (not 2x padding)
        self.buffer_size = max_items * avg_item_size
        
        # Single shared memory buffer for data
        self.data_buffer = shared_memory.SharedMemory(
            create=True, 
            size=self.buffer_size,
            name=f"fast_queue_data_{id(self)}"
        )
        
        # **FIXED**: Add extra bytes to prevent off-by-one errors
        # Need 8 bytes per slot (4 for size + 4 for offset) + padding
        meta_buffer_size = (max_items * 8) + 16  # Add 16 bytes padding
        self.meta_buffer = shared_memory.SharedMemory(
            create=True,
            size=meta_buffer_size,
            name=f"fast_queue_meta_{id(self)}"
        )
        
        # Lock-free atomic counters
        self.head = Value('L', 0)          # Read position (4 bytes)
        self.tail = Value('L', 0)          # Write position (4 bytes) 
        self.write_pos = Value('L', 0)     # Current write offset in buffer
        
        # **SINGLE LOCK** for both read and write operations
        self.operation_lock = Lock()
        
        # Peek cache (thread-local to avoid locks where possible)
        self._peek_cache = None
        self._peek_slot = -1
        
        print(f"UltraFastQueue: Data={self.buffer_size/(1024**2):.1f} MB, Meta={meta_buffer_size/(1024**2):.1f} MB")

    def put(self, item):
        """Fast put with single lock"""
        # Pre-serialize outside the lock
        data = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        size = len(data)
        
        # Quick size check
        if size > self.avg_item_size * 4:
            raise ValueError(f"Item too large: {size} > {self.avg_item_size * 4}")
        
        with self.operation_lock:
            # Clear peek cache when adding items
            self._peek_cache = None
            self._peek_slot = -1
            
            # Check if queue is full
            if self.tail.value - self.head.value >= self.max_items:
                raise Exception("Queue full")
            
            # Check buffer space
            if self.write_pos.value + size > self.buffer_size:
                # Simple overflow handling: reject item
                raise Exception(f"Buffer full - item size {size}, remaining space {self.buffer_size - self.write_pos.value}")
            
            # **FIXED**: Get write slot - use absolute tail value, not modulo
            # The modulo should only be used for the metadata offset calculation
            slot_index = self.tail.value % self.max_items
            
            # **FIXED**: More robust bounds check for metadata buffer
            meta_offset = slot_index * 8
            if meta_offset >= len(self.meta_buffer.buf) - 8:  # Ensure we have 8 bytes available
                raise Exception(f"Metadata buffer overflow: offset={meta_offset}, available={len(self.meta_buffer.buf) - 8}")
            
            # Write data to buffer
            current_pos = self.write_pos.value
            if current_pos + size > len(self.data_buffer.buf):
                raise Exception(f"Data buffer overflow: pos={current_pos}, size={size}, buffer_size={len(self.data_buffer.buf)}")
            
            self.data_buffer.buf[current_pos:current_pos + size] = data
            
            # Write metadata (size and offset)
            struct.pack_into('LL', self.meta_buffer.buf, meta_offset, size, current_pos)
            
            # Update pointers atomically (inside lock)
            self.write_pos.value += size
            self.tail.value += 1

    def get(self):
        """Fast get with single lock"""
        with self.operation_lock:
            # Check peek cache first
            if self._peek_cache is not None and self._peek_slot == self.head.value:
                item = self._peek_cache
                self._peek_cache = None
                self._peek_slot = -1
                self.head.value += 1
                return item
            
            # Check if empty
            if self.head.value >= self.tail.value:
                raise Exception("Queue empty")
            
            # **FIXED**: Get current slot with proper bounds checking
            slot_index = self.head.value % self.max_items
            
            # **FIXED**: More robust bounds check for metadata buffer
            meta_offset = slot_index * 8
            if meta_offset >= len(self.meta_buffer.buf) - 8:
                raise Exception(f"Metadata buffer underflow: offset={meta_offset}, available={len(self.meta_buffer.buf) - 8}")
            
            # Read metadata
            size, offset = struct.unpack_from('LL', self.meta_buffer.buf, meta_offset)
            
            # **FIXED**: Bounds check for data buffer
            if offset >= len(self.data_buffer.buf) or offset + size > len(self.data_buffer.buf):
                raise Exception(f"Data buffer underflow: offset={offset}, size={size}, buffer_size={len(self.data_buffer.buf)}")
            
            # Read and deserialize data
            data = bytes(self.data_buffer.buf[offset:offset + size])
            item = pickle.loads(data)
            
            # Update head
            self.head.value += 1
            
            return item

    def peek(self):
        """Peek without removing - cached for performance"""
        with self.operation_lock:
            # Return cached peek if valid
            if self._peek_cache is not None and self._peek_slot == self.head.value:
                return self._peek_cache
            
            # Check if empty
            if self.head.value >= self.tail.value:
                raise Exception("Queue empty")
            
            # Get current slot
            slot_index = self.head.value % self.max_items
            
            # Read metadata with bounds check
            meta_offset = slot_index * 8
            if meta_offset >= len(self.meta_buffer.buf) - 8:
                raise Exception(f"Peek metadata buffer underflow: offset={meta_offset}")
                
            size, offset = struct.unpack_from('LL', self.meta_buffer.buf, meta_offset)
            
            # Read and deserialize data with bounds check
            if offset >= len(self.data_buffer.buf) or offset + size > len(self.data_buffer.buf):
                raise Exception(f"Peek data buffer underflow: offset={offset}, size={size}")
                
            data = bytes(self.data_buffer.buf[offset:offset + size])
            item = pickle.loads(data)
            
            # Cache the result
            self._peek_cache = item
            self._peek_slot = self.head.value
            
            return item

    def get_if_top(self, predicate, timeout=None, **kwargs):
        """Get item if predicate matches top item"""
        start_time = time.time()
        timeout_sec = timeout / 1000 if timeout is not None else None
        
        while True:
            try:
                # Peek at top item (this will cache it)
                top_item = self.peek()
                
                # Test predicate
                if predicate(top_item, **kwargs):
                    # Get the item (will use cached version)
                    return self.get()
                else:
                    # Predicate failed - check timeout
                    if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
                        raise Exception("Timeout waiting for matching item")
                    
                    # Brief sleep to avoid busy waiting
                    time.sleep(0.0001)  # 0.1ms
                    
            except Exception as e:
                if "empty" in str(e).lower():
                    if timeout_sec is not None and (time.time() - start_time) >= timeout_sec:
                        raise Exception("Queue empty")
                    time.sleep(0.0001)
                else:
                    raise

    def get_if_top_nowait(self, predicate, **kwargs):
        """Non-blocking get_if_top"""
        try:
            top_item = self.peek()
            if predicate(top_item, **kwargs):
                return self.get()
            else:
                raise Exception("Top item doesn't match predicate")
        except Exception as e:
            if "empty" in str(e).lower():
                raise Exception("Queue empty")
            raise

    def get_nowait(self):
        """Non-blocking get"""
        try:
            return self.get()
        except Exception as e:
            if "empty" in str(e).lower():
                raise Exception("Queue empty")
            raise

    def put_many(self, items):
        """Batch put for maximum throughput"""
        if not items:
            return
            
        # Pre-serialize all items
        serialized_items = []
        total_size = 0
        
        for item in items:
            data = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
            serialized_items.append((len(data), data))
            total_size += len(data)
        
        with self.operation_lock:
            # Clear peek cache
            self._peek_cache = None
            self._peek_slot = -1
            
            # Check space
            if self.tail.value - self.head.value + len(items) > self.max_items:
                raise Exception("Not enough queue space")
            
            if self.write_pos.value + total_size > self.buffer_size:
                raise Exception("Not enough buffer space")
            
            # Write all items
            current_pos = self.write_pos.value
            
            for i, (size, data) in enumerate(serialized_items):
                slot_index = (self.tail.value + i) % self.max_items
                
                # **FIXED**: Bounds check for metadata
                meta_offset = slot_index * 8
                if meta_offset >= len(self.meta_buffer.buf) - 8:
                    raise Exception(f"Batch put metadata overflow: slot={slot_index}, offset={meta_offset}")
                
                # Write data
                self.data_buffer.buf[current_pos:current_pos + size] = data
                
                # Write metadata
                struct.pack_into('LL', self.meta_buffer.buf, meta_offset, size, current_pos)
                
                current_pos += size
            
            # Update pointers
            self.write_pos.value = current_pos
            self.tail.value += len(items)

    def get_many(self, count):
        """Batch get for maximum throughput"""
        items = []
        
        with self.operation_lock:
            # Clear peek cache
            self._peek_cache = None
            self._peek_slot = -1
            
            available = self.tail.value - self.head.value
            actual_count = min(count, available)
            
            if actual_count == 0:
                raise Exception("Queue empty")
            
            # Read items
            for i in range(actual_count):
                slot_index = (self.head.value + i) % self.max_items
                
                # **FIXED**: Bounds check for metadata
                meta_offset = slot_index * 8
                if meta_offset >= len(self.meta_buffer.buf) - 8:
                    raise Exception(f"Batch get metadata underflow: slot={slot_index}, offset={meta_offset}")
                
                # Read metadata
                size, offset = struct.unpack_from('LL', self.meta_buffer.buf, meta_offset)
                
                # **FIXED**: Bounds check for data
                if offset >= len(self.data_buffer.buf) or offset + size > len(self.data_buffer.buf):
                    raise Exception(f"Batch get data underflow: offset={offset}, size={size}")
                
                # Read and deserialize data
                data = bytes(self.data_buffer.buf[offset:offset + size])
                item = pickle.loads(data)
                items.append(item)
            
            # Update head
            self.head.value += actual_count
        
        return items

    def qsize(self):
        """Get queue size"""
        return self.tail.value - self.head.value

    def empty(self):
        """Check if empty"""
        return self.head.value >= self.tail.value

    def full(self):
        """Check if full"""
        return self.tail.value - self.head.value >= self.max_items

    def cleanup(self):
        """Cleanup shared memory"""
        try:
            self.data_buffer.close()
            self.data_buffer.unlink()
            self.meta_buffer.close()
            self.meta_buffer.unlink()
        except:
            pass