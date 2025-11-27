import unittest
import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Dict, List
import os
import sys


# Add the parent directory to Python path to find model_services
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_services.fast_fifo_queue import FifoQueue
from model_services.task_queue import TaskQueue
from model_services.ultra_fast_queue import UltraFastQueue

@dataclass
class TestObject:
    """Test object to put in queue"""
    id: int
    name: str
    data: Dict[str, Any]


def producer(queue, objects, start_event, stats_queue):
    """Producer process function - moved to module level"""
    start_event.wait()  # Wait for start signal
    start_time = time.perf_counter()
    
    for obj in objects:
        queue.put(obj)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    stats_queue.put(("write", duration))


def consumer(queue, count, start_event, stats_queue):
    """Consumer process function - moved to module level"""
    start_event.wait()  # Wait for start signal
    start_time = time.perf_counter()
    
    objects = []
    for _ in range(count):
        obj = queue.get()
        objects.append(obj)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    stats_queue.put(("read", duration, len(objects)))


class TestQueueManagement(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.queue_size = 1_000_000
        self.test_objects = self._create_test_objects(self.queue_size)
    
    def _create_test_objects(self, count: int) -> List[TestObject]:
        """Create test objects for queue operations"""
        objects = []
        for i in range(count):
            obj = TestObject(
                id=i,
                name=f"test_object_{i}",
                data={
                    "value": i * 2,
                    "description": f"Test data for object {i}",
                    "metadata": {"created": time.time(), "type": "test"}
                }
            )
            objects.append(obj)
        return objects
    
    def test_custom_queue_performance_one_million_objects(self):
        """Test writing and reading one million objects to/from custom task queue"""
        print(f"\n{'='*60}")
        print(f"Testing Custom FifoQueue with {self.queue_size:,} objects")
        print(f"{'='*60}")
        
        # Initialize your custom queue
        queue = FifoQueue(self.queue_size)
        
        # Measure writing time
        print("Starting write operation...")
        write_start_time = time.perf_counter()
        
        for obj in self.test_objects:
            queue.put(obj)
        
        write_end_time = time.perf_counter()
        write_duration = write_end_time - write_start_time
        
        print(f"✅ Write completed: {write_duration:.4f} seconds")
        print(f"   Write rate: {self.queue_size / write_duration:,.0f} objects/second")
        
        # Assert write performance requirement
        self.assertLess(write_duration, 0.9, 
                       f"Write operation took {write_duration:.4f}s, expected < 0.9s")
        
        # Verify queue size
        queue_size = queue.qsize()
        print(f"   Queue size: {queue_size:,} objects")
        
        # Measure reading time
        print("Starting read operation...")
        read_start_time = time.perf_counter()
        
        read_objects = []
        for _ in range(self.queue_size):
            obj = queue.get()
            read_objects.append(obj)
        
        read_end_time = time.perf_counter()
        read_duration = read_end_time - read_start_time
        
        print(f"✅ Read completed: {read_duration:.4f} seconds")
        print(f"   Read rate: {self.queue_size / read_duration:,.0f} objects/second")
        
        # Assert read performance requirement
        self.assertLess(read_duration, 20.0, 
                       f"Read operation took {read_duration:.4f}s, expected < 20.0s")
        
        # Verify data integrity
        self.assertEqual(len(read_objects), self.queue_size)
        self.assertEqual(read_objects[0].id, 0)
        self.assertEqual(read_objects[-1].id, self.queue_size - 1)
        self.assertEqual(read_objects[0].name, "test_object_0")
        self.assertEqual(read_objects[-1].name, f"test_object_{self.queue_size - 1}")
        
        # Performance summary
        total_duration = write_duration + read_duration
        print(f"\n📊 Performance Summary:")
        print(f"   Objects processed: {self.queue_size:,}")
        print(f"   Write time: {write_duration:.4f}s ({self.queue_size / write_duration:,.0f} obj/s)")
        print(f"   Read time: {read_duration:.4f}s ({self.queue_size / read_duration:,.0f} obj/s)")
        print(f"   Total time: {total_duration:.4f}s ({self.queue_size / total_duration:,.0f} obj/s)")
        
        # Additional performance assertions
        self.assertLess(total_duration, 20.9, 
                       f"Total operation took {total_duration:.4f}s, expected < 20.9s")
        
        # Assert minimum throughput requirements
        write_throughput = self.queue_size / write_duration
        read_throughput = self.queue_size / read_duration
        
        self.assertGreater(write_throughput, 1_111_111, 
                          f"Write throughput {write_throughput:,.0f} obj/s, expected > 1,111,111 obj/s")
        self.assertGreater(read_throughput, 50_000, 
                          f"Read throughput {read_throughput:,.0f} obj/s, expected > 50,000 obj/s")
        
        # Memory usage estimation
        import sys
        single_object_size = sys.getsizeof(self.test_objects[0])
        estimated_memory = single_object_size * self.queue_size / (1024 * 1024)  # MB
        print(f"   Estimated memory: {estimated_memory:.2f} MB")
        
        # Verify queue is empty
        self.assertTrue(queue.empty())
        
        # Clean up
        # manager.shutdown()
        
        print(f"✅ All performance assertions passed!")
        print(f"{'='*60}")
    
    def test_multiprocessing_queue_comparison(self):
        """Compare performance with standard multiprocessing.Queue"""
        print(f"\n{'='*60}")
        print(f"Comparing with multiprocessing.Queue")
        print(f"{'='*60}")
        
        # Test with smaller dataset for comparison
        test_size = 100_000
        test_objects = self.test_objects[:test_size]
        
        # Test standard multiprocessing queue
        mp_queue = multiprocessing.Queue()
        
        # Write to multiprocessing queue
        mp_write_start = time.perf_counter()
        for obj in test_objects:
            mp_queue.put(obj)
        mp_write_duration = time.perf_counter() - mp_write_start
        
        # Read from multiprocessing queue
        mp_read_start = time.perf_counter()
        mp_objects = []
        for _ in range(test_size):
            mp_objects.append(mp_queue.get())
        mp_read_duration = time.perf_counter() - mp_read_start
        
        # Test your custom queue
        custom_queue = FifoQueue(test_size)
        
        # Write to custom queue
        custom_write_start = time.perf_counter()
        for obj in test_objects:
            custom_queue.put(obj)
        custom_write_duration = time.perf_counter() - custom_write_start
        
        # Read from custom queue
        custom_read_start = time.perf_counter()
        custom_objects = []
        for _ in range(test_size):
            custom_objects.append(custom_queue.get())
        custom_read_duration = time.perf_counter() - custom_read_start
        
        # Print comparison
        print(f"📊 Performance Comparison ({test_size:,} objects):")
        print(f"   Multiprocessing Queue:")
        print(f"     Write: {mp_write_duration:.4f}s ({test_size / mp_write_duration:,.0f} obj/s)")
        print(f"     Read:  {mp_read_duration:.4f}s ({test_size / mp_read_duration:,.0f} obj/s)")
        print(f"   Custom Queue:")
        print(f"     Write: {custom_write_duration:.4f}s ({test_size / custom_write_duration:,.0f} obj/s)")
        print(f"     Read:  {custom_read_duration:.4f}s ({test_size / custom_read_duration:,.0f} obj/s)")
        
        # Calculate performance ratios
        write_ratio = mp_write_duration / custom_write_duration
        read_ratio = mp_read_duration / custom_read_duration
        
        print(f"   Performance Ratios (higher is better for custom queue):")
        print(f"     Write: {write_ratio:.2f}x {'faster' if write_ratio > 1 else 'slower'}")
        print(f"     Read:  {read_ratio:.2f}x {'faster' if read_ratio > 1 else 'slower'}")
        
        # Verify data integrity
        self.assertEqual(len(mp_objects), test_size)
        self.assertEqual(len(custom_objects), test_size)
        
        print(f"{'='*60}")


if __name__ == '__main__':
    # Set start method for multiprocessing (important for cross-platform compatibility)
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('fork')  # Use 'fork' instead of 'spawn'
        except RuntimeError:
            pass  # Already set
    
    # Run tests with verbose output
    unittest.main(verbosity=2)