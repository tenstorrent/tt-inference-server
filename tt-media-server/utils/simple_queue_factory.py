from config.constants import QueueType
from model_services.queues.tt_faster_fifo_queue import TTFasterFifoQueue
from model_services.queues.tt_batch_fifo_queue import TTBatchFifoQueue
from model_services.queues.tt_queue import TTQueue
from model_services.queues.memory_queue import SharedMemoryChunkQueue
from model_services.queues.tt_queue_interface import TTQueueInterface


def get_queue(
    queue_type: str, size: int = 0, name: str = "queue", create: bool = True
) -> TTQueueInterface:
    if queue_type == QueueType.FasterFifo.value:
        return TTFasterFifoQueue(size)
    elif queue_type == QueueType.TTQueue.value:
        return TTQueue(size)
    elif queue_type == QueueType.MemoryQueue.value:
        return SharedMemoryChunkQueue(capacity=size, name=name, create=create)


def get_task_queue(queue_type: str, size: int) -> TTQueueInterface:
    """Get a queue suitable for task objects (must serialize arbitrary Python objects)."""
    if queue_type == QueueType.FasterFifo.value:
        from config.settings import get_settings
        settings = get_settings()
        # Use batch-aware queue for better worker utilization
        return TTBatchFifoQueue(max_size=size, batch_size=settings.max_batch_size)
    else:
        # SharedMemoryChunkQueue cannot hold arbitrary objects, use standard Queue
        return TTQueue(size)
