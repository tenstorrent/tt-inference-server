from config.constants import QueueType
from model_services.tt_faster_fifo_queue import TTFasterFifoQueue
from model_services.tt_queue import TTQueue
from model_services.memory_queue import SharedMemoryChunkQueue
from model_services.tt_queue_interface import TTQueueInterface


def get_queue(queue_type: str, size: int = 0, name: str = "queue", create: bool = True) -> TTQueueInterface:
    if queue_type == QueueType.FasterFifo.value:
        return TTFasterFifoQueue(size)
    elif queue_type == QueueType.TTQueue.value:
        return TTQueue(size)
    elif queue_type == QueueType.MemoryQueue.value:
        return SharedMemoryChunkQueue(capacity=size, name=name, create=create)
