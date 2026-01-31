from abc import ABC, abstractmethod
from typing import List, Optional, Any


class TTQueueInterface(ABC):
    """
    Abstract Base Class representing the full interface of
    multiprocessing.Queue as of Python 3.10+
    """

    @abstractmethod
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Remove and return an item from the queue."""
        pass

    @abstractmethod
    def get_nowait(self) -> Any:
        """Equivalent to get(False)."""
        pass

    @abstractmethod
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None):
        """Put an item into the queue."""
        pass

    @abstractmethod
    def put_nowait(self, item: Any):
        """Equivalent to put(obj, False)."""
        pass

    @abstractmethod
    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        pass

    @abstractmethod
    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise (approximate)."""
        pass

    @abstractmethod
    def full(self) -> bool:
        """Return True if the queue is full, False otherwise (approximate)."""
        pass

    @abstractmethod
    def put_many(
        self, items: List[Any], block: bool = True, timeout: Optional[float] = None
    ):
        """Put multiple items into the queue."""
        pass

    @abstractmethod
    def get_many(
        self,
        max_messages_to_get: int = 100,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Get multiple items from the queue."""
        pass

    @abstractmethod
    def peek_next(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Peek at next item for conditional processing."""
        pass

    @abstractmethod
    def peek(self, n: int, timeout: Optional[float] = None) -> List[Any]:
        """Peek at next n items for conditional processing."""
        pass
