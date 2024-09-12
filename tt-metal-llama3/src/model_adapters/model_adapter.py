from abc import ABC, abstractmethod
from typing import List


class ModelAdapterABC(ABC):
    @abstractmethod
    def __init__(self, device, inference_config):
        pass

    @abstractmethod
    def tokenize_prompt(
        self,
        prompt: str,
        rag_context: str = None,
        add_special_tokens: bool = True,
        **kwargs
    ) -> List[int]:
        pass

    @abstractmethod
    def initialize_inputs(self):
        pass

    @abstractmethod
    def prefill(self):
        pass

    @abstractmethod
    def decode(self):
        pass
