from enum import Enum

AVAILABLE_DATASETS = [
    "sst2",
]

AVAILABLE_MODELS = [
    "google/gemma-1.1-2b-it",
    "google/gemma-3-1b-it",
    "microsoft/phi-1",
    "meta-llama/Llama-3.2-1B",
]

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"

class JobType(str, Enum):
    LORA = "lora"