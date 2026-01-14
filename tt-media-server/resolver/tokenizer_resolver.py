from transformers import AutoTokenizer
from config.constants import SupportedModels
from utils.logger import TTLogger
import threading

tokenizer_cache = {}
tokenizer_cache_lock = threading.Lock()

logger = TTLogger()


def get_tokenizer(model: SupportedModels) -> AutoTokenizer:
    with tokenizer_cache_lock:
        if model not in tokenizer_cache:
            try:
                tokenizer_cache[model] = AutoTokenizer.from_pretrained(model.value)
            except Exception as e:
                logger.error(f"Error loading tokenizer for model {model}: {e}")
                raise ValueError(f"Error loading tokenizer for model {model}: {e}")
    return tokenizer_cache[model]
