
from config import settings
from config.settings import get_settings
from domain.text_completion_request import TextCompletionRequest
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
from utils.logger import TTLogger
import vllm


class VLLMForgeEmbeddingQwenRunner(BaseDeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.settings = get_settings()
        self.pipeline = None
        self.logger = TTLogger()

    def get_device(self):
        return None

    def close_device(self, device) -> bool:
        return True

    @log_execution_time("Model warmpup")
    async def load_model(self, device)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
        llm_args = {
            "model": settings.SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "task": "embed",
            "dtype": "bfloat16",
            "max_model_len": 64,
            "disable_sliding_window": True,
            "max_num_batched_tokens": 64,
            "max_num_seqs": 2,
        }
        self.llm = vllm.LLM(**llm_args)

        output_embedding = self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time("Qwen text embedding inference")
    def run_inference(self, requests: list[TextEmbeddingRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")

        output_embedding = self.llm.embed(requests[0].input)
        embedding = output_embedding[0].outputs.embedding

        self.logger.debug(f"Device {self.device_id}: Inference output: {embedding}")
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        
        return [embedding]