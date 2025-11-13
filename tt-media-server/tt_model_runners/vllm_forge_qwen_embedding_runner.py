# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from config.settings import SupportedModels
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from transformers import AutoTokenizer
from utils.helpers import log_execution_time
import vllm


class VLLMForgeEmbeddingQwenRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time("Model warmup")
    async def load_model(self, device)->bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(SupportedModels.QWEN_3_EMBEDDING_4B.value)

        prompts = [
            "The capital of France is Paris",
        ]
        llm_args = {
            "model": SupportedModels.QWEN_3_EMBEDDING_4B.value,
            "task": "embed",
            "dtype": "bfloat16",
            "disable_sliding_window": True,
            "enable_prefix_caching": False,
            "max_model_len": self.settings.max_model_length,
            "max_num_batched_tokens": self.settings.max_num_batched_tokens,
            "max_num_seqs": self.settings.max_num_seqs,
            "additional_config": {
                "enable_const_eval": False,
            },
            "hf_overrides": {
                "is_matryoshka": True,
            },
        }
        self.llm = vllm.LLM(**llm_args)

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time("Qwen text embedding inference")
    def run_inference(self, requests: list[TextEmbeddingRequest]):
        request = requests[0]

        num_tokens = len(self.tokenizer.encode(request.input))
        if num_tokens > self.settings.max_model_length:
            raise ValueError(f"Input text exceeds maximum model length of {self.settings.max_model_length} tokens. Got {num_tokens} tokens.")

        self.logger.debug(f"Device {self.device_id}: Running inference")

        pooling_params = None
        if request.dimensions is not None:
            pooling_params = vllm.PoolingParams(dimensions=request.dimensions)

        output_embedding = self.llm.embed(request.input, pooling_params=pooling_params)
        embedding = output_embedding[0].outputs.embedding

        self.logger.debug(f"Device {self.device_id}: Inference output: {embedding}")
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        
        return [embedding]