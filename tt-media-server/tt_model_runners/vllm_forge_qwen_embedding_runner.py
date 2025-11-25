# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import vllm
from config.settings import SupportedModels
from domain.text_embedding_request import TextEmbeddingRequest
from transformers import AutoTokenizer
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time


class VLLMForgeEmbeddingQwenRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time("Model warmup")
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            SupportedModels.QWEN_3_EMBEDDING_4B.value
        )

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
                "batch_size": self.settings.max_batch_size,
            },
            "hf_overrides": {
                "is_matryoshka": True,
            },
        }
        self.llm = vllm.LLM(**llm_args)

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def set_device(self):
        return {}
    
    def is_request_batchable(self, request, batch=None):
        if not batch:
            return True
        # all requests must have the same dimensions to be batched
        # and number of tokens must be within limits
        num_tokens=0
        for existing_request in batch:
            num_tokens += len(self.tokenizer.encode(existing_request.input))
            if num_tokens > self.settings.max_num_batched_tokens: 
                return False
            if request.dimensions != existing_request.dimensions:
                return False
        return True

    @log_execution_time("Qwen text embedding inference")
    def run_inference(self, requests: list[TextEmbeddingRequest]):
        input = [request.input for request in requests]

        """num_tokens = len(self.tokenizer.encode(" ".join(input)))
        if num_tokens > self.settings.max_model_length:	        if num_tokens > self.settings.max_model_length:
            raise ValueError(	            raise ValueError(
                f"Input text exceeds maximum model length of {self.settings.max_model_length} tokens. Got {num_tokens} tokens."	                f"Input text exceeds maximum model length of {self.settings.max_model_length} tokens. Got {num_tokens} tokens."
            )	            )"""

        self.logger.debug(f"Device {self.device_id}: Running inference")

        pooling_params = None
        if requests[0].dimensions is not None:
            pooling_params = vllm.PoolingParams(dimensions=requests[0].dimensions)

        output_embedding = self.llm.embed(input, pooling_params=pooling_params)
        embeddings = []
        for output in output_embedding:
            embeddings.append(output.outputs.embedding)

        self.logger.debug(f"Device {self.device_id}: Inference output: {embeddings}")
        self.logger.debug(f"Device {self.device_id}: Inference completed")

        return embeddings