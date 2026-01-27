# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

try:
    import vllm
except ImportError:
    vllm = None

from config.constants import SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from transformers import AutoTokenizer
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time


class VLLMBGEImplementation(BaseDeviceRunner):
    """VLLM-based implementation of BGE Large EN embedding model runner."""

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        if vllm is None:
            raise ImportError(
                "vllm module is not installed. Install it with: pip install vllm"
            )
        self.num_tokens_in_batch = 0
        self.dimensions_in_batch = None

    @log_execution_time("BGE Large EN model warmup")
    async def warmup(self) -> bool:
        """Warm up the VLLM model."""
        self.logger.info(f"Device {self.device_id}: Loading BGE Large EN model with VLLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            SupportedModels.BGE_LARGE_EN_V1_5.value
        )

        prompts = ["The capital of France is Paris"]

        llm_args = {
            "model": SupportedModels.BGE_LARGE_EN_V1_5.value,
            "task": "embed",
            "dtype": "bfloat16",
            "disable_sliding_window": True,
            "enable_prefix_caching": False,
            "max_model_len": self.settings.vllm.max_model_length,
            "max_num_batched_tokens": self.settings.vllm.max_num_batched_tokens,
            "max_num_seqs": self.settings.vllm.max_num_seqs,
            "additional_config": {
                "enable_const_eval": False,
                "batch_size": self.settings.max_batch_size,
                "min_context_len": self.settings.vllm.min_context_length,
            },
        }
        self.llm = vllm.LLM(**llm_args)

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: BGE Large EN model warmup completed")

        return True

    def set_device(self):
        """VLLM handles device management internally."""
        return {}

    def is_request_batchable(self, request, batch=None):
        """Check if request can be batched with existing batch."""
        num_tokens = len(self.tokenizer.encode(request.input))

        if num_tokens > self.settings.vllm.max_model_length:
            raise ValueError(
                f"Input text exceeds maximum model length of {self.settings.vllm.max_model_length}. Got {num_tokens} tokens."
            )

        if self.num_tokens_in_batch == 0:
            self.num_tokens_in_batch = num_tokens
            self.dimensions_in_batch = request.dimensions

        # All requests must have the same dimensions to be batched and number of tokens must be within limits
        if (
            self.num_tokens_in_batch + num_tokens
            > self.settings.vllm.max_num_batched_tokens
            or request.dimensions != self.dimensions_in_batch
            or request.model != SupportedModels.BGE_LARGE_EN_V1_5.value
            or (batch is not None and request.model != batch[0].model)
        ):
            return False

        self.num_tokens_in_batch += num_tokens
        return True

    @log_execution_time("BGE Large EN text embedding inference")
    def run(self, requests: list[TextEmbeddingRequest]):
        """Run inference on text embedding requests using VLLM."""
        input_texts = [req.input for req in requests]

        # Validate model if only one request in batch
        if self.num_tokens_in_batch == 0:
            if requests[0].model != SupportedModels.BGE_LARGE_EN_V1_5.value:
                raise ValueError(
                    f"Model {requests[0].model} is not supported by VLLMBGEImplementation"
                )
            self.dimensions_in_batch = requests[0].dimensions
            num_tokens = len(self.tokenizer.encode(requests[0].input))
            if num_tokens > self.settings.vllm.max_model_length:
                raise ValueError(
                    f"Batched input text exceeds maximum number of batched tokens of {self.settings.vllm.max_model_length}. Got {num_tokens} tokens."
                )

        self.logger.debug(f"Device {self.device_id}: Running inference")

        pooling_params = None
        if self.dimensions_in_batch is not None:
            pooling_params = vllm.PoolingParams(dimensions=self.dimensions_in_batch)

        output_embedding = self.llm.embed(input_texts, pooling_params=pooling_params)

        self.num_tokens_in_batch = 0
        self.dimensions_in_batch = None

        return [
            EmbeddingResponse(
                embedding=output.outputs.embedding,
                total_tokens=len(output.prompt_token_ids),
            )
            for output in output_embedding
        ]
