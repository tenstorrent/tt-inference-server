# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.constants import SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.bge_large_en.ttnn_implementation import TTNNBGELargeENRunner

try:
    from vllm import LLM
except ImportError:
    LLM = None


class VLLMEmbeddingRunner(BaseDeviceRunner):
    """Base VLLM embedding runner. Subclasses set model, max_model_len, max_num_seqs."""

    async def warmup(self) -> bool:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["HF_MODEL"] = self.model
        self.logger.info(f"Device {self.device_id}: Loading model...")
        prompts = ["The capital of France is Paris"]
        self.llm = LLM(
            model=self.model,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_model_len * self.max_num_seqs,
            use_tqdm_on_load=False,
        )
        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    def run(self, requests: list[TextEmbeddingRequest]):
        self.logger.debug(
            f"{self.model}: Running inference for {len(requests)} requests"
        )
        for req in requests:
            if req.model != self.model:
                raise ValueError(f"Model {req.model} is not supported by {self.model}")
        prompts = [req.input for req in requests]
        result = self.llm.embed(prompts)
        return [
            EmbeddingResponse(
                embedding=output.outputs.embedding,
                total_tokens=len(output.prompt_token_ids),
            )
            for output in result
        ]


class VLLMBGELargeENRunner:
    """
    Facade: use_vllm_bge True -> VLLMBGEImplementation; False -> TTNNBGELargeENRunner.
    """

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        from config.settings import get_settings

        settings = get_settings()
        if settings.use_vllm_bge:
            from tt_model_runners.bge_large_en.vllm_implementation import (
                VLLMBGEImplementation,
            )

            self._impl = VLLMBGEImplementation(device_id, num_torch_threads)
        else:
            self._impl = TTNNBGELargeENRunner(device_id, num_torch_threads)

    @property
    def logger(self):
        return self._impl.logger

    @property
    def settings(self):
        return self._impl.settings

    @property
    def device_id(self):
        return self._impl.device_id

    async def warmup(self) -> bool:
        return await self._impl.warmup()

    def run(self, requests: list[TextEmbeddingRequest]):
        return self._impl.run(requests)

    def set_device(self):
        return self._impl.set_device()

    def is_request_batchable(self, request, batch=None):
        return self._impl.is_request_batchable(request, batch)

    def get_pipeline_device_params(self):
        if hasattr(self._impl, "get_pipeline_device_params"):
            return self._impl.get_pipeline_device_params()
        return {}

    def close_device(self) -> bool:
        if hasattr(self._impl, "close_device"):
            return self._impl.close_device()
        return True


class VLLMQwen3Embedding8BRunner(VLLMEmbeddingRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.max_model_len = 128
        self.max_num_seqs = 1
        self.model = SupportedModels.QWEN_3_EMBEDDING_8B.value
