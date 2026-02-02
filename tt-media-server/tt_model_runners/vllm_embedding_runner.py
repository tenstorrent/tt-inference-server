# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.constants import SupportedModels
from domain.embedding_response import EmbeddingResponse
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from vllm import LLM


class VLLMEmbeddingRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        # Disable vLLM multiprocessing to ensure full batch utilization.
        # When enabled, the engine core runs in a separate process (ZMQ IPC),
        # causing non-deterministic scheduling that splits batches inefficiently
        # (e.g., batch of 8 becomes 7+1, doubling forward passes).
        # See: https://github.com/tenstorrent/tt-inference-server/issues/1453
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["HF_MODEL"] = self.model
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
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


class VLLMBGELargeENRunner(VLLMEmbeddingRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.max_model_len = 384
        self.max_num_seqs = 8 * self.settings.device_mesh_shape[0]
        self.model = SupportedModels.BGE_LARGE_EN_V1_5.value


class VLLMQwen3Embedding8BRunner(VLLMEmbeddingRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.max_model_len = self.settings.vllm.max_model_length
        self.max_num_seqs = self.settings.vllm.max_num_seqs
        self.model = SupportedModels.QWEN_3_EMBEDDING_8B.value
