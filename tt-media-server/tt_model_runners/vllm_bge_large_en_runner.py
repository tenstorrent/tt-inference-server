# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.constants import SupportedModels
from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from tt_model_runners.embedding_response import EmbeddingResponse
from utils.decorators import log_execution_time
from vllm import LLM


class VLLMBGELargeENRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    @log_execution_time("Model warmup")
    async def load_model(self) -> bool:
        # Disable vLLM multiprocessing to ensure full batch utilization.
        # When enabled, the engine core runs in a separate process (ZMQ IPC),
        # causing non-deterministic scheduling that splits batches inefficiently
        # (e.g., batch of 8 becomes 7+1, doubling forward passes).
        # See: https://github.com/tenstorrent/tt-inference-server/issues/1453
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
        max_model_len = 384
        max_num_seqs = 8
        self.llm = LLM(
            model=SupportedModels.BGE_LARGE_EN_V1_5.value,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_model_len * max_num_seqs,
        )

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def run_inference(self, requests: list[TextEmbeddingRequest]):
        self.logger.debug(
            f"VLLMBGELargeENRunner: Running inference for {len(requests)} requests"
        )
        for req in requests:
            if req.model != SupportedModels.BGE_LARGE_EN_V1_5.value:
                raise ValueError(
                    f"Model {req.model} is not supported by VLLMBGELargeENRunner"
                )
        prompts = [req.input for req in requests]
        result = self.llm.embed(prompts)
        return [
            EmbeddingResponse(
                embedding=output.outputs.embedding,
                total_tokens=len(output.prompt_token_ids),
            )
            for output in result
        ]
