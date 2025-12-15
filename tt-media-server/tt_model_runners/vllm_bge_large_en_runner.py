# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_embedding_request import TextEmbeddingRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from vllm import LLM


class VLLMBGELargeENRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time("Model warmup")
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
        self.llm = LLM(
            model="BAAI/bge-large-en-v1.5",
            max_model_len=384,
            max_num_seqs=8,
            max_num_batched_tokens=3072,
        )

        self.llm.embed(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def run_inference(self, requests: list[TextEmbeddingRequest]):
        self.logger.debug(
            f"VLLMBGELargeENRunner: Running inference for {len(requests)} requests"
        )
        prompts = [req.input for req in requests]
        result = self.llm.embed(prompts)
        self.logger.debug(f"VLLMBGELargeENRunner: result length: {len(result)}")
        return [output.outputs.embedding for output in result]
