# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from vllm import LLM, SamplingParams
from config.settings import SupportedModels
from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time


class VLLMTinyLlamaChatRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    @log_execution_time("Model warmup")
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "The capital of France is Paris",
        ]
        max_model_len = 2048
        max_num_batched_tokens = 2048
        max_num_seqs = 1
        min_context_len = 32
        self.llm = LLM(
            model=SupportedModels.TINYLLAMA_1_1B_CHAT_V1_0.value,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=False,
            additional_config={
                "enable_const_eval": False,
                "min_context_len": min_context_len,
            },
        )

        self.llm.generate(prompts)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    def run(self, requests: list[CompletionRequest]):
        self.logger.debug(
            f"VLLMTinyLlamaChatRunner: Running inference for {len(requests)} requests"
        )
        for req in requests:
            if req.model != SupportedModels.TINYLLAMA_1_1B_CHAT_V1_0.value:
                raise ValueError(
                    f"Model {req.model} is not supported by VLLMTinyLlamaChatRunner"
                )
        sampling_params = SamplingParams(
            max_tokens=requests[0].max_tokens,
            temperature=requests[0].temperature,
            top_p=requests[0].top_p,
            top_k=requests[0].top_k,
        )
        result = self.llm.generate(requests[0].prompt, sampling_params)
        return [
            CompletionStreamChunk(text=output.outputs[0].text)
            for output in result.outputs
        ]
