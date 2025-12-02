# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import vllm
from domain.text_completion_request import TextCompletionRequest
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.helpers import log_execution_time


class VLLMForgeRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None
        
    def set_device(self):
        return {}

    @log_execution_time("Model warmpup")
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading model...")

        prompts = [
            "Hello, my name is",
        ]
        sampling_params = vllm.SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=32
        )
        llm_args = {
            "model": "facebook/opt-125m",
            "max_num_batched_tokens": 32,
            "max_num_seqs": 1,
            "max_model_len": 32,
        }
        self.llm = vllm.LLM(**llm_args)

        self.llm.generate(prompts, sampling_params)[0].outputs[0].text
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time("SD35 inference")
    def run_inference(self, requests: list[TextCompletionRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        sampling_params = vllm.SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=32
        )
        output = self.llm.generate(requests[0].text, sampling_params)
        self.logger.debug(output)
        output_text = output[0].outputs[0].text
        self.logger.debug(f"Device {self.device_id}: Inference output: {output_text}")
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return [output_text]
