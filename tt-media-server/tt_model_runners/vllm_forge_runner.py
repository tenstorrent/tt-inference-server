# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from domain.text_completion_request import TextCompletionRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.helpers import log_execution_time
import vllm


class VLLMForgeRunner(BaseDeviceRunner):

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.pipeline = None

    def set_device(self):
        return True

    @log_execution_time("Model warmpup")
    async def load_model(self)->bool:
        self.logger.info(f"{VLLMForgeRunner.__name__} Device {self.device_id}: Loading model...")

        prompts = [
            "Hello, my name is",
        ]
        sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
        llm_args = {
            "model": "/home/dmadic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/",
            "max_model_len": 2048,           # Reasonable context length
            "max_num_seqs": 1,               # Keep if you want 1 request at a time
            "max_num_batched_tokens": 2048,  # Should match or exceed max_model_len
            "block_size": 64,                # Smaller block size (8, 16, or 32)
            "gpu_memory_utilization": 0.8,   # Add this
        }
        self.llm = vllm.LLM(**llm_args)

        self.llm.generate(prompts, sampling_params)[0].outputs[0].text
        self.logger.info(f"Device {self.device_id}: Model warmup completed")

        return True

    @log_execution_time("SD35 inference")
    def run_inference(self, requests: list[TextCompletionRequest]):
        self.logger.debug(f"Device {self.device_id}: Running inference")
        sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
        output = self.llm.generate(requests[0].text, sampling_params)
        self.logger.debug(output)
        output_text = output[0].outputs[0].text
        self.logger.debug(f"Device {self.device_id}: Inference output: {output_text}")
        self.logger.debug(f"Device {self.device_id}: Inference completed")
        return [output_text]