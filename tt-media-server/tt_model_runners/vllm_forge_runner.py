# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import asyncio
import os

from domain.text_completion_request import TextCompletionRequest
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.helpers import log_execution_time
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


class VLLMForgeRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.sampling_params = SamplingParams(
            temperature=0.8, top_p=0.95, max_tokens=32
        )

    def set_device(self):
        return {}

    @log_execution_time(
        "VLLM Forge model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM Forge model...")
        prompts = [
            "Hello, it's me",
        ]
        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=65536,
            max_num_seqs=32,
            enable_chunked_prefill=False,
            block_size=64,
            max_num_batched_tokens=65536,
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)

        self.logger.info(f"Device {self.device_id}: Starting model warmup")
        self.llm.generate(prompts[0], self.sampling_params, -1)
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    @log_execution_time(
        "Run VLLM Forge inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run_inference(self, requests: list[TextCompletionRequest]):
        """Synchronous wrapper for async inference"""
        return asyncio.run(self._run_inference_async(requests))

    async def _run_inference_async(self, requests: list[TextCompletionRequest]):
        try:
            self.logger.debug(f"Device {self.device_id}: Running inference")
            request = requests[0]
            return self._generate_streaming(request.text, request._task_id)

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    async def _generate_streaming(self, prompt, task_id):
        try:
            results_generator = self.llm.generate(prompt, self.sampling_params, 1)
            streaming_chunks = []
            async for request_output in results_generator:
                for output in request_output.outputs:
                    print(output.text, end="", flush=True)
                    cleaned_text = TextUtils.extract_text(output.text)

                    # Yield non-empty chunks
                    if not cleaned_text:
                        continue
                    streaming_chunks.append(cleaned_text)
                    yield {
                        "type": "streaming_chunk",
                        "chunk": cleaned_text,
                        "task_id": task_id,
                    }

            yield {
                "type": "final_result",
                "result": TextUtils.concatenate_chunks(streaming_chunks),
                "task_id": task_id,
            }
        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e
