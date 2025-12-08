# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import os
import traceback

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.helpers import log_execution_time
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


class VLLMForgeRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    def set_device(self):
        return {}

    @log_execution_time(
        "VLLM Forge model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM Forge model...")
        prompt = "Hello, it's me"
        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=65536,
            max_num_seqs=32,
            enable_chunked_prefill=False,
            block_size=64,
            max_num_batched_tokens=65536,
            seed=9472,
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.logger.info(f"Device {self.device_id}: Starting model warmup")
        warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        warmup_generator = self.llm_engine.generate(
            prompt, warmup_sampling_params, "warmup_task_id"
        )
        async for _ in warmup_generator:
            pass
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    async def run_inference(self, requests: list[CompletionRequest]):
        """**SIMPLIFIED**: Just use generate() - vLLM batches automatically"""
        request = requests[0]
        return self._generate_streaming(request)

    async def _generate_streaming(self, request: CompletionRequest):
        """Simple streaming using generate() - vLLM handles batching internally"""
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.8,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 16,
        )

        previous_text = ""
        chunk_count = 0

        try:
            # **KEY**: Just use generate() - it automatically batches with other concurrent calls
            async for request_output in self.llm_engine.generate(
                request.prompt, sampling_params, request._task_id
            ):
                for output in request_output.outputs:
                    current_text = output.text
                    delta_text = current_text[len(previous_text) :]
                    previous_text = current_text

                    cleaned_delta = TextUtils.extract_text(delta_text)

                    if not cleaned_delta:
                        continue

                    chunk_count += 1
                    yield {
                        "type": "streaming_chunk",
                        "chunk": CompletionStreamChunk(text=cleaned_delta),
                        "task_id": request._task_id,
                    }

            yield {
                "type": "final_result",
                "result": CompletionStreamChunk(
                    text=TextUtils.extract_text(previous_text)
                ),
                "task_id": request._task_id,
                "return": False,
            }

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Error: {type(e).__name__}: {e}"
            )
            self.logger.error(f"Device {self.device_id}: {traceback.format_exc()}")
            raise
