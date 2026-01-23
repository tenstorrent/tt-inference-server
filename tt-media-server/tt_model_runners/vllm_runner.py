# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import asyncio
import os
import traceback

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.sampling_params_builder import build_sampling_params
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


class VLLMRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    @log_execution_time(
        "VLLM model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM model...")
        prompt = "Hello, it's me"
        engine_args = AsyncEngineArgs(
            model=self.settings.vllm.model,
            max_model_len=self.settings.vllm.max_model_length,
            max_num_batched_tokens=self.settings.vllm.max_num_batched_tokens,
            max_num_seqs=self.settings.vllm.max_num_seqs,
            enable_chunked_prefill=False,
            gpu_memory_utilization=self.settings.vllm.gpu_memory_utilization,
            additional_config={
                "enable_const_eval": False,
                "min_context_len": self.settings.vllm.min_context_length,
            },
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.logger.info(f"Device {self.device_id}: Starting model warmup")
        warmup_sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        warmup_generator = self.llm_engine.generate(
            prompt, warmup_sampling_params, "warmup_task_id"
        )
        async for _ in warmup_generator:
            pass  # Just consume the generator for warmup
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    @log_execution_time(
        "Run VLLM inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[CompletionRequest]):
        """Synchronous wrapper for async inference"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_async(requests))

    async def _run_async(self, requests: list[CompletionRequest]):
        try:
            self.logger.debug(f"Device {self.device_id}: Running inference")

            request = requests[0]
            if request.stream:
                return self._generate_streaming(request)
            else:
                return await self._generate_non_streaming(requests)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    async def _generate_streaming(self, request: CompletionRequest):
        """✅ Yields tuples of (task_id, is_final, text) to avoid pickling"""
        task_id = request._task_id

        chunks = []
        chunks_append = chunks.append

        strip_eos = TextUtils.strip_eos
        sampling_params = build_sampling_params(request)

        async for request_output in self.llm_engine.generate(
            request.prompt, sampling_params, task_id
        ):
            outputs = request_output.outputs
            if not outputs:
                continue

            for output in outputs:
                chunk_text = output.text
                if not chunk_text:
                    continue

                if chunk_text.endswith(("</s>", "<|endoftext|>", "<|im_end|>")):
                    chunk_text = strip_eos(chunk_text)
                    if not chunk_text:
                        continue

                chunks_append(chunk_text)

                yield (task_id, 0, chunk_text)

        # do this on purpose to avoid over max decode issues
        yield (task_id, 1, "final_text")

        self.logger.info(f"Device {self.device_id}: Streaming generation completed")

    async def process_request_non_streaming(self, request: CompletionRequest):
        self.logger.info(f"Device {self.device_id}: Starting non-streaming generation")

        sampling_params = build_sampling_params(request)

        generated_text = []
        async for request_output in self.llm_engine.generate(
            request.prompt, sampling_params, request._task_id
        ):
            if request_output.outputs:
                generated_text.append(request_output.outputs[0].text)

        generated_text = "".join(generated_text)

        self.logger.info(f"Device {self.device_id}: Non-streaming generation completed")

        return CompletionStreamChunk(text=generated_text)

    async def _generate_non_streaming(self, requests: list[CompletionRequest]):
        tasks = [self.process_request_non_streaming(request) for request in requests]
        self.logger.info(
            f"Device {self.device_id}: Starting non-streaming batch generation for {len(tasks)} requests"
        )
        results = await asyncio.gather(*tasks)
        self.logger.info(
            f"Device {self.device_id}: Non-streaming batch generation completed for {len(tasks)} requests"
        )
        return results
