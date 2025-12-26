# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import asyncio
import os
import traceback

from config.constants import SupportedModels
from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind


class VLLMForgeRunner(BaseDeviceRunner):
    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)

    @log_execution_time(
        "VLLM Forge model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM Forge model...")
        prompt = "Hello, it's me"
        engine_args = AsyncEngineArgs(
            model=self.settings.model,
            max_model_len=self.settings.max_model_length,
            max_num_batched_tokens=self.settings.max_num_batched_tokens,
            max_num_seqs=self.settings.max_num_seqs,
            enable_chunked_prefill=False,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            additional_config={
                "enable_const_eval": False,
                "min_context_len": self.settings.min_context_length,
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
        "Run VLLM Forge inference",
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
            # Harcode those sampling params
            # SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=231, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None)
            sampling_params = SamplingParams(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                seed=None,
                stop=[],
                stop_token_ids=[],
                bad_words=[],
                include_stop_str_in_output=False,
                ignore_eos=False,
                min_tokens=0,
                logprobs=None,
                prompt_logprobs=None,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                max_tokens=request.max_tokens if request.max_tokens else 65536,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
                truncate_prompt_tokens=None,
                guided_decoding=None,
                extra_args=None,
                output_kind=RequestOutputKind.DELTA,
            )
            if request.stream:
                return self._generate_streaming(request, sampling_params)
            else:
                return await self._generate_non_streaming(request, sampling_params)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    async def _generate_streaming(
        self, request: CompletionRequest, sampling_params: SamplingParams
    ):
        task_id = request._task_id
        chunk_type = "streaming_chunk"
        final_type = "final_result"

        chunks = []
        chunks_append = chunks.append

        strip_eos = TextUtils.strip_eos

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

                yield {
                    "type": chunk_type,
                    "chunk": CompletionStreamChunk(text=chunk_text),
                    "task_id": task_id,
                }

        if chunks:
            final_text = TextUtils.clean_text("".join(chunks))
        else:
            final_text = ""

        yield {
            "type": final_type,
            "result": CompletionStreamChunk(text=final_text),
            "task_id": task_id,
            "return": False,
        }

        self.logger.info(f"Device {self.device_id}: Streaming generation completed")

    async def _generate_non_streaming(
        self, request: CompletionRequest, sampling_params: SamplingParams
    ):
        self.logger.info(f"Device {self.device_id}: Starting non-streaming generation")

        generated_text = []
        async for request_output in self.llm_engine.generate(
            request.prompt, sampling_params, request._task_id
        ):
            if request_output.outputs:
                generated_text.append(request_output.outputs[0].text)

        generated_text = "".join(generated_text)

        self.logger.info(f"Device {self.device_id}: Non-streaming generation completed")

        return [CompletionStreamChunk(text=generated_text)]
