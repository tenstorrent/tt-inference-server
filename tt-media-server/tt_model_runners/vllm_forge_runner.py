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
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind
from kv_cache.kv_cache_storage import KVCache, KVCacheMetadata
from typing import Optional


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

    def _create_sampling_params(self, request: CompletionRequest) -> SamplingParams:
        """
        Create sampling parameters from request

        Centralized method for creating sampling params - used by all inference modes.
        """
        return SamplingParams(
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

    @log_execution_time(
        "Run VLLM Forge inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[CompletionRequest]):
        """Synchronous wrapper for async inference"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_async(requests))

    async def _run_async(self, requests: list[CompletionRequest], mode: str = "full"):
        """
        Run inference (full, prefill-only, or decode-only)

        Args:
            requests: List of completion requests
            mode: Execution mode - "full" (default), "prefill", or "decode"
        """
        try:
            self.logger.debug(f"Device {self.device_id}: Running inference in {mode} mode")

            request = requests[0]

            # Create sampling params (centralized in runner)
            sampling_params = self._create_sampling_params(request)

            if mode == "prefill":
                # Prefill-only mode - extract KV cache
                return await self.run_prefill_only(request, mode=mode)
            elif mode == "decode":
                # Decode mode requires KV cache - should use load_kv_cache_and_decode instead
                raise ValueError("Decode mode requires KV cache. Use load_kv_cache_and_decode() instead.")
            else:
                # Full mode - normal inference
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

    async def run_prefill_only(
        self, request: CompletionRequest, mode: str = "prefill"
    ) -> Optional[KVCache]:
        """
        Run prefill phase only and extract KV cache

        This method runs the prefill phase and stops before decode,
        then extracts the KV cache for transfer to decode worker.

        Args:
            request: Completion request
            mode: Execution mode ("prefill" or "decode")

        Returns:
            KVCache object if successful, None otherwise
        """
        try:
            self.logger.info(
                f"Device {self.device_id}: Running prefill-only for task {request._task_id}"
            )

            # Create sampling params using centralized method
            sampling_params = self._create_sampling_params(request)

            # TODO: Implement actual prefill-only mode in vLLM
            # This requires modifying vLLM engine to support stopping after prefill
            # For now, this is a placeholder that would need integration with vLLM internals

            # The actual implementation would:
            # 1. Run prefill phase (process prompt tokens)
            # 2. Stop before decode phase
            # 3. Extract KV cache from engine's internal state
            # 4. Return KVCache object

            # Placeholder: Access engine's scheduler to get sequence metadata
            # scheduler = self.llm_engine.scheduler
            # seq_group = scheduler.get_seq_group(request._task_id)
            # if seq_group:
            #     # Extract KV cache from sequence group
            #     kv_cache = self._extract_kv_cache_from_seq_group(seq_group)
            #     return kv_cache

            self.logger.warning(
                f"Device {self.device_id}: Prefill-only mode not yet fully implemented. "
                "This requires vLLM engine modifications."
            )
            return None

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Prefill-only failed: {type(e).__name__}: {e}"
            )
            return None

    def _extract_kv_cache_from_engine(self, task_id: str) -> Optional[KVCache]:
        """
        Extract KV cache from vLLM engine's internal state

        This is a placeholder method that would need to access vLLM's internal
        KV cache storage. The actual implementation depends on vLLM's internal API.

        Args:
            task_id: Task ID to extract KV cache for

        Returns:
            KVCache object if successful, None otherwise
        """
        # TODO: Implement actual KV cache extraction from vLLM engine
        # This would require:
        # 1. Access to engine's cache engine (cache_engine)
        # 2. Access to sequence metadata to get KV cache blocks
        # 3. Conversion from vLLM's KV cache format to our KVCache format

        # Placeholder implementation:
        # cache_engine = self.llm_engine.cache_engine
        # if cache_engine:
        #     # Get KV cache blocks for this sequence
        #     # Convert to our format
        #     pass

        return None

    async def load_kv_cache_and_decode(
        self,
        request: CompletionRequest,
        kv_cache: KVCache,
        mode: str = "decode",
    ):
        """
        Load KV cache into engine and continue decode phase

        This method loads a transferred KV cache into the engine and
        continues the decode phase from where prefill left off.

        Args:
            request: Completion request
            kv_cache: KV cache to load
            mode: Execution mode ("prefill" or "decode")

        Yields:
            Decode chunks as they are generated
        """
        try:
            self.logger.info(
                f"Device {self.device_id}: Loading KV cache for task {request._task_id}"
            )

            # Create sampling params using centralized method
            sampling_params = self._create_sampling_params(request)

            # TODO: Implement actual KV cache loading into vLLM engine
            # This requires:
            # 1. Loading KV cache into engine's cache engine
            # 2. Setting up sequence metadata to point to loaded cache
            # 3. Continuing decode from the loaded cache state

            # Placeholder implementation:
            # cache_engine = self.llm_engine.cache_engine
            # if cache_engine:
            #     # Load KV cache blocks
            #     # Set up sequence to use loaded cache
            #     pass

            # Continue decode
            task_id = request._task_id
            async for request_output in self.llm_engine.generate(
                "",  # Empty prompt since we're continuing from KV cache
                sampling_params,
                task_id,
            ):
                outputs = request_output.outputs
                if not outputs:
                    continue

                for output in outputs:
                    chunk_text = output.text
                    if not chunk_text:
                        continue

                    yield {
                        "type": "streaming_chunk",
                        "chunk": CompletionStreamChunk(text=chunk_text),
                        "task_id": task_id,
                    }

            self.logger.info(
                f"Device {self.device_id}: Decode with loaded KV cache completed"
            )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Load KV cache and decode failed: "
                f"{type(e).__name__}: {e}"
            )
            raise RuntimeError(f"Load KV cache and decode failed: {str(e)}") from e
