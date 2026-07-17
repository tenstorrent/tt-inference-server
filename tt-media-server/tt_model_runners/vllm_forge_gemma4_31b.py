# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
import asyncio
import os
import traceback

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionOutput, CompletionResult
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_device_runner import BaseDeviceRunner
from utils.decorators import log_execution_time
from utils.sampling_params_builder import build_sampling_params
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

CHUNK_TYPE = "streaming_chunk"
FINAL_TYPE = "final_result"


class VLLMForgeGemma4_31BRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time(
        "VLLM Forge Gemma-4 31B model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(
            f"Device {self.device_id}: Loading VLLM Forge Gemma-4 31B model..."
        )
        prompt = "Hello, it's me"
        # Tunable per-run via env vars (mirrors vllm_runner.py). Defaults now
        # match the validated b32/high-seq-len/chunked-prefill/b1-prefill
        # config qualified for gemma-4-31b-it TP on qb2-blackhole (P300X2,
        # 4-chip (1,4) mesh): 32768 ctx, bfp8 weights+KV, chunk 1024, opt=1.
        #   ENABLE_TRACE=true     decode-graph replay; the dominant lever
        #                         (greedy 4.8 -> 8.1 tok/s, +71% at small ctx).
        #   CPU_SAMPLING=false    on-device sampling; +13% greedy ON TOP of trace.
        #                         (TTConfig's own default is False; the old
        #                         hardcoded True was the deviation.)
        #   OPTIMIZATION_LEVEL=1  the tt-mlir OpModel worker-grid validation
        #                         abort on Blackhole P300 (device {10,13} vs
        #                         system-desc {10,11}; tt-xla#5204/tt-mlir#8767,
        #                         fixed by tt-mlir#8769) does NOT reproduce on
        #                         the current tt-xla build -- opt=1 qualified
        #                         clean across the full 128-65536 ctx ladder.
        #                         Override to 0 only if a fresh Blackhole crash
        #                         with that signature reappears.
        # Weights stay bfp_bf8: measured FASTER than bf16 (greedy 9.2 vs 8.0),
        # since bf16 doubles per-token weight DRAM traffic.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        cpu_sampling = os.getenv("CPU_SAMPLING", "false").lower() == "true"
        enable_trace = os.getenv("ENABLE_TRACE", "true").lower() == "true"
        # BFP8 KV cache ("bfp_bf8" halves the KV footprint vs bf16; "" -> bf16).
        kv_cache_dtype = os.getenv("KV_CACHE_DTYPE", "bfp_bf8")
        # On-device chunked-SDPA prefill chunk size: without it the full-context
        # prefill SDPA buffer blows up compile time / OOMs DRAM at 32K/64K
        # (tt-xla #4986). Only passed when set.
        prefill_chunk_size = os.getenv("PREFILL_CHUNK_SIZE")
        # "false" -> bf16 matmul dest accumulation (smaller buffers). Only
        # passed when set, so other runs keep the plugin default.
        fp32_dest_acc_en = os.getenv("FP32_DEST_ACC_EN")
        # b1-prefill (tt-xla #5281): compile a small [min_num_seqs, n] prefill
        # graph alongside b32 and route <= prefill_batch_threshold pending
        # prefills to it (lower TTFT) while decode stays at max_num_seqs. Only
        # passed when set.
        min_num_seqs = os.getenv("MIN_NUM_SEQS")
        prefill_batch_threshold = os.getenv("PREFILL_BATCH_THRESHOLD")
        # Debug/testing only: truncate the model to N decoder layers.
        num_hidden_layers = os.getenv("NUM_HIDDEN_LAYERS")
        additional_config = {
            "enable_const_eval": True,
            "min_context_len": self.settings.vllm.min_context_length,
            "enable_tensor_parallel": True,
            "use_2d_mesh": False,
            "experimental_weight_dtype": "bfp_bf8",
            "experimental_kv_cache_dtype": kv_cache_dtype,
            "cpu_sampling": cpu_sampling,
            "optimization_level": optimization_level,
            "enable_trace": enable_trace,
        }
        if prefill_chunk_size:
            additional_config["prefill_chunk_size"] = int(prefill_chunk_size)
        if fp32_dest_acc_en is not None:
            additional_config["fp32_dest_acc_en"] = fp32_dest_acc_en.lower() == "true"
        if min_num_seqs:
            additional_config["min_num_seqs"] = int(min_num_seqs)
        if prefill_batch_threshold:
            additional_config["prefill_batch_threshold"] = int(prefill_batch_threshold)
        if num_hidden_layers:
            additional_config["num_hidden_layers"] = int(num_hidden_layers)
        engine_args = AsyncEngineArgs(
            model=self.settings.vllm.model,
            max_model_len=self.settings.vllm.max_model_length,
            max_num_batched_tokens=self.settings.vllm.max_num_batched_tokens,
            max_num_seqs=self.settings.vllm.max_num_seqs,
            enable_chunked_prefill=False,
            gpu_memory_utilization=self.settings.vllm.gpu_memory_utilization,
            additional_config=additional_config,
            # Gemma-4 is multimodal but this Forge LLM path serves text-only;
            # zeroing every modality keeps the vision/audio tower from ever
            # precompiling. Without this, _precompile_mm_encoder's video path
            # calls current_platform.mem_get_info(), which the TT platform
            # doesn't implement -> TypeError: 'NoneType' object is not callable.
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        )
        self.logger.info(
            f"Device {self.device_id}: additional_config={engine_args.additional_config}"
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

    def _build_vllm_input(self, request: CompletionRequest):
        if isinstance(request.prompt, str):
            return request.prompt
        elif isinstance(request.prompt, list):
            return {"prompt_token_ids": request.prompt}
        raise ValueError(f"Invalid prompt type: {type(request.prompt)}")

    @log_execution_time(
        "Run VLLM Forge Gemma-4 31B inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[CompletionRequest]):
        """Synchronous wrapper for async inference"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_async(requests))

    async def _run_async(self, requests: list[CompletionRequest]):
        try:
            self.logger.debug(f"Device {self.device_id}: Running Gemma-4 31B inference")

            request = requests[0]
            if request.stream:
                return self._generate_streaming(request)
            else:
                return await self._generate_non_streaming(requests)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Gemma-4 31B inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Gemma-4 31B inference failed: {str(e)}") from e

    async def _generate_streaming(self, request: CompletionRequest):
        """Yields CompletionOutput dicts for streaming generation."""

        chunks = []
        strip_eos = TextUtils.strip_eos
        sampling_params = build_sampling_params(request)

        async for request_output in self.llm_engine.generate(
            self._build_vllm_input(request), sampling_params, request._task_id
        ):
            outputs = request_output.outputs
            if not outputs:
                continue

            for output in outputs:
                chunk_text = output.text
                if not chunk_text:
                    continue

                if chunk_text.endswith(
                    ("</s>", "<|endoftext|>", "<|im_end|>", "<end_of_turn>")
                ):
                    chunk_text = strip_eos(chunk_text)
                    if not chunk_text:
                        continue

                chunks.append(chunk_text)

                yield CompletionOutput(
                    type=CHUNK_TYPE,
                    data=CompletionResult(text=chunk_text),
                )

        self.logger.info(
            f"Device {self.device_id}: Gemma-4 31B streaming generation completed"
        )

        yield CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=""),
        )

    async def process_request_non_streaming(
        self, request: CompletionRequest
    ) -> CompletionOutput:
        self.logger.info(
            f"Device {self.device_id}: Starting Gemma-4 31B non-streaming generation"
        )

        sampling_params = build_sampling_params(request)

        generated_text = []
        async for request_output in self.llm_engine.generate(
            self._build_vllm_input(request), sampling_params, request._task_id
        ):
            if request_output.outputs:
                generated_text.append(request_output.outputs[0].text)

        generated_text = "".join(generated_text)

        self.logger.info(
            f"Device {self.device_id}: Gemma-4 31B non-streaming generation completed"
        )

        return CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=generated_text),
        )

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
