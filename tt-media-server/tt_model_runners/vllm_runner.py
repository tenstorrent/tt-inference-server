# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import asyncio
import os
import time
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


class VLLMForgeRunner(BaseDeviceRunner):
    # Sampling defaults for Forge LLM inference (overrides global greedy defaults)
    SAMPLING_DEFAULTS = {
        "temperature": 0.6,
        # repetition_penalty=1.0 (off) to match every sibling runner — the Forge TP
        # runners (Llama-70B, Qwen-32B, Gemma-4-31B), the metal runner, and the
        # repo-wide _DEFAULT_SAMPLING_PARAMS all use 1.0. The old 1.1 here was the
        # lone outlier and triggered the O(N^2) decode regression (tt-xla#4278) on
        # the seeded/penalty path. Opting into 1.1 per-request stays cheap once the
        # tt-xla incremental-counts fix lands (flat ~50 vs ~76 tok/s, no decay).
        "repetition_penalty": 1.0,
    }

    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time(
        "VLLM model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM model...")
        prompt = "Hello, it's me"
        # Tunable per-run via env vars. Defaults are 1.2.0-safe:
        # optimization_level=0 (required for the 1.2.0 wheel — opt>=1 aborts in
        # the tt-mlir MemoryLayoutPropagation pass; see tt-xla#4990),
        # cpu_sampling enabled. ENABLE_TRACE is off-by-default and gated on
        # opt_level=0; replaying the decode graph works around the 1.2.0
        # decode regression (+16-89% aggregate tok/s validated across the 5
        # forge LLM P150 specs at b4/16K).
        # Caveats for ENABLE_TRACE:
        #   - vllm_tt's TTConfig rejects enable_trace=True + opt>=1 + cpu_sampling=False
        #     (only safe at optimization_level=0).
        #   - crashes at high batch (b16 hits a RuntimeError in
        #     tt::runtime::ttnn::operations::trace::run; b16/16K won't compile).
        #   - trace-capture needs extra DRAM scratch — validate fit on 7B+ models
        #     before enabling.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "0"))
        cpu_sampling = os.getenv("CPU_SAMPLING", "true").lower() == "true"
        enable_trace = os.getenv("ENABLE_TRACE", "false").lower() == "true"
        # KV-cache quantization, independent of experimental_weight_dtype. The
        # tt-xla plugin reads experimental_kv_cache_dtype: "bfp_bf8" → BFP8 KV
        # (halves KV-cache footprint vs bf16, lets larger contexts fit),
        # "bfp_bf4" → BFP4, "" / unset → bf16 default. Env-driven so it can be
        # set per-model via spec env_vars the same way as MAX_MODEL_LENGTH.
        kv_cache_dtype = os.getenv("KV_CACHE_DTYPE", "")
        # On-device chunked-SDPA prefill chunk size (tt-xla plugin reads
        # additional_config["prefill_chunk_size"]). Without it, prefill for a
        # long prompt allocates one contiguous SDPA buffer for the full context
        # and OOMs the DRAM banks at 32K/64K; chunking runs the op in slices
        # (e.g. 2048) so it fits. Independent of vLLM's scheduler-level
        # enable_chunked_prefill. Env-driven; only passed when set so existing
        # short-context models keep the default (whole-prompt) path.
        prefill_chunk_size = os.getenv("PREFILL_CHUNK_SIZE")
        # Matmul dest-accumulation dtype. fp32_dest_acc_en=false uses bf16 dest
        # accumulation (smaller matmul dest buffers, less DRAM) vs the fp32
        # default; required to fit alongside a higher gpu_memory_utilization
        # (the tt-xla chunked-prefill sweep sets this false at gmu 0.35).
        # Env-driven; only passed when set so other models keep the plugin
        # default. "true"/"false".
        fp32_dest_acc_en = os.getenv("FP32_DEST_ACC_EN")
        # Debug/iteration knob: override the model's transformer depth (the tt-xla
        # plugin reads additional_config["num_hidden_layers"] and rewrites the HF
        # config + filters weights at load, so only N layers compile/run). Slashes
        # compile time for pipecleaning. 0/unset = full model. e.g. NUM_HIDDEN_LAYERS=1.
        num_hidden_layers = os.getenv("NUM_HIDDEN_LAYERS")
        additional_config = {
            "enable_const_eval": True,
            "min_context_len": self.settings.vllm.min_context_length,
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
        )
        self.logger.info(
            f"Device {self.device_id}: additional_config={engine_args.additional_config}"
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.logger.info(f"Device {self.device_id}: Starting model warmup")
        warmup_sampling_params = SamplingParams(
            **self.SAMPLING_DEFAULTS,
            max_tokens=10,
        )
        warmup_generator = self.llm_engine.generate(
            prompt, warmup_sampling_params, "warmup_task_id"
        )
        async for _ in warmup_generator:
            pass  # Just consume the generator for warmup
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    def _build_vllm_input(self, request: CompletionRequest):
        if isinstance(request.prompt, str):
            return request.prompt
        elif isinstance(request.prompt, list):
            return {"prompt_token_ids": request.prompt}
        raise ValueError(f"Invalid prompt type: {type(request.prompt)}")

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
        """Yields CompletionOutput dicts for streaming generation."""

        chunks = []
        strip_eos = TextUtils.strip_eos
        sampling_params = build_sampling_params(request, self.SAMPLING_DEFAULTS)

        start = time.time()
        num_tokens = 0
        async for request_output in self.llm_engine.generate(
            self._build_vllm_input(request), sampling_params, request._task_id
        ):
            outputs = request_output.outputs
            if not outputs:
                continue

            # token_ids is cumulative in vLLM, so the last value is the total
            num_tokens = len(outputs[0].token_ids)

            for output in outputs:
                chunk_text = output.text
                if not chunk_text:
                    continue

                if chunk_text.endswith(("</s>", "<|endoftext|>", "<|im_end|>")):
                    chunk_text = strip_eos(chunk_text)
                    if not chunk_text:
                        continue

                chunks.append(chunk_text)

                yield CompletionOutput(
                    type=CHUNK_TYPE,
                    data=CompletionResult(text=chunk_text),
                )

        duration = time.time() - start
        rate = num_tokens / duration if duration > 0 else 0.0
        self.logger.info(
            f"Device {self.device_id}: Streaming generation completed: "
            f"{num_tokens} tokens in {duration:.4f} seconds ({rate:.2f} tok/sec)"
        )

        yield CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=""),
        )

    async def process_request_non_streaming(
        self, request: CompletionRequest
    ) -> CompletionOutput:
        self.logger.info(f"Device {self.device_id}: Starting non-streaming generation")

        sampling_params = build_sampling_params(request, self.SAMPLING_DEFAULTS)

        start = time.time()
        generated_text = []
        num_tokens = 0
        async for request_output in self.llm_engine.generate(
            self._build_vllm_input(request), sampling_params, request._task_id
        ):
            if request_output.outputs:
                generated_text.append(request_output.outputs[0].text)
                # token_ids is cumulative in vLLM, so the last value is the total
                num_tokens = len(request_output.outputs[0].token_ids)

        generated_text = "".join(generated_text)

        duration = time.time() - start
        rate = num_tokens / duration if duration > 0 else 0.0
        self.logger.info(
            f"Device {self.device_id}: Non-streaming generation completed: "
            f"{num_tokens} tokens in {duration:.4f} seconds ({rate:.2f} tok/sec)"
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
