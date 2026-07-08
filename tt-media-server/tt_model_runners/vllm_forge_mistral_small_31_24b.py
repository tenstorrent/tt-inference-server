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


class VLLMForgeMistralSmall31_24BRunner(BaseDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time(
        "VLLM Forge Mistral-Small-3.1 24B model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(
            f"Device {self.device_id}: Loading VLLM Forge Mistral-Small-3.1 24B model..."
        )
        prompt = "Hello, it's me"
        # additional_config mirrors the tt-xla Stage 1 vLLM benchmark
        # (_mistral_small_31_tp_config in tests/benchmark/test_vllm_benchmarks.py):
        # WH-galaxy 8x4 TP mesh, bfp8 weights + KV cache, const eval, and the
        # b1-prefill optimization (min_num_seqs=1 + prefill_batch_threshold=16 ->
        # small prefills served serially instead of a wasted b32 batch).
        # Tunable per-run via env vars (mirrors vllm_forge_gemma4_31b.py):
        #   ENABLE_TRACE=true      decode-graph replay; dominant throughput lever.
        #   OPTIMIZATION_LEVEL=1   validated in Stage 1 on WH galaxy (the gemma
        #                          opt=0 pin is a Blackhole-P300-only workaround).
        #   CPU_SAMPLING=true      opt>=1 + trace requires CPU sampling (TTConfig
        #                          rejects opt>=1 AND trace AND cpu_sampling=false),
        #                          same as the Stage 1 opt=1 path. Flip to on-device
        #                          (false) only together with OPTIMIZATION_LEVEL=0.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        cpu_sampling = os.getenv("CPU_SAMPLING", "true").lower() == "true"
        enable_trace = os.getenv("ENABLE_TRACE", "true").lower() == "true"
        # Explicit TP mesh is REQUIRED on the 32-device galaxy. From
        # ModelConfigs (device_mesh_shape=(8,4) for GALAXY/BLACKHOLE_GALAXY):
        # [8,4] = 4-way TP x 8-way data, matching Stage 1 and Mistral's GQA (8 KV
        # heads -> 2/device). Without it the plugin auto-picks (4,8) -> 1 KV
        # head/device (GQA-degenerate), which fails the stablehlo pipeline with
        # "tensor sharding is incompatible with tensor shape" (and trips SDPA
        # tree-reduction). See tt-metal#43210 for the MGD (4,8) reshape warning.
        mesh_shape = list(getattr(self.settings, "device_mesh_shape", ()) or ())
        if len(mesh_shape) != 2 or mesh_shape[0] * mesh_shape[1] < 2:
            mesh_shape = [8, 4]
        engine_args = AsyncEngineArgs(
            model=self.settings.vllm.model,
            max_model_len=self.settings.vllm.max_model_length,
            max_num_batched_tokens=self.settings.vllm.max_num_batched_tokens,
            max_num_seqs=self.settings.vllm.max_num_seqs,
            enable_chunked_prefill=False,
            gpu_memory_utilization=self.settings.vllm.gpu_memory_utilization,
            # Pixtral-based multimodal model served text-only: zero the image cap
            # so the vision tower never compiles (mirrors Stage 1).
            limit_mm_per_prompt={"image": 0},
            additional_config={
                "enable_const_eval": True,
                "min_context_len": 32,
                "enable_tensor_parallel": True,
                "use_2d_mesh": False,
                "mesh_shape": mesh_shape,
                "experimental_weight_dtype": "bfp_bf8",
                "experimental_kv_cache_dtype": "bfp_bf8",
                # b1-prefill: needs min_num_seqs < max_num_seqs (MAX_NUM_SEQS=32).
                "min_num_seqs": 1,
                "prefill_batch_threshold": 16,
                "cpu_sampling": cpu_sampling,
                "optimization_level": optimization_level,
                "enable_trace": enable_trace,
            },
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
        "Run VLLM Forge Mistral-Small-3.1 24B inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[CompletionRequest]):
        """Synchronous wrapper for async inference"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_async(requests))

    async def _run_async(self, requests: list[CompletionRequest]):
        try:
            self.logger.debug(
                f"Device {self.device_id}: Running Mistral-Small-3.1 24B inference"
            )

            request = requests[0]
            if request.stream:
                return self._generate_streaming(request)
            else:
                return await self._generate_non_streaming(requests)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Mistral-Small-3.1 24B inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(
                f"Mistral-Small-3.1 24B inference failed: {str(e)}"
            ) from e

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
            f"Device {self.device_id}: Mistral-Small-3.1 24B streaming generation completed"
        )

        yield CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=""),
        )

    async def process_request_non_streaming(
        self, request: CompletionRequest
    ) -> CompletionOutput:
        self.logger.info(
            f"Device {self.device_id}: Starting Mistral-Small-3.1 24B non-streaming generation"
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
            f"Device {self.device_id}: Mistral-Small-3.1 24B non-streaming generation completed"
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
