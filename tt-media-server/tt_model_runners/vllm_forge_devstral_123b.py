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


class VLLMForgeDevstral123BRunner(BaseDeviceRunner):
    """Forge vLLM runner for Devstral-2-123B on the BH Galaxy as a 4x8 DP+TP mesh.

    Unlike the pure-TP forge runners (gemma-4-31b, qwen3-32b), Devstral is the
    first DP+TP model wired into tt-media-server, so this runner sets DP-specific
    additional_config keys that the TP runners never touch:
      - enable_data_parallel: True        -> fan the request batch across the DP replicas
      - shard_weights_on_batch_axis: False -> classic DP+TP: weights are TP-sharded and
                                              REPLICATED across the 4 DP replicas (fewer CCLs)
      - use_2d_mesh: True                  -> the (DP, TP) grid is a 2D mesh (TP runners use 1D)
      - mesh_shape: [DP, TP]               -> taken from settings.device_mesh_shape ((4, 8))
    These mirror the validated standalone config in tt-xla
    (branch ssalice/devstral-qwen-wip-07-13-2026, tests/benchmark/test_vllm_benchmarks.py
    id "devstral-123b-galaxy-tp", and the DP+TP generation test). The plugin owns
    the 2D split internally; tt-media-server hands it one worker over all 32 chips.

    Requires env TT_RUNTIME_USING_BH_GALAXY=1 (set via the dev/cnn.yaml spec).
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)

    @log_execution_time(
        "VLLM Forge Devstral-2-123B model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        self.logger.info(
            f"Device {self.device_id}: Loading VLLM Forge Devstral-2-123B model "
            f"(mesh_shape={tuple(self.settings.device_mesh_shape)} DP+TP)..."
        )
        prompt = "Hello, it's me"
        # Tunable per-run via env vars (mirrors vllm_forge_qwen_32b.py). Defaults
        # follow the validated tt-xla devstral-qwen-wip-07-13-2026 config:
        #   ENABLE_TRACE=true     decode-graph replay (dominant decode-speed lever).
        #   CPU_SAMPLING=true     REQUIRED on the 2D DP+TP mesh today: the on-device
        #                         sampler produces token-soup with >1 sample/device on
        #                         a 2D mesh (tt-inference-server#4440). Flip to false
        #                         once that is fixed. NOTE: this is the opposite of the
        #                         pure-TP runners' default (they use on-device sampling).
        #   OPTIMIZATION_LEVEL=0  matches the tt-xla benchmark (opt unset => 0). Raise
        #                         once the forge wheel supports opt>=1 on BH galaxy.
        # Weights stay bfp_bf8 (the fp8 checkpoint is dequantised to bf16 then stored
        # as bfp8); measured faster than bf16 and required to fit the 123B in the TP-8
        # weight slice.
        optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "0"))
        cpu_sampling = os.getenv("CPU_SAMPLING", "true").lower() == "true"
        enable_trace = os.getenv("ENABLE_TRACE", "true").lower() == "true"
        mesh_shape = list(self.settings.device_mesh_shape)  # (DP, TP) -> [4, 8]
        engine_args = AsyncEngineArgs(
            model=self.settings.vllm.model,
            max_model_len=self.settings.vllm.max_model_length,
            max_num_batched_tokens=self.settings.vllm.max_num_batched_tokens,
            max_num_seqs=self.settings.vllm.max_num_seqs,
            enable_chunked_prefill=False,
            gpu_memory_utilization=self.settings.vllm.gpu_memory_utilization,
            additional_config={
                "enable_const_eval": True,
                # 32 is the validated value across every tt-xla Devstral source
                # (examples/.../Devstral-2-123B-Instruct-2512/service.sh, the
                # benchmark _tp_config default, and the DP+TP generation test).
                # NOT settings.vllm.min_context_length, whose default is 128.
                "min_context_len": 32,
                "enable_tensor_parallel": True,
                "enable_data_parallel": True,
                "shard_weights_on_batch_axis": False,
                "use_2d_mesh": True,
                "mesh_shape": mesh_shape,
                "experimental_weight_dtype": "bfp_bf8",
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
        "Run VLLM Forge Devstral-2-123B inference",
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
                f"Device {self.device_id}: Running Devstral-2-123B inference"
            )

            request = requests[0]
            if request.stream:
                return self._generate_streaming(request)
            else:
                return await self._generate_non_streaming(requests)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Devstral-2-123B inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Devstral-2-123B inference failed: {str(e)}") from e

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
            f"Device {self.device_id}: Devstral-2-123B streaming generation completed"
        )

        yield CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=""),
        )

    async def process_request_non_streaming(
        self, request: CompletionRequest
    ) -> CompletionOutput:
        self.logger.info(
            f"Device {self.device_id}: Starting Devstral-2-123B non-streaming generation"
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
            f"Device {self.device_id}: Devstral-2-123B non-streaming generation completed"
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
