# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import asyncio
import os
import traceback

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.helpers import log_execution_time
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.sampling_params import RequestOutputKind


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
            pass  # Just consume the generator for warmup
        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    @log_execution_time(
        "Run VLLM Forge inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run_inference(self, requests: list[CompletionRequest]):
        """Synchronous wrapper for async inference"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._run_inference_async(requests))

    def _build_sampling_params(self, request: CompletionRequest) -> SamplingParams:
        """Build sampling params for a single request."""
        return SamplingParams(
            temperature=request.temperature if request.temperature else 0.8,
            top_p=request.top_p if request.top_p else 0.95,
            max_tokens=request.max_tokens if request.max_tokens else 16,
            output_kind=RequestOutputKind.DELTA
            if request.stream
            else RequestOutputKind.FINAL_ONLY,
        )

    async def _run_inference_async(self, requests: list[CompletionRequest]):
        try:
            task_ids = [r._task_id for r in requests]
            self.logger.info(
                f"Device {self.device_id}: _run_inference_async called with {len(requests)} request(s), "
                f"task_ids={task_ids}"
            )

            # Build sampling params for each request
            request_params = [
                (request, self._build_sampling_params(request)) for request in requests
            ]

            # Check if any request requires streaming
            has_streaming = any(r.stream for r in requests)
            self.logger.info(
                f"Device {self.device_id}: has_streaming={has_streaming}, "
                f"stream_flags={[r.stream for r in requests]}"
            )

            if has_streaming:
                return self._generate_streaming_batch(request_params)
            else:
                return await self._generate_non_streaming_batch(request_params)
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Inference failed: {type(e).__name__}: {e}"
            )
            self.logger.error(
                f"Device {self.device_id}: Full traceback: {traceback.format_exc()}"
            )
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    async def _generate_streaming_batch(
        self, request_params: list[tuple[CompletionRequest, SamplingParams]]
    ):
        """Handle streaming generation for multiple requests concurrently."""
        task_ids = [req._task_id for req, _ in request_params]
        self.logger.info(
            f"Device {self.device_id}: _generate_streaming_batch called with "
            f"{len(request_params)} request(s), task_ids={task_ids}"
        )

        # Use a queue to collect chunks from all generators
        chunk_queue: asyncio.Queue = asyncio.Queue()
        chunk_counts = {req._task_id: 0 for req, _ in request_params}

        async def stream_to_queue(
            request: CompletionRequest, sampling_params: SamplingParams
        ):
            """Stream chunks from a single request to the shared queue."""
            generated_text = ""
            local_chunk_count = 0
            self.logger.info(
                f"Device {self.device_id}: Starting stream_to_queue for task {request._task_id}"
            )
            try:
                async for request_output in self.llm_engine.generate(
                    request.prompt, sampling_params, request._task_id
                ):
                    for output in request_output.outputs:
                        cleaned_text = TextUtils.clean_text(output.text)
                        if not cleaned_text:
                            continue
                        generated_text += cleaned_text
                        local_chunk_count += 1
                        await chunk_queue.put({
                            "type": "streaming_chunk",
                            "chunk": CompletionStreamChunk(text=cleaned_text),
                            "task_id": request._task_id,
                        })
                self.logger.info(
                    f"Device {self.device_id}: stream_to_queue finished for task {request._task_id}, "
                    f"produced {local_chunk_count} chunks"
                )
            finally:
                # Signal completion for this request
                await chunk_queue.put({
                    "type": "final_result",
                    "result": CompletionStreamChunk(text=generated_text),
                    "task_id": request._task_id,
                    "return": False,
                })

        # Start all streaming tasks concurrently
        self.logger.info(
            f"Device {self.device_id}: Creating {len(request_params)} concurrent stream tasks"
        )
        tasks = [
            asyncio.create_task(stream_to_queue(request, sampling_params))
            for request, sampling_params in request_params
        ]

        # Yield chunks as they arrive until all tasks complete
        completed_count = 0
        total_requests = len(request_params)
        total_chunks_yielded = 0

        self.logger.info(
            f"Device {self.device_id}: Waiting for chunks from {total_requests} concurrent streams"
        )
        while completed_count < total_requests:
            chunk = await chunk_queue.get()
            total_chunks_yielded += 1
            if chunk["type"] == "streaming_chunk":
                chunk_counts[chunk["task_id"]] += 1
            yield chunk
            if chunk["type"] == "final_result":
                completed_count += 1
                self.logger.info(
                    f"Device {self.device_id}: Task {chunk['task_id']} completed, "
                    f"{completed_count}/{total_requests} done"
                )

        # Ensure all tasks are done (should already be complete)
        await asyncio.gather(*tasks)

        self.logger.info(
            f"Device {self.device_id}: Streaming generation completed. "
            f"Total chunks yielded: {total_chunks_yielded}, per-task: {chunk_counts}"
        )

    async def _generate_non_streaming_batch(
        self, request_params: list[tuple[CompletionRequest, SamplingParams]]
    ):
        """Handle non-streaming generation for multiple requests concurrently."""
        self.logger.info(
            f"Device {self.device_id}: Starting non-streaming generation for {len(request_params)} request(s)"
        )

        async def generate_single(
            request: CompletionRequest, sampling_params: SamplingParams
        ) -> CompletionStreamChunk:
            generated_text = ""
            async for request_output in self.llm_engine.generate(
                request.prompt, sampling_params, request._task_id
            ):
                if request_output.outputs:
                    generated_text = TextUtils.clean_text(
                        request_output.outputs[0].text
                    )
                    break
            return CompletionStreamChunk(text=generated_text)

        # Run all generations concurrently
        tasks = [
            generate_single(request, sampling_params)
            for request, sampling_params in request_params
        ]
        results = await asyncio.gather(*tasks)

        self.logger.info(f"Device {self.device_id}: Non-streaming generation completed")

        return list(results)
