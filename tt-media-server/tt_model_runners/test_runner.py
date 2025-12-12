# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
from typing import AsyncGenerator

from domain.completion_request import CompletionRequest
from domain.completion_response import (
    CompletionStreamChunk,
    FinalResultOutput,
    StreamingChunkOutput,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner


class TestRunnerConfig:
    def __init__(self):
        self.model_warmup_duration_ms = int(os.getenv("TEST_RUNNER_WARMUP_MS", "100"))
        self.streaming_frequency_ms = int(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50"))
        self.total_tokens = int(os.getenv("TEST_RUNNER_TOTAL_TOKENS", "100"))


class TestRunner(BaseDeviceRunner):
    def __init__(
        self,
        device_id: str,
        model_warmup_duration_ms: int = None,
        streaming_frequency_ms: int = None,
        total_tokens: int = None,
    ):
        super().__init__(device_id)
        config = TestRunnerConfig()

        # Use provided values or fall back to config/environment
        self.model_warmup_duration_ms = (
            model_warmup_duration_ms
            if model_warmup_duration_ms is not None
            else config.model_warmup_duration_ms
        )
        self.streaming_frequency_ms = (
            streaming_frequency_ms
            if streaming_frequency_ms is not None
            else config.streaming_frequency_ms
        )
        self.total_tokens = (
            total_tokens if total_tokens is not None else config.total_tokens
        )

        self.logger.info(
            f"TestRunner initialized for device {self.device_id}: "
            f"warmup={self.model_warmup_duration_ms}ms, "
            f"frequency={self.streaming_frequency_ms}ms, "
            f"tokens={self.total_tokens}"
        )

    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        return True

    async def load_model(self) -> bool:
        self.logger.info(
            f"Loading model (simulated warmup: {self.model_warmup_duration_ms}ms)..."
        )
        await asyncio.sleep(self.model_warmup_duration_ms / 1000)
        return True

    async def _run_inference_async(self, requests: list[CompletionRequest]):
        """Returns an async generator for streaming inference."""
        request = requests[0]
        return self._generate_streaming(request)

    async def _generate_streaming(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamingChunkOutput | FinalResultOutput, None]:
        # Use request.max_tokens if provided, otherwise use configured total_tokens
        num_tokens = request.max_tokens if request.max_tokens else self.total_tokens
        self.logger.info(
            f"Running inference: {num_tokens} tokens at "
            f"{self.streaming_frequency_ms}ms intervals"
        )

        frequency_seconds = self.streaming_frequency_ms / 1000

        for i in range(num_tokens):
            await asyncio.sleep(frequency_seconds)

            self.logger.info(
                f"TestRunner yielding streaming chunk: {i} for task {request._task_id}"
            )
            yield StreamingChunkOutput(
                type="streaming_chunk",
                chunk=CompletionStreamChunk(text=f"token_{i}"),
                task_id=request._task_id,
            )

        self.logger.info("TestRunner yielding final result")
        yield FinalResultOutput(
            type="final_result",
            result=CompletionStreamChunk(text="[DONE]"),
            task_id=request._task_id,
        )

    def run_inference(self, requests: list[CompletionRequest]):
        return []
