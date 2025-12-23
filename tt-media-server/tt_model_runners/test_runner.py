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


class TestRunner(BaseDeviceRunner):
    MILLISECONDS_PER_SECOND = 1000

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.num_torch_threads = num_torch_threads
        self.streaming_frequency_ms = int(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50"))

        self.logger.info(
            f"TestRunner initialized for device {self.device_id}: "
            f"frequency={self.streaming_frequency_ms}ms, "
        )

    async def load_model(self) -> bool:
        self.logger.info("Loading model...")
        return True

    async def _run_inference_async(self, requests: list[CompletionRequest]):
        """Returns an async generator for streaming inference."""
        request = requests[0]
        return self._generate_streaming(request)

    async def _generate_streaming(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamingChunkOutput | FinalResultOutput, None]:
        self.logger.info(
            f"Running inference: {request.max_tokens} tokens at "
            f"{self.streaming_frequency_ms}ms intervals"
        )

        frequency_seconds = (
            self.streaming_frequency_ms / TestRunner.MILLISECONDS_PER_SECOND
        )

        for i in range(request.max_tokens):
            await asyncio.sleep(frequency_seconds)

            self.logger.info(
                f"TestRunner yielding streaming chunk: {i} for task {request._task_id}"
            )
            yield StreamingChunkOutput(
                type="streaming_chunk",
                chunk=CompletionStreamChunk(
                    text=f"token_{i}", index=i, finish_reason=None
                ),
                task_id=request._task_id,
            )

        self.logger.info("TestRunner yielding final result")
        yield FinalResultOutput(
            type="final_result",
            result=CompletionStreamChunk(text="[DONE]", index=0, finish_reason=None),
            task_id=request._task_id,
            return_result=True,
        )

    def run_inference(self, requests: list[CompletionRequest]):
        return []
