# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
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
        # use float to allow fractional values
        self.streaming_frequency_ms = float(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50"))
        self.tokens_per_second = int(os.getenv("TOKENS_PER_SECOND", 0))

        self.tokens_per_second = 12000
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
        frequency_seconds = (
            self.streaming_frequency_ms / TestRunner.MILLISECONDS_PER_SECOND
        )
        task_id = request._task_id

        # Precreate streaming chunks to reduce overhead
        streaming_chunks = [
            StreamingChunkOutput(
                type="streaming_chunk",
                chunk=CompletionStreamChunk(
                    text=f"token_{i}",
                    index=i,
                    finish_reason=None,
                ),
                task_id=task_id,
            )
            for i in range(self.tokens_per_second)
        ]

        start_time = time.perf_counter()

        self.logger.info("Starting device streaming")

        if self.tokens_per_second:
            for i, chunk in enumerate(streaming_chunks):
                yield chunk
            # wait till the second elapses
            elapsed = time.perf_counter() - start_time
            wait_time = 1.0 - elapsed  # 1 second minus elapsed time
            if wait_time > 0:
                self.logger.info("Timer didn't elapse")
                await asyncio.sleep(wait_time)
            else:
                self.logger.info("Timer already elapsed, no need to wait")
        else:
            for i, chunk in enumerate(streaming_chunks):
                # Calculate exact target time for this token
                target_time = start_time + (i * frequency_seconds)
                current_time = time.perf_counter()

                # Only sleep if we're running ahead of schedule
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                yield chunk

        yield FinalResultOutput(
            type="final_result",
            result=CompletionStreamChunk(text="[DONE]", index=0, finish_reason=None),
            task_id=task_id,
            return_result=False,
        )

    def run_inference(self, requests: list[CompletionRequest]):
        return []
