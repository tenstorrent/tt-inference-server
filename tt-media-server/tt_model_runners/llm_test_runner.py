# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
from typing import AsyncGenerator

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionOutput, CompletionResult
from tt_model_runners.base_device_runner import BaseDeviceRunner

CHUNK_TYPE = "streaming_chunk"
FINAL_TYPE = "final_result"


class LLMTestRunner(BaseDeviceRunner):
    """Test runner for LLM streaming performance tests.

    Generates fake tokens at a configurable frequency to test the streaming
    infrastructure without requiring actual model inference.
    """

    MILLISECONDS_PER_SECOND = 1000

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.num_torch_threads = num_torch_threads
        # Frequency is set via TEST_RUNNER_FREQUENCY_MS env var, configured in
        # performance_tests/conftest.py which is the single source of truth
        self.streaming_frequency_ms = float(os.getenv("TEST_RUNNER_FREQUENCY_MS", "1"))

        self.logger.info(
            f"LLMTestRunner initialized for device {self.device_id}: "
            f"frequency={self.streaming_frequency_ms}ms, "
        )

    async def warmup(self) -> bool:
        self.logger.info("Loading model...")
        return True

    async def _run_async(self, requests: list[CompletionRequest]):
        """Match VLLMRunner behavior: async generator for streaming, list for non-streaming."""
        request = requests[0]
        if request.stream:
            return self._generate_streaming(request)
        else:
            return await self._generate_non_streaming(requests)

    async def _generate_non_streaming(self, requests: list[CompletionRequest]):
        """Non-streaming async inference - returns list of CompletionOutput."""
        results = []
        for request in requests:
            tokens = [f"token_{i}" for i in range(request.max_tokens)]
            final_text = "".join(tokens)
            results.append(
                CompletionOutput(
                    type=FINAL_TYPE,
                    data=CompletionResult(text=final_text),
                )
            )
        return results

    async def _generate_streaming(
        self, request: CompletionRequest
    ) -> AsyncGenerator[CompletionOutput, None]:
        frequency_seconds = (
            self.streaming_frequency_ms / LLMTestRunner.MILLISECONDS_PER_SECOND
        )

        chunks = []
        start_time = time.perf_counter()

        for i in range(request.max_tokens):
            # Calculate exact target time for this token
            target_time = start_time + (i * frequency_seconds)
            current_time = time.perf_counter()

            # Only sleep if we're running ahead of schedule
            sleep_time = target_time - current_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            chunk_text = f"token_{i}"
            chunks.append(chunk_text)

            yield CompletionOutput(
                type=CHUNK_TYPE,
                data=CompletionResult(text=chunk_text),
            )

        self.logger.info(f"Device {self.device_id}: Streaming generation completed")

        final_text = ""

        yield CompletionOutput(
            type=FINAL_TYPE,
            data=CompletionResult(text=final_text),
        )

    def run(self, requests: list[CompletionRequest]):
        """Non-streaming inference - returns complete results."""
        results = []
        for request in requests:
            tokens = [f"token_{i}" for i in range(request.max_tokens)]
            final_text = "".join(tokens)
            results.append(
                CompletionOutput(
                    type=FINAL_TYPE,
                    data=CompletionResult(text=final_text),
                )
            )
        return results
