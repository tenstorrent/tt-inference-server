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
        self.streaming_frequency_ms = float(
            os.getenv("TEST_RUNNER_FREQUENCY_MS", "0.01")
        )

        # Pre-cache tokens for zero-allocation streaming
        self._token_cache = [f"tok{i} " for i in range(1000)]

        target_rate = (
            1000 / self.streaming_frequency_ms
            if self.streaming_frequency_ms > 0
            else float("inf")
        )
        self.logger.info(
            f"TestRunner initialized for device {self.device_id}: "
            f"frequency={self.streaming_frequency_ms}ms, "
            f"target_rate={target_rate:.0f} tokens/sec"
        )

    async def warmup(self) -> bool:
        self.logger.info("Loading model...")
        return True

    async def _run_async(self, requests: list[CompletionRequest]):
        """Returns an async generator for streaming inference."""
        request = requests[0]
        return self._generate_streaming(request)

    async def _generate_streaming(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamingChunkOutput | FinalResultOutput, None]:
        task_id = request._task_id
        max_tokens = request.max_tokens
        use_memory_queue = self.settings.use_memory_queue

        # Log start
        target_rate = (
            1000 / self.streaming_frequency_ms
            if self.streaming_frequency_ms > 0
            else float("inf")
        )
        expected_time_ms = (
            max_tokens / target_rate * 1000 if target_rate != float("inf") else 0
        )
        self.logger.info(
            f"[TestRunner] Starting generation: {max_tokens} tokens, "
            f"target_rate={target_rate:.0f} tok/s, "
            f"expected_time={expected_time_ms:.1f}ms"
        )

        start_time = time.perf_counter()
        last_log_time = start_time
        last_log_tokens = 0
        log_interval = max(1000, max_tokens // 10)  # Log ~10 times or every 1000 tokens

        # Batch size for yielding control - larger batch = higher throughput
        batch_size = 100

        tokens_generated = 0

        while tokens_generated < max_tokens:
            batch_end = min(tokens_generated + batch_size, max_tokens)

            # Generate batch without any sleeps
            for i in range(tokens_generated, batch_end):
                token_text = self._token_cache[i % 1000]

                if use_memory_queue:
                    yield (task_id, 0, token_text)
                else:
                    yield StreamingChunkOutput(
                        type="streaming_chunk",
                        chunk=CompletionStreamChunk(
                            text=token_text,
                            index=i,
                            finish_reason=None,
                        ),
                        task_id=task_id,
                    )

            tokens_generated = batch_end

            # Log progress
            if tokens_generated - last_log_tokens >= log_interval:
                now = time.perf_counter()
                elapsed = now - start_time
                rate = tokens_generated / elapsed if elapsed > 0 else 0
                self.logger.info(
                    f"[TestRunner] Progress: {tokens_generated}/{max_tokens} tokens, "
                    f"elapsed={elapsed * 1000:.1f}ms, rate={rate:.0f} tok/s"
                )
                last_log_time = now
                last_log_tokens = tokens_generated

            # Yield control to event loop after each batch
            await asyncio.sleep(0)

        # Final stats
        end_time = time.perf_counter()
        total_time = end_time - start_time
        actual_rate = max_tokens / total_time if total_time > 0 else 0

        self.logger.info(
            f"[TestRunner] === COMPLETE ===\n"
            f"  Tokens: {max_tokens}\n"
            f"  Time: {total_time * 1000:.2f}ms\n"
            f"  Rate: {actual_rate:.0f} tokens/sec\n"
            f"  Target: {target_rate:.0f} tokens/sec"
        )

        # Yield final marker
        if use_memory_queue:
            yield (task_id, 1, "[DONE]")
        else:
            yield FinalResultOutput(
                type="final_result",
                result=CompletionStreamChunk(
                    text="[DONE]", index=0, finish_reason=None
                ),
                task_id=task_id,
                return_result=True,
            )

    def run(self, requests: list[CompletionRequest]):
        return []
