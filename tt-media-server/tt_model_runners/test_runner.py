# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import time
from typing import AsyncGenerator

from domain.completion_request import CompletionRequest
from tt_model_runners.base_device_runner import BaseDeviceRunner


class TestRunner(BaseDeviceRunner):
    MILLISECONDS_PER_SECOND = 1000

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.num_torch_threads = num_torch_threads
        self.streaming_frequency_ms = float(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50"))
        self.tokens_per_second = int(os.getenv("TOKENS_PER_SECOND", 0))
        self.tokens_per_second = 12000

        self.logger.info(
            f"TestRunner initialized for device {self.device_id}: "
            f"frequency={self.streaming_frequency_ms}ms, tokens={self.tokens_per_second}"
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
    ) -> AsyncGenerator[tuple[str, int, str], None]:
        """Yields tuples of (task_id, is_final, text)"""
        task_id = request._task_id

        start_time = time.perf_counter()
        self.logger.info("Starting device streaming")

        # ✅ TIME THE ACTUAL GENERATION
        gen_start = time.perf_counter()
        for i in range(self.tokens_per_second):
            yield (task_id, 0, f"token_{i}")
        gen_time = time.perf_counter() - gen_start

        self.logger.info(
            f"Generator yielded {self.tokens_per_second} items in {gen_time:.4f}s"
        )

        # ✅ FINAL CHUNK
        yield (task_id, 1, "[DONE]")

        total_time = time.perf_counter() - start_time
        self.logger.info(f"Total _generate_streaming time: {total_time:.4f}s")

    def run(self, requests: list[CompletionRequest]):
        return []
