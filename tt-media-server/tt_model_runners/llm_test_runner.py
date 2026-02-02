# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from typing import AsyncGenerator

from domain.completion_request import CompletionRequest
from domain.completion_response import (
    CompletionStreamChunk,
    FinalResultOutput,
    StreamingChunkOutput,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner


class LLMTestRunner(BaseDeviceRunner):
    """Test runner for LLM streaming performance tests.

    Generates fake tokens as fast as possible to test the streaming
    infrastructure without requiring actual model inference.
    """

    def __init__(self, device_id: str, num_torch_threads: int = 1):
        super().__init__(device_id, num_torch_threads)
        self.num_torch_threads = num_torch_threads

        self.logger.info(
            f"LLMTestRunner initialized for device {self.device_id}: "
            f"sending tokens as fast as possible (no frequency limit)"
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

        # StreamingChunkOutput format - generate tokens as fast as possible
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
            for i in range(request.max_tokens)
        ]

        for chunk in streaming_chunks:
            yield chunk

        yield FinalResultOutput(
            type="final_result",
            result=CompletionStreamChunk(text="[DONE]", index=0, finish_reason=None),
            task_id=task_id,
            return_result=True,
        )

    def run(self, requests: list[CompletionRequest]):
        return []
