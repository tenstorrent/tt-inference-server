import asyncio
from typing import AsyncGenerator

from domain.completion_request import CompletionRequest
from domain.completion_response import (
    CompletionStreamChunk,
    FinalResultOutput,
    StreamingChunkOutput,
)
from tt_model_runners.base_device_runner import BaseDeviceRunner


class TestRunner(BaseDeviceRunner):
    def __init__(
        self,
        device_id: str,
        model_warmup_duration_ms: int,
        streaming_frequency_ms: int,
        total_tokens: int,
    ):
        super().__init__(device_id)
        self.model_warmup_duration_ms = model_warmup_duration_ms
        self.streaming_frequency_ms = streaming_frequency_ms
        self.total_tokens = total_tokens
        self.logger.info(
            f"TestRunner initialized for device {self.device_id} with model warmup duration {self.model_warmup_duration_ms}ms and streaming frequency {self.streaming_frequency_ms}ms"
        )

    def close_device(self) -> bool:
        self.logger.info("Closing device...")
        return True

    async def load_model(self) -> bool:
        self.logger.info("Loading model...")
        await asyncio.sleep(self.model_warmup_duration_ms / 1000)
        return True

    async def _run_inference_async(self, requests: list[CompletionRequest]):
        """Returns an async generator for streaming inference."""
        request = requests[0]
        return self._generate_streaming(request)

    async def _generate_streaming(
        self, request: CompletionRequest
    ) -> AsyncGenerator[StreamingChunkOutput | FinalResultOutput, None]:
        self.logger.info("Running inference...")
        await asyncio.sleep(self.streaming_frequency_ms / 1000)
        for i in range(self.total_tokens):
            yield StreamingChunkOutput(
                type="streaming_chunk",
                chunk=CompletionStreamChunk(text="test_text"),
                task_id=request._task_id,
            )
        yield FinalResultOutput(
            type="final_result",
            result=CompletionStreamChunk(text="test_text"),
            task_id=request._task_id,
        )

    def run_inference(self, requests: list[CompletionRequest]):
        return []
