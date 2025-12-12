# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import time
from dataclasses import dataclass

import aiohttp

from performance_tests.streaming_metrics import ChunkTiming, StreamingMetrics


@dataclass
class LLMStreamingRequestConfig:
    """Configuration for a streaming request."""

    base_url: str = "http://localhost:9000"
    endpoint: str = "/v1/completions"
    api_key: str = "your-secret-key"
    model: str = "test"
    prompt: str = "Hello"
    max_tokens: int = 100
    # Expected values from TestRunner config (for validation)
    expected_chunks: int = None
    expected_frequency_ms: float = None

    @classmethod
    def from_env(cls) -> "LLMStreamingRequestConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("TEST_SERVER_URL", "http://localhost:9000"),
            api_key=os.getenv("TEST_API_KEY", "your-secret-key"),
            max_tokens=int(os.getenv("TEST_RUNNER_TOTAL_TOKENS", "100")),
            expected_chunks=int(os.getenv("TEST_RUNNER_TOTAL_TOKENS", "100")),
            expected_frequency_ms=float(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50")),
        )

    @property
    def url(self) -> str:
        return f"{self.base_url}{self.endpoint}"

    def build_payload(self) -> dict:
        """Build the request payload for a streaming completion request."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": 0,
        }


class LLMStreamingClient:
    def __init__(self, config: LLMStreamingRequestConfig):
        self.config = config

    async def make_streaming_request(self) -> StreamingMetrics:
        metrics = StreamingMetrics(
            expected_chunks=self.config.expected_chunks,
            expected_send_interval_ms=self.config.expected_frequency_ms,
        )
        metrics.request_start_ns = time.time_ns()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        sock_read_timeout_seconds = 10
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_read=sock_read_timeout_seconds,
        )
        chunk_index = 0

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.config.url,
                json=self.config.build_payload(),
                headers=headers,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(
                        f"Request failed with status {response.status}: {error_text}"
                    )

                async for _ in response.content.iter_any():
                    receive_timestamp_ns = time.time_ns()
                    chunk_timing = ChunkTiming(
                        chunk_index=chunk_index,
                        receive_timestamp_ns=receive_timestamp_ns,
                    )
                    metrics.add_chunk(chunk_timing)
                    chunk_index += 1

                    # Show progress every 10 chunks
                    if chunk_index % 10 == 0:
                        print(".", end="", flush=True)

        print(f" Done! ({chunk_index} chunks)")

        return metrics
