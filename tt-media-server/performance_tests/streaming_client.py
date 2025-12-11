# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""HTTP streaming client for performance testing.

This module provides an async HTTP client that makes streaming requests
and collects detailed timing metrics for performance analysis.
"""

import asyncio
import os
import time
from dataclasses import dataclass

import aiohttp

from performance_tests.streaming_metrics import ChunkTiming, StreamingMetrics


@dataclass
class StreamingRequestConfig:
    """Configuration for a streaming request."""

    base_url: str = "http://localhost:8000"
    endpoint: str = "/v1/completions"
    api_key: str = "your-secret-key"
    model: str = "test"
    prompt: str = "Hello"
    max_tokens: int = 100
    timeout_seconds: int = 300
    # Expected values from TestRunner config (for validation)
    expected_chunks: int = None
    expected_frequency_ms: float = None

    @classmethod
    def from_env(cls) -> "StreamingRequestConfig":
        """Create config from environment variables."""
        return cls(
            base_url=os.getenv("TEST_SERVER_URL", "http://localhost:8000"),
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


class StreamingClient:
    """Async HTTP client for streaming performance tests.

    Makes streaming HTTP requests and collects detailed timing information
    for each received chunk.
    """

    def __init__(self, config: StreamingRequestConfig):
        self.config = config

    async def make_streaming_request(self) -> StreamingMetrics:
        """Make a streaming request and collect timing metrics.

        Returns:
            StreamingMetrics with timing information for all received chunks.
        """
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

        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
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

                async for line in response.content:
                    receive_timestamp_ns = time.time_ns()
                    line_text = line.decode("utf-8").strip()

                    if not line_text:
                        continue

                    chunk_timing = ChunkTiming(
                        chunk_index=chunk_index,
                        receive_timestamp_ns=receive_timestamp_ns,
                        text=line_text,
                    )
                    metrics.add_chunk(chunk_timing)
                    chunk_index += 1

        return metrics


async def run_concurrent_requests(
    config: StreamingRequestConfig,
    num_requests: int = 1,
) -> list[StreamingMetrics]:
    """Run multiple concurrent streaming requests.

    Args:
        config: Request configuration
        num_requests: Number of concurrent requests to make

    Returns:
        List of StreamingMetrics, one for each request
    """
    client = StreamingClient(config)
    tasks = [client.make_streaming_request() for _ in range(num_requests)]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def run_sequential_requests(
    config: StreamingRequestConfig,
    num_requests: int = 1,
) -> list[StreamingMetrics]:
    """Run multiple sequential streaming requests.

    Args:
        config: Request configuration
        num_requests: Number of sequential requests to make

    Returns:
        List of StreamingMetrics, one for each request
    """
    client = StreamingClient(config)
    results = []
    for _ in range(num_requests):
        result = await client.make_streaming_request()
        results.append(result)
    return results
