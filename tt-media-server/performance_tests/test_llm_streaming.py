# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import sys
from dataclasses import dataclass

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_tests.llm_streaming_client import (
    LLMStreamingClient,
)


@dataclass
class PerformanceThresholds:
    """Performance thresholds loaded from environment variables."""

    max_per_token_overhead_ms: int = 3  # Allow 10ms overhead per token by default

    @classmethod
    def from_env(cls) -> "PerformanceThresholds":
        return cls(
            max_per_token_overhead_ms=int(
                os.getenv("TEST_RUNNER_MAX_PER_TOKEN_OVERHEAD_MS", "3")
            ),
        )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_performance_full(server_process):
    # Arrange
    thresholds = PerformanceThresholds.from_env()
    os.environ["TEST_RUNNER_TOTAL_TOKENS"] = "100"
    client = LLMStreamingClient(
        url="http://localhost:9000/v1/completions", api_key="your-secret-key"
    )

    # Act
    metrics = await client.make_streaming_request(token_count=100)

    # Assert
    failures = []
    if metrics.received_token_count != 100:
        failures.append(f"Received tokens: {metrics.received_token_count} != 100")

    if metrics.token_overhead_ms > thresholds.max_per_token_overhead_ms:
        failures.append(
            f"Overhead per token: {metrics.token_overhead_ms:.2f}ms > {thresholds.max_per_token_overhead_ms:.2f}ms"
        )

    print(metrics)

    assert not failures, f"Performance test failed: {', '.join(failures)}"


if __name__ == "__main__":
    # Allow running directly for quick testing
    asyncio.run(test_streaming_performance_full())
