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

    per_token_overhead_ms: int = 3  # Allow 10ms overhead per token by default

    @classmethod
    def from_env(cls) -> "PerformanceThresholds":
        return cls(
            per_token_overhead_ms=int(os.getenv("PERF_PER_TOKEN_OVERHEAD_MS", "3")),
        )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_performance_full(server_process):
    thresholds = PerformanceThresholds.from_env()

    client = LLMStreamingClient(
        url="http://localhost:9000/v1/completions", api_key="your-secret-key"
    )
    metrics = await client.make_streaming_request(token_count=100)

    print(metrics)

    # Collect all failures
    failures = []

    if metrics.received_token_count != 100:
        failures.append(f"RECEIVED CHUNKS: {metrics.received_token_count} != 100")

    if metrics.mean_receive_interval_ms >= thresholds.per_token_overhead_ms:
        failures.append(
            f"MEAN RECEIVE INTERVAL: {metrics.mean_receive_interval_ms:.2f}ms >= {thresholds.per_token_overhead_ms:.2f}ms"
        )

    assert not failures, f"Performance test failed: {', '.join(failures)}"


if __name__ == "__main__":
    # Allow running directly for quick testing
    asyncio.run(test_streaming_performance_full())
