# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
from dataclasses import dataclass

import pytest

from performance_tests.conftest import SERVER_BASE_URL, TEST_RUNNER_FREQUENCY_MS
from performance_tests.llm_streaming_client import (
    LLMStreamingClient,
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PerformanceThresholds:
    """Performance thresholds loaded from environment variables."""

    max_per_token_overhead_us: int = 3000  # Allow 3000us (3ms) overhead per token by default

    @classmethod
    def from_env(cls) -> "PerformanceThresholds":
        return cls(
            max_per_token_overhead_us=int(
                os.getenv("TEST_RUNNER_MAX_PER_TOKEN_OVERHEAD_US", "3000")
            ),
        )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_performance_full(server_process):
    # Arrange
    thresholds = PerformanceThresholds.from_env()
    client = LLMStreamingClient(
        url=f"{SERVER_BASE_URL}/v1/completions", api_key="your-secret-key"
    )

    # Act
    token_count = 2048
    metrics = await client.make_streaming_request(token_count=token_count)

    # Assert
    failures = []
    if metrics.received_token_count != token_count:
        failures.append(
            f"Received tokens: {metrics.received_token_count} != {token_count}"
        )

    token_overhead_us = metrics.calculate_overhead_us(
        test_runner_frequency_ms=TEST_RUNNER_FREQUENCY_MS
    )
    if token_overhead_us > thresholds.max_per_token_overhead_us:
        failures.append(
            f"Overhead per token: {token_overhead_us:.2f}us > {thresholds.max_per_token_overhead_us:.2f}us"
        )

    # CI-friendly report format (key=value for easy parsing)
    print("\n::CI_REPORT_START::")
    print(f"tokens_received={metrics.received_token_count}")
    print(f"total_time_ms={metrics.total_streaming_time_ms:.2f}")
    print(f"mean_interval_ms={metrics.mean_receive_interval_ms:.2f}")
    print(f"throughput_tps={metrics.throughput_tokens_per_second:.2f}")
    print(f"overhead_us={token_overhead_us:.2f}")
    print(f"threshold_us={thresholds.max_per_token_overhead_us}")
    print("::CI_REPORT_END::")

    assert not failures, f"Performance test failed: {', '.join(failures)}"
