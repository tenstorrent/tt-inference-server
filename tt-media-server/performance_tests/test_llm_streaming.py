# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""LLM Streaming Performance Tests.

This module provides pytest-based performance tests for the LLM streaming endpoint.
Tests verify that the streaming infrastructure meets performance requirements:

1. No chunk loss: All tokens sent by TestRunner are received by the client
2. Latency ratio: The ratio of receive/send intervals stays below threshold

Configuration via environment variables:
    TEST_SERVER_URL: Server URL (default: http://localhost:8000)
    TEST_RUNNER_WARMUP_MS: TestRunner warmup time (default: 100)
    TEST_RUNNER_FREQUENCY_MS: Token emission interval (default: 50)
    TEST_RUNNER_TOTAL_TOKENS: Number of tokens to emit (default: 100)

    Performance thresholds:
    PERF_MAX_CHUNK_LOSS_RATIO: Maximum allowed chunk loss (default: 0.0)
    PERF_MAX_LATENCY_RATIO: Maximum receive/send interval ratio (default: 1.5)

Usage:
    # Configure TestRunner and run tests
    export TEST_RUNNER_FREQUENCY_MS=50
    export TEST_RUNNER_TOTAL_TOKENS=100
    pytest performance_tests/test_llm_streaming.py -v

    # Run with custom thresholds
    export PERF_MAX_LATENCY_RATIO=1.2
    pytest performance_tests/test_llm_streaming.py -v
"""

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_tests.streaming_client import (
    StreamingClient,
    StreamingRequestConfig,
)
from performance_tests.streaming_metrics import StreamingMetrics


@dataclass
class PerformanceThresholds:
    """Performance thresholds loaded from environment variables."""

    max_chunk_loss_ratio: float = 0.0  # No loss allowed by default
    max_latency_ratio: float = 1.10  # Allow 10% overhead by default
    max_time_to_first_chunk_ms: float = 1000.0  # 1 second max TTFC

    @classmethod
    def from_env(cls) -> "PerformanceThresholds":
        return cls(
            max_chunk_loss_ratio=float(os.getenv("PERF_MAX_CHUNK_LOSS_RATIO", "0.0")),
            max_latency_ratio=float(os.getenv("PERF_MAX_LATENCY_RATIO", "1.10")),
            max_time_to_first_chunk_ms=float(os.getenv("PERF_MAX_TTFC_MS", "1000.0")),
        )


def get_server_config() -> StreamingRequestConfig:
    """Get server configuration from environment."""
    return StreamingRequestConfig.from_env()


def check_server_health(base_url: str, max_attempts: int = 60) -> bool:
    """Check if the server is healthy and model is ready."""
    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                ["curl", "-s", f"{base_url}/tt-liveness"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                response = result.stdout
                if '"status":"alive"' in response and '"model_ready":true' in response:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def print_metrics_summary(metrics: StreamingMetrics, name: str = "Request") -> None:
    """Print a formatted summary of streaming metrics."""
    summary = metrics.summary()
    print(f"\n{'=' * 60}")
    print(f"{name} Metrics Summary")
    print(f"{'=' * 60}")
    print(
        f"Chunks: {summary['received_chunks']}/{summary['expected_chunks'] or 'unknown'}"
    )
    print(f"Chunk Loss Ratio: {summary['chunk_loss_ratio']:.4f}")
    print(
        f"Time to First Chunk: {summary['time_to_first_chunk_ms']:.2f}ms"
        if summary["time_to_first_chunk_ms"]
        else "N/A"
    )
    print(
        f"Total Streaming Time: {summary['total_streaming_time_ms']:.2f}ms"
        if summary["total_streaming_time_ms"]
        else "N/A"
    )
    print(
        f"Expected Send Interval: {summary['expected_send_interval_ms']:.2f}ms"
        if summary["expected_send_interval_ms"]
        else "N/A"
    )
    print(
        f"Mean Receive Interval: {summary['mean_receive_interval_ms']:.2f}ms"
        if summary["mean_receive_interval_ms"]
        else "N/A"
    )
    print(
        f"Latency Ratio (recv/expected): {summary['latency_ratio']:.4f}"
        if summary["latency_ratio"]
        else "N/A"
    )
    print(
        f"Throughput: {summary['throughput_chunks_per_second']:.2f} chunks/s"
        if summary["throughput_chunks_per_second"]
        else "N/A"
    )
    print(f"{'=' * 60}\n")


@pytest.mark.performance
@pytest.mark.usefixtures("server_process")
class TestLLMStreamingPerformance:
    """Performance test suite for LLM streaming endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test configuration and thresholds."""
        self.config = get_server_config()
        self.thresholds = PerformanceThresholds.from_env()


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_performance_full(server_process):
    """Run a comprehensive streaming performance test.

    This standalone test runs all performance checks and provides a detailed report.
    Useful for manual testing and CI integration.
    """
    config = get_server_config()
    thresholds = PerformanceThresholds.from_env()

    print("\n" + "=" * 70)
    print("LLM STREAMING PERFORMANCE TEST")
    print("=" * 70)
    print(f"Server URL: {config.base_url}")
    print(f"Max Tokens: {config.max_tokens}")
    print("Thresholds:")
    print(f"  - Max Chunk Loss Ratio: {thresholds.max_chunk_loss_ratio}")
    print(f"  - Max Latency Ratio: {thresholds.max_latency_ratio}")
    print(f"  - Max Time to First Chunk: {thresholds.max_time_to_first_chunk_ms}ms")
    print("=" * 70)

    client = StreamingClient(config)
    metrics = await client.make_streaming_request()

    print_metrics_summary(metrics, "Performance Test Results")

    # Collect all failures
    failures = []

    if metrics.expected_chunks is not None:
        if metrics.chunk_loss_ratio > thresholds.max_chunk_loss_ratio:
            failures.append(
                f"CHUNK LOSS: {metrics.chunk_loss_ratio:.4f} > "
                f"{thresholds.max_chunk_loss_ratio:.4f}"
            )

    if metrics.latency_ratio is not None:
        if metrics.latency_ratio > thresholds.max_latency_ratio:
            failures.append(
                f"LATENCY RATIO: {metrics.latency_ratio:.4f} > "
                f"{thresholds.max_latency_ratio:.4f}"
            )

    if metrics.time_to_first_chunk_ms is not None:
        if metrics.time_to_first_chunk_ms > thresholds.max_time_to_first_chunk_ms:
            failures.append(
                f"TIME TO FIRST CHUNK: {metrics.time_to_first_chunk_ms:.2f}ms > "
                f"{thresholds.max_time_to_first_chunk_ms:.2f}ms"
            )

    if failures:
        print("\n" + "!" * 70)
        print("PERFORMANCE TEST FAILED")
        print("!" * 70)
        for failure in failures:
            print(f"  ✗ {failure}")
        print("!" * 70 + "\n")
        pytest.fail(f"Performance test failed: {', '.join(failures)}")
    else:
        print("\n" + "=" * 70)
        print("PERFORMANCE TEST PASSED ✓")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Allow running directly for quick testing
    asyncio.run(test_streaming_performance_full())
