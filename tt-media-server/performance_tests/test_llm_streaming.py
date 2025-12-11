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
    run_concurrent_requests,
)
from performance_tests.streaming_metrics import StreamingMetrics


@dataclass
class PerformanceThresholds:
    """Performance thresholds loaded from environment variables."""

    max_chunk_loss_ratio: float = 0.0  # No loss allowed by default
    max_latency_ratio: float = 1.5  # Allow 50% overhead by default
    max_time_to_first_chunk_ms: float = 5000.0  # 5 seconds max TTFC

    @classmethod
    def from_env(cls) -> "PerformanceThresholds":
        return cls(
            max_chunk_loss_ratio=float(os.getenv("PERF_MAX_CHUNK_LOSS_RATIO", "0.0")),
            max_latency_ratio=float(os.getenv("PERF_MAX_LATENCY_RATIO", "1.5")),
            max_time_to_first_chunk_ms=float(os.getenv("PERF_MAX_TTFC_MS", "5000.0")),
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
class TestLLMStreamingPerformance:
    """Performance test suite for LLM streaming endpoint."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test configuration and thresholds."""
        self.config = get_server_config()
        self.thresholds = PerformanceThresholds.from_env()

    @pytest.mark.asyncio
    async def test_no_chunk_loss(self):
        """Test that all chunks sent by TestRunner are received by the client.

        This test verifies that the streaming infrastructure does not drop any
        chunks during transmission from the TestRunner through the device worker,
        scheduler, and HTTP streaming response.
        """
        client = StreamingClient(self.config)
        metrics = await client.make_streaming_request()

        print_metrics_summary(metrics, "Chunk Loss Test")

        # Verify expected chunks is configured
        assert metrics.expected_chunks is not None, (
            "Expected chunks not configured. Set TEST_RUNNER_TOTAL_TOKENS."
        )

        # Check chunk loss
        assert metrics.chunk_loss_ratio <= self.thresholds.max_chunk_loss_ratio, (
            f"Chunk loss ratio {metrics.chunk_loss_ratio:.4f} exceeds threshold "
            f"{self.thresholds.max_chunk_loss_ratio:.4f}. "
            f"Received {metrics.received_chunks}/{metrics.expected_chunks} chunks."
        )

    @pytest.mark.asyncio
    async def test_latency_ratio(self):
        """Test that the latency ratio stays within acceptable bounds.

        The latency ratio is the ratio of mean receive interval to expected send interval.
        A ratio close to 1.0 indicates minimal system overhead.
        A ratio > threshold indicates performance degradation.
        """
        client = StreamingClient(self.config)
        metrics = await client.make_streaming_request()

        print_metrics_summary(metrics, "Latency Ratio Test")

        # Verify we have configuration
        assert metrics.expected_send_interval_ms is not None, (
            "Expected send interval not configured. Set TEST_RUNNER_FREQUENCY_MS."
        )
        assert metrics.mean_receive_interval_ms is not None, (
            "Could not calculate receive intervals. Need at least 2 chunks."
        )

        latency_ratio = metrics.latency_ratio
        assert latency_ratio is not None, "Could not calculate latency ratio."

        assert latency_ratio <= self.thresholds.max_latency_ratio, (
            f"Latency ratio {latency_ratio:.4f} exceeds threshold "
            f"{self.thresholds.max_latency_ratio:.4f}. "
            f"Expected send interval: {metrics.expected_send_interval_ms:.2f}ms, "
            f"Mean receive interval: {metrics.mean_receive_interval_ms:.2f}ms."
        )

    @pytest.mark.asyncio
    async def test_time_to_first_chunk(self):
        """Test that time to first chunk is within acceptable bounds.

        This measures the end-to-end latency from request initiation to
        receiving the first streaming chunk.
        """
        client = StreamingClient(self.config)
        metrics = await client.make_streaming_request()

        print_metrics_summary(metrics, "Time to First Chunk Test")

        ttfc = metrics.time_to_first_chunk_ms
        assert ttfc is not None, "Could not measure time to first chunk."

        assert ttfc <= self.thresholds.max_time_to_first_chunk_ms, (
            f"Time to first chunk {ttfc:.2f}ms exceeds threshold "
            f"{self.thresholds.max_time_to_first_chunk_ms:.2f}ms."
        )

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test performance under concurrent load.

        Verifies that the system maintains performance with multiple
        simultaneous streaming requests.
        """
        num_concurrent = int(os.getenv("PERF_CONCURRENT_REQUESTS", "2"))

        results = await run_concurrent_requests(self.config, num_concurrent)

        # Filter out exceptions
        successful_results = [r for r in results if isinstance(r, StreamingMetrics)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        print(f"\n{'=' * 60}")
        print(f"Concurrent Requests Test ({num_concurrent} requests)")
        print(f"{'=' * 60}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")

        for i, result in enumerate(successful_results):
            print_metrics_summary(result, f"Request {i + 1}")

        # All requests should succeed
        assert len(failed_results) == 0, (
            f"{len(failed_results)} requests failed: {[str(e) for e in failed_results]}"
        )

        # All successful requests should meet thresholds
        for i, metrics in enumerate(successful_results):
            if metrics.expected_chunks is not None:
                assert (
                    metrics.chunk_loss_ratio <= self.thresholds.max_chunk_loss_ratio
                ), (
                    f"Request {i + 1}: Chunk loss ratio {metrics.chunk_loss_ratio:.4f} "
                    f"exceeds threshold {self.thresholds.max_chunk_loss_ratio:.4f}."
                )

            if metrics.latency_ratio is not None:
                assert metrics.latency_ratio <= self.thresholds.max_latency_ratio * 2, (
                    f"Request {i + 1}: Latency ratio {metrics.latency_ratio:.4f} "
                    f"exceeds concurrent threshold {self.thresholds.max_latency_ratio * 2:.4f}."
                )


@pytest.mark.performance
@pytest.mark.asyncio
async def test_streaming_performance_full():
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
