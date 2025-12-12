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
    LLMStreamingRequestConfig,
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


def get_server_config() -> LLMStreamingRequestConfig:
    """Get server configuration from environment."""
    return LLMStreamingRequestConfig.from_env()


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

    client = LLMStreamingClient(config)
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
