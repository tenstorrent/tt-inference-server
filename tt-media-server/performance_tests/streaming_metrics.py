# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Streaming performance metrics collection and analysis.

This module provides data classes and utilities for collecting and analyzing
streaming performance metrics, including chunk timing, latency, and throughput.

The metrics compare actual receive timings against expected send frequency
from TestRunner configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkTiming:
    """Timing information for a single streaming chunk."""

    chunk_index: int
    receive_timestamp_ns: int
    text: str = ""


@dataclass
class StreamingMetrics:
    expected_chunks: Optional[int] = None
    expected_send_interval_ms: Optional[float] = None
    chunks: list[ChunkTiming] = field(default_factory=list)
    request_start_ns: int = 0
    first_chunk_ns: Optional[int] = None
    last_chunk_ns: Optional[int] = None

    def add_chunk(self, chunk: ChunkTiming) -> None:
        """Add a chunk timing to the metrics collection."""
        if not self.chunks:
            self.first_chunk_ns = chunk.receive_timestamp_ns
        self.last_chunk_ns = chunk.receive_timestamp_ns
        self.chunks.append(chunk)

    @property
    def received_chunks(self) -> int:
        """Number of chunks received."""
        return len(self.chunks)

    @property
    def chunks_lost(self) -> int:
        """Number of chunks lost (expected - received)."""
        if self.expected_chunks is None:
            return 0
        return max(0, self.expected_chunks - self.received_chunks)

    @property
    def chunk_loss_ratio(self) -> float:
        """Ratio of lost chunks (0.0 = no loss, 1.0 = all lost)."""
        if self.expected_chunks is None or self.expected_chunks == 0:
            return 0.0
        return self.chunks_lost / self.expected_chunks

    @property
    def time_to_first_chunk_ms(self) -> Optional[float]:
        """Time from request start to first chunk in milliseconds."""
        if self.first_chunk_ns is None or self.request_start_ns == 0:
            return None
        return (self.first_chunk_ns - self.request_start_ns) / 1_000_000

    @property
    def total_streaming_time_ms(self) -> Optional[float]:
        """Total time from first to last chunk in milliseconds."""
        if self.first_chunk_ns is None or self.last_chunk_ns is None:
            return None
        return (self.last_chunk_ns - self.first_chunk_ns) / 1_000_000

    def get_receive_intervals_ms(self) -> list[float]:
        """Calculate intervals between consecutive chunk receives in milliseconds."""
        if len(self.chunks) < 2:
            return []
        intervals = []
        sorted_chunks = sorted(self.chunks, key=lambda c: c.chunk_index)
        for i in range(1, len(sorted_chunks)):
            interval_ns = (
                sorted_chunks[i].receive_timestamp_ns
                - sorted_chunks[i - 1].receive_timestamp_ns
            )
            intervals.append(interval_ns / 1_000_000)
        return intervals

    @property
    def mean_receive_interval_ms(self) -> Optional[float]:
        """Mean interval between chunk receives in milliseconds."""
        intervals = self.get_receive_intervals_ms()
        return sum(intervals) / len(intervals) if intervals else None

    @property
    def latency_ratio(self) -> Optional[float]:
        """Ratio of mean receive interval to expected send interval.

        A ratio close to 1.0 indicates minimal system overhead.
        Ratio > 1.0 indicates receive is slower than expected (system bottleneck).

        Uses the configured TEST_RUNNER_FREQUENCY_MS as the expected send interval.
        """
        if (
            self.expected_send_interval_ms is None
            or self.expected_send_interval_ms == 0
        ):
            return None
        receive_interval = self.mean_receive_interval_ms
        if receive_interval is None:
            return None
        return receive_interval / self.expected_send_interval_ms

    @property
    def throughput_chunks_per_second(self) -> Optional[float]:
        """Throughput in chunks per second."""
        streaming_time = self.total_streaming_time_ms
        if streaming_time is None or streaming_time == 0:
            return None
        return (self.received_chunks - 1) / (streaming_time / 1000)

    def summary(self) -> dict:
        """Return a summary dictionary of all metrics."""
        return {
            "expected_chunks": self.expected_chunks,
            "received_chunks": self.received_chunks,
            "chunks_lost": self.chunks_lost,
            "chunk_loss_ratio": self.chunk_loss_ratio,
            "time_to_first_chunk_ms": self.time_to_first_chunk_ms,
            "total_streaming_time_ms": self.total_streaming_time_ms,
            "mean_receive_interval_ms": self.mean_receive_interval_ms,
            "expected_send_interval_ms": self.expected_send_interval_ms,
            "latency_ratio": self.latency_ratio,
            "throughput_chunks_per_second": self.throughput_chunks_per_second,
        }
