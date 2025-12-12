# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Streaming performance metrics collection and analysis.

This module provides data classes and utilities for collecting and analyzing
streaming performance metrics, including chunk timing, latency, and throughput.

The metrics compare actual receive timings against expected send frequency
from TestRunner configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenTimeSample:
    """Timing information for a single streaming token."""

    token_index: int
    receive_timestamp_ns: int


@dataclass
class StreamingMetrics:
    def __init__(self, samples: list[TokenTimeSample], request_start_ns: int):
        self.samples = samples
        self.request_start_ns = request_start_ns

    @property
    def received_token_count(self) -> int:
        """Number of tokens received."""
        return len(self.samples)

    @property
    def total_streaming_time_ms(self) -> Optional[float]:
        """Total time from first to last token in milliseconds."""
        return (
            self.samples[-1].receive_timestamp_ns - self.request_start_ns
        ) / 1_000_000

    def get_receive_intervals_ms(self) -> list[float]:
        """Calculate intervals between consecutive token receives in milliseconds."""
        if len(self.samples) < 2:
            return []
        intervals = []
        sorted_samples = sorted(self.samples, key=lambda s: s.token_index)
        for i in range(1, len(sorted_samples)):
            interval_ns = (
                sorted_samples[i].receive_timestamp_ns
                - sorted_samples[i - 1].receive_timestamp_ns
            )
            intervals.append(interval_ns / 1_000_000)
        return intervals

    @property
    def mean_receive_interval_ms(self) -> Optional[float]:
        """Mean interval between token receives in milliseconds."""
        intervals = self.get_receive_intervals_ms()
        return sum(intervals) / len(intervals) if intervals else None

    @property
    def throughput_tokens_per_second(self) -> Optional[float]:
        """Throughput in tokens per second."""
        streaming_time = self.total_streaming_time_ms
        if streaming_time is None or streaming_time == 0:
            return None
        return (self.received_token_count - 1) / (streaming_time / 1000)

    def __str__(self) -> str:
        return f"StreamingMetrics(received_chunks={self.received_token_count}, total_streaming_time_ms={self.total_streaming_time_ms}, mean_receive_interval_ms={self.mean_receive_interval_ms}, throughput_tokens_per_second={self.throughput_tokens_per_second})"
