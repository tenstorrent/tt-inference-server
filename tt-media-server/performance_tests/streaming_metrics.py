# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

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
        return len(self.samples)

    @property
    def total_streaming_time_ms(self) -> Optional[float]:
        return (
            self.samples[-1].receive_timestamp_ns - self.request_start_ns
        ) / 1_000_000

    def get_receive_intervals_ms(self) -> list[float]:
        if len(self.samples) < 2:
            return []
        sorted_samples = sorted(self.samples, key=lambda s: s.token_index)
        return [
            (
                sorted_samples[i].receive_timestamp_ns
                - sorted_samples[i - 1].receive_timestamp_ns
            )
            / 1_000_000
            for i in range(1, len(sorted_samples))
        ]

    @property
    def mean_receive_interval_ms(self) -> Optional[float]:
        intervals = self.get_receive_intervals_ms()
        return sum(intervals) / len(intervals) if intervals else None

    def calculate_overhead_ms(self, test_runner_frequency_ms: int) -> float:
        return self.mean_receive_interval_ms - test_runner_frequency_ms

    @property
    def throughput_tokens_per_second(self) -> Optional[float]:
        streaming_time = self.total_streaming_time_ms
        if streaming_time is None or streaming_time == 0:
            return None
        return (self.received_token_count - 1) / (streaming_time / 1000)

    def __str__(self) -> str:
        return f"""StreamingMetrics(
            received_tokens={self.received_token_count},
            total_streaming_time_ms={self.total_streaming_time_ms:.2f},
            mean_receive_interval_ms={self.mean_receive_interval_ms:.2f},
            throughput_tokens_per_second={self.throughput_tokens_per_second:.2f},
        )"""
