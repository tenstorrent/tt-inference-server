# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Performance testing framework for TT Inference Server.

This package provides tools for measuring and validating the performance
of the inference server's streaming infrastructure.

Modules:
    streaming_metrics: Data classes for collecting and analyzing streaming metrics
    llm_streaming_client: HTTP client for making streaming requests with timing
    test_llm_streaming: Pytest-based performance tests for LLM streaming

Usage:
    # Run all performance tests
    pytest performance_tests/ -v

    # Run with custom configuration
    export TEST_RUNNER_FREQUENCY_MS=50
    export TEST_RUNNER_TOTAL_TOKENS=100
    export PERF_MAX_LATENCY_RATIO=1.2
    pytest performance_tests/test_llm_streaming.py -v
"""

from performance_tests.llm_streaming_client import (
    LLMStreamingClient,
)
from performance_tests.streaming_metrics import (
    StreamingMetrics,
    TokenTimeSample,
)

__all__ = [
    "TokenTimeSample",
    "StreamingMetrics",
    "LLMStreamingClient",
]
