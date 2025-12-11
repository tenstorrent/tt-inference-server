# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Pytest configuration for performance tests.

This module provides pytest fixtures and configuration for performance testing.
Performance tests require a running server and are marked with @pytest.mark.performance.

Usage:
    # Run only performance tests
    pytest performance_tests/ -v -m performance

    # Skip performance tests in regular test runs
    pytest tests/ -v -m "not performance"
"""

import os
import subprocess
import time

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests (requires running server)",
    )


@pytest.fixture(scope="session")
def server_url():
    """Get the server URL from environment or default."""
    return os.getenv("TEST_SERVER_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def wait_for_server(server_url):
    """Wait for the server to be ready before running tests.

    This fixture blocks until the server is healthy or timeout is reached.
    """
    max_wait_seconds = int(os.getenv("PERF_SERVER_WAIT_SECONDS", "60"))
    poll_interval = 2

    for _ in range(max_wait_seconds // poll_interval):
        try:
            result = subprocess.run(
                ["curl", "-s", f"{server_url}/tt-liveness"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                response = result.stdout
                if '"status":"alive"' in response and '"model_ready":true' in response:
                    print(f"\nServer at {server_url} is ready")
                    return True
        except Exception:
            pass
        time.sleep(poll_interval)

    pytest.skip(f"Server at {server_url} not ready after {max_wait_seconds}s")
    return False


@pytest.fixture(scope="session")
def test_runner_config():
    """Get TestRunner configuration from environment."""
    return {
        "warmup_ms": int(os.getenv("TEST_RUNNER_WARMUP_MS", "100")),
        "frequency_ms": int(os.getenv("TEST_RUNNER_FREQUENCY_MS", "50")),
        "total_tokens": int(os.getenv("TEST_RUNNER_TOTAL_TOKENS", "100")),
    }


@pytest.fixture(scope="session")
def performance_thresholds():
    """Get performance thresholds from environment."""
    return {
        "max_chunk_loss_ratio": float(os.getenv("PERF_MAX_CHUNK_LOSS_RATIO", "0.0")),
        "max_latency_ratio": float(os.getenv("PERF_MAX_LATENCY_RATIO", "1.5")),
        "max_ttfc_ms": float(os.getenv("PERF_MAX_TTFC_MS", "5000.0")),
    }
