# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import sys
import time

import pytest

from performance_tests.test_llm_streaming import TEST_RUNNER_FREQUENCY_MS


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests",
    )


@pytest.fixture(scope="session")
def server_process():
    """Start the server process for performance tests.

    This fixture starts the server in a subprocess with TestRunner configuration,
    waits for it to be ready, and shuts it down after all tests complete.
    """

    # Build environment for server process
    env = os.environ.copy()
    env["MODEL_RUNNER"] = "test"
    env["TEST_RUNNER_FREQUENCY_MS"] = str(TEST_RUNNER_FREQUENCY_MS)

    # Get the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\nStarting server on http://localhost:8000...")

    # Start server process
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--lifespan",
            "on",
        ],
        cwd=project_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    print("Waiting 5 seconds for server to be ready...")
    time.sleep(5)
    print("Assuming server is ready!")

    yield process

    # Cleanup: stop the server
    print("\nStopping server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    print("Server stopped")
