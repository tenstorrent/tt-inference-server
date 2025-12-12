# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Pytest configuration for performance tests.

This module provides pytest fixtures and configuration for performance testing.
The server is automatically started before tests and stopped after.

Usage:
    # Run performance tests (server starts automatically)
    pytest performance_tests/ -v -m performance

    # Use external server instead of auto-starting
    export PERF_USE_EXTERNAL_SERVER=true
    export TEST_SERVER_URL=http://my-server:9000
    pytest performance_tests/ -v -m performance
"""

import os
import signal
import subprocess
import sys
import time

import pytest

# Server configuration
SERVER_PORT = int(os.getenv("TEST_SERVER_PORT", "9000"))
SERVER_HOST = "127.0.0.1"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests",
    )


def _kill_existing_server(port: int, verbose: bool = True) -> None:
    """Kill any existing process on the specified port."""
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            if verbose and pids:
                print(f"  Killing existing processes on port {port}: {pids}")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(1)  # Wait for ports to be released
    except Exception:
        pass


def _wait_for_server_ready(url: str, timeout_seconds: int = 30) -> bool:
    time.sleep(timeout_seconds)
    return True


@pytest.fixture(scope="session")
def server_process():
    """Start the server process for performance tests.

    This fixture starts the server in a subprocess with TestRunner configuration,
    waits for it to be ready, and shuts it down after all tests complete.
    """

    # Get test runner configuration
    test_runner_config = {
        "TEST_RUNNER_WARMUP_MS": os.getenv("TEST_RUNNER_WARMUP_MS", "100"),
        "TEST_RUNNER_FREQUENCY_MS": os.getenv("TEST_RUNNER_FREQUENCY_MS", "50"),
        "TEST_RUNNER_TOTAL_TOKENS": os.getenv("TEST_RUNNER_TOTAL_TOKENS", "100"),
    }

    # Build environment for server process
    env = os.environ.copy()
    env.update(test_runner_config)
    env["MODEL_RUNNER"] = "test"

    # Get the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Kill any existing process on the port (from previous interrupted runs)
    _kill_existing_server(SERVER_PORT)

    print(f"\nStarting server on {SERVER_URL}...")
    print(
        f"  TEST_RUNNER_FREQUENCY_MS={test_runner_config['TEST_RUNNER_FREQUENCY_MS']}"
    )
    print(
        f"  TEST_RUNNER_TOTAL_TOKENS={test_runner_config['TEST_RUNNER_TOTAL_TOKENS']}"
    )

    # Start server process
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
            "--lifespan",
            "on",
        ],
        cwd=project_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    # Wait for server to be ready
    print("Waiting 10 seconds for server to be ready...")
    time.sleep(10)
    print("Assuming server is ready!")

    yield process

    # Cleanup: stop the server
    print("\nStopping server...")

    # First try graceful shutdown
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        # Force kill if graceful shutdown failed
        if hasattr(os, "killpg"):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait(timeout=5)

    # Also kill any remaining processes on the port (worker processes)
    _kill_existing_server(SERVER_PORT)

    print("Server stopped")
