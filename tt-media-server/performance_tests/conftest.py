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
    """Wait for server to be ready by polling with a streaming request.

    The server might respond quickly but the model takes time to load.
    We use a streaming request to ensure the model is fully ready.
    """
    poll_interval = 1.0
    max_attempts = int(timeout_seconds / poll_interval)

    print(f"  Waiting for server to be ready (max {timeout_seconds}s)...")

    for attempt in range(max_attempts):
        try:
            # Try a streaming request - this will fail if model isn't ready
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-X",
                    "POST",
                    f"{url}/v1/completions",
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    "Authorization: Bearer your-secret-key",
                    "-d",
                    '{"model":"test","prompt":"test","stream":true,"max_tokens":5}',
                    "--max-time",
                    "10",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            # Check if we got actual token output - this means model is ready
            # Note: rc=28 is curl timeout, but if we got tokens, server is working
            if "token_" in result.stdout:
                print(f"  ✓ Server ready after {attempt + 1}s")
                return True
            else:
                # Show why we're waiting
                status = f"rc={result.returncode}"
                if result.stderr:
                    status += f", err={result.stderr[:50]}"
                elif result.stdout:
                    # Truncate output for readability
                    out = result.stdout[:80].replace("\n", " ")
                    status += f", out={out}"
                print(f"  [{attempt + 1}/{max_attempts}] Waiting... ({status})")
        except Exception as e:
            print(
                f"  [{attempt + 1}/{max_attempts}] Server not responding: {type(e).__name__}"
            )
        time.sleep(poll_interval)

    print(f"  ✗ Server not ready after {timeout_seconds}s")
    return False


@pytest.fixture(scope="session")
def server_process():
    """Start the server process for performance tests.

    This fixture starts the server in a subprocess with TestRunner configuration,
    waits for it to be ready, and shuts it down after all tests complete.

    Set PERF_USE_EXTERNAL_SERVER=true to skip auto-starting and use an external server.
    """
    use_external = os.getenv("PERF_USE_EXTERNAL_SERVER", "false").lower() == "true"

    if use_external:
        # Use external server - just verify it's running
        external_url = os.getenv("TEST_SERVER_URL", SERVER_URL)
        print(f"\nUsing external server at {external_url}")
        if not _wait_for_server_ready(external_url, timeout_seconds=10):
            pytest.skip(f"External server at {external_url} is not responding")
        yield None
        return

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
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )

    # Wait for server to be ready
    if not _wait_for_server_ready(SERVER_URL, timeout_seconds=30):
        # Server didn't start - get output for debugging
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
            print(f"Server failed to start. Output:\n{output.decode()}")
        except Exception:
            process.kill()
        pytest.fail("Server failed to start within 30 seconds")

    print(f"Server ready at {SERVER_URL}")

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


@pytest.fixture(scope="session")
def server_url(server_process):
    """Get the server URL, ensuring server is running."""
    if os.getenv("PERF_USE_EXTERNAL_SERVER", "false").lower() == "true":
        return os.getenv("TEST_SERVER_URL", SERVER_URL)
    return SERVER_URL


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
        "max_latency_ratio": float(os.getenv("PERF_MAX_LATENCY_RATIO", "1.05")),
        "max_ttfc_ms": float(os.getenv("PERF_MAX_TTFC_MS", "1000.0")),
    }
