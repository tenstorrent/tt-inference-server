# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import sys
import time

import pytest
import requests

DEVICE_IDS = "(0)"
IS_GALAXY = "False"
SERVER_BASE_URL = "http://127.0.0.1:8000"
SERVER_READY_TIMEOUT_SEC = 30
SERVER_READY_POLL_INTERVAL_SEC = 0.5
TEST_RUNNER_FREQUENCY_MS = 20


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests",
    )


def wait_for_server_ready(
    base_url: str = SERVER_BASE_URL,
    timeout: int = SERVER_READY_TIMEOUT_SEC,
    poll_interval: float = SERVER_READY_POLL_INTERVAL_SEC,
) -> bool:
    """Poll the tt-liveness endpoint until the server is ready or timeout is reached."""
    start_time = time.time()
    liveness_url = f"{base_url}/tt-liveness"

    while time.time() - start_time < timeout:
        try:
            response = requests.get(liveness_url, timeout=2)
            if response.status_code == 200 and response.json().get("status") == "alive":
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(poll_interval)

    return False


@pytest.fixture(scope="session")
def server_process():
    """Start the server process for performance tests.

    This fixture starts the server in a subprocess with LLMTestRunner configuration,
    waits for it to be ready, and shuts it down after all tests complete.
    Server stdout/stderr is logged to performance_tests/server.log.
    """

    # Build environment for server process
    env = os.environ.copy()
    env["MODEL_RUNNER"] = "llm_test"
    env["TEST_RUNNER_FREQUENCY_MS"] = str(TEST_RUNNER_FREQUENCY_MS)
    env["DEVICE_IDS"] = str(DEVICE_IDS)
    env["IS_GALAXY"] = str(IS_GALAXY)

    # Get the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(os.path.dirname(__file__), "server.log")

    print("\nStarting server on http://localhost:8000...")
    print(f"Server logs: {log_file_path}")

    # Open log file for server output
    log_file = open(log_file_path, "w")

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
        stdout=log_file,
        stderr=subprocess.STDOUT,  # Redirect stderr to stdout (the log file)
    )

    # Wait for server to be ready by polling the liveness endpoint
    print(f"Waiting for server to be ready (timeout: {SERVER_READY_TIMEOUT_SEC}s)...")
    if not wait_for_server_ready():
        process.terminate()
        process.wait(timeout=5)
        log_file.close()
        raise RuntimeError(
            f"Server failed to become ready within {SERVER_READY_TIMEOUT_SEC} seconds"
        )
    print("Server is ready!")

    yield process

    # Cleanup: stop the server
    print("\nStopping server...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    log_file.close()
    print("Server stopped")
