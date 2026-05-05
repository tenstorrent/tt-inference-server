# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Pytest fixtures for the runner-event integration tests.

A session-scoped fixture boots a single-worker mock server with the
in-memory event recorder enabled (`TT_RUNNER_RECORDER_ENABLED=1`); a
function-scoped fixture clears the event buffer before each test so
assertions only see events produced by that test.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from typing import Any

import pytest
import requests


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
SERVER_BIN = PROJECT_DIR / "build" / "tt_media_server_cpp"
DEFAULT_API_KEY = "your-secret-key"
READY_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 10


class RunnerEventClient:
    """Thin HTTP wrapper around `/v1/...` and `/debug/runner-events`."""

    def __init__(self, base_url: str, api_key: str = DEFAULT_API_KEY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def clear_events(self) -> int:
        resp = requests.delete(
            f"{self.base_url}/debug/runner-events", timeout=5
        )
        resp.raise_for_status()
        return resp.json()["last_seq"]

    def get_events(self, since_seq: int = 0) -> list[dict[str, Any]]:
        url = f"{self.base_url}/debug/runner-events"
        if since_seq:
            url += f"?since_seq={since_seq}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()["events"]

    def chat(self, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            timeout=60,
        )

    def stream_chat(self, payload: dict[str, Any]) -> int:
        chunks = 0
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._auth_headers(),
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                if raw[len("data: "):].strip() == "[DONE]":
                    break
                chunks += 1
        return chunks


def _wait_for_ready(base_url: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Server exited prematurely with code {proc.returncode}; "
                "see build/recorder_logs/server.log"
            )
        try:
            resp = requests.get(f"{base_url}/tt-liveness", timeout=2)
            if resp.ok and resp.json().get("model_ready") is True:
                return
        except (requests.ConnectionError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError(
        f"Server not ready at {base_url} after {READY_TIMEOUT_S}s; "
        "see build/recorder_logs/server.log"
    )


@pytest.fixture(scope="session")
def server_url() -> str:
    """Boot a single-worker mock server with the recorder enabled.

    Honors `RECORDER_PORT` env var (default 8099) so the tests can be
    pointed at a custom port if 8099 is taken. The server log is written
    to `build/recorder_logs/server.log` for post-mortem debugging.
    """
    if not SERVER_BIN.is_file() or not os.access(SERVER_BIN, os.X_OK):
        pytest.skip(f"Server binary not found at {SERVER_BIN}; run ./build.sh")

    port = int(os.environ.get("RECORDER_PORT", "8099"))
    base_url = f"http://127.0.0.1:{port}"

    log_dir = PROJECT_DIR / "build" / "recorder_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "server.log"

    env = os.environ.copy()
    env["TT_RUNNER_RECORDER_ENABLED"] = "1"
    env["LLM_DEVICE_BACKEND"] = "mock"
    env["MODEL_SERVICE"] = "llm"
    env["NUM_WORKERS"] = "1"

    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        [str(SERVER_BIN), "-p", str(port), "-h", "127.0.0.1", "-t", "4"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR),
    )

    try:
        _wait_for_ready(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()


@pytest.fixture
def client(server_url: str) -> RunnerEventClient:
    """A fresh HTTP client with the event buffer cleared.

    Each test gets its own slice of the event log, so assertions never
    see leftovers from a prior test.
    """
    c = RunnerEventClient(server_url)
    c.clear_events()
    return c
