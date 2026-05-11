# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Server lifecycle helpers for cpp_server CI integration tests.

Centralizes the start-and-wait pattern that used to be inlined repeatedly in
.github/workflows/test-gate.yml — start the binary in the background, redirect
stdout/stderr to a log file, poll /tt-liveness until model_ready, terminate on
teardown.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import requests

API_KEY = "your-secret-key"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_THREADS = 4
DEFAULT_READY_TIMEOUT_SEC = 30.0
DEFAULT_READY_POLL_SEC = 1.0
DEFAULT_LIVENESS_PATH = "/tt-liveness"
DEFAULT_STOP_TIMEOUT_SEC = 5.0


@dataclass
class ServerHandle:
    """Reference to a running tt_media_server_cpp process."""

    name: str
    host: str
    port: int
    process: subprocess.Popen
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def liveness_url(self) -> str:
        return f"{self.base_url}{DEFAULT_LIVENESS_PATH}"

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.poll() is None


def liveness(handle: ServerHandle, *, timeout: float = 2.0) -> Optional[dict]:
    """Fetch /tt-liveness; return parsed JSON or None on connection/HTTP error."""
    try:
        response = requests.get(handle.liveness_url, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def wait_for_ready(
    handle: ServerHandle,
    *,
    timeout: float = DEFAULT_READY_TIMEOUT_SEC,
    poll_interval: float = DEFAULT_READY_POLL_SEC,
    require: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Poll /tt-liveness until required fields match or timeout.

    `require` defaults to {"model_ready": True}; pass a dict to add additional
    fields (e.g. {"socket_status": "client:connected"}).
    """
    required: dict[str, Any] = {"model_ready": True}
    if require is not None:
        required.update(require)

    deadline = time.monotonic() + timeout
    last_payload: Optional[dict] = None
    while time.monotonic() < deadline:
        if not handle.is_alive():
            raise RuntimeError(
                f"[{handle.name}] server exited with code {handle.process.returncode} "
                f"during startup (log: {handle.log_path})"
            )
        payload = liveness(handle)
        if payload is not None:
            last_payload = payload
            if all(payload.get(k) == v for k, v in required.items()):
                return payload
        time.sleep(poll_interval)

    raise TimeoutError(
        f"[{handle.name}] not ready within {timeout:.1f}s "
        f"(required={required}, last={last_payload}, log={handle.log_path})"
    )


def start_server(
    *,
    name: str,
    binary: Path,
    log_path: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    threads: int = DEFAULT_THREADS,
    extra_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> ServerHandle:
    """Spawn tt_media_server_cpp; redirect output to log_path; return handle.

    The process is launched in its own session so SIGTERM in stop_server can be
    delivered to the whole process group (covers any threads/children spawned
    by the server, including the python prefill runner).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = [str(binary), "-p", str(port), "-h", host, "-t", str(threads)]
    if extra_args:
        args.extend(extra_args)

    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})

    log_file = open(log_path, "wb")
    process = subprocess.Popen(
        args,
        cwd=str(cwd) if cwd else None,
        env=proc_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ServerHandle(
        name=name,
        host=host,
        port=port,
        process=process,
        log_path=log_path,
    )


def start_python_runner(
    *,
    name: str,
    script: Path,
    log_path: Path,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
    args: Optional[Sequence[str]] = None,
    settle_sec: float = 2.0,
) -> subprocess.Popen:
    """Spawn a Python helper process (e.g. mock_prefill_runner.py) and return it.

    Used by the prefill/decode-split round 2 setup, where rank-0 coordinator runs
    as a Python script. After `settle_sec`, raise if the process has already
    exited (catches immediate import/argument errors).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(script)]
    if args:
        cmd.extend(str(a) for a in args)

    proc_env = os.environ.copy()
    if env:
        proc_env.update({k: str(v) for k, v in env.items()})

    log_file = open(log_path, "wb")
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=proc_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    if settle_sec > 0:
        time.sleep(settle_sec)
        if process.poll() is not None:
            raise RuntimeError(
                f"[{name}] python runner exited with code {process.returncode} "
                f"during startup (log: {log_path})"
            )
    return process


def stop_process(
    process: subprocess.Popen,
    *,
    name: str = "process",
    timeout: float = DEFAULT_STOP_TIMEOUT_SEC,
) -> None:
    """Send SIGTERM to the process group; SIGKILL if it doesn't exit in time."""
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Last resort — leave it; the harness exit will reap it.
        pass


def stop_server(
    handle: ServerHandle, *, timeout: float = DEFAULT_STOP_TIMEOUT_SEC
) -> None:
    stop_process(handle.process, name=handle.name, timeout=timeout)
