# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

SERVER_STARTUP_TIMEOUT = 90
SERVER_SHUTDOWN_TIMEOUT = 20
SERVER_DEFAULT_URL = "http://127.0.0.1:8000/cnn/search-image"
DEFAULT_AUTHORIZATION = "your-secret-key"
TT_MEDIA_SERVER_DIR = Path(__file__).resolve().parents[3] / "tt-media-server"
READY_LOG_TEXT = "All devices are warmed up and ready"
LOG_DIR = Path(__file__).resolve().parent / "server_logs"


def launch_cpu_server(
    model_runner: str,
    port: int = 8000,
    extra_env: Optional[Dict[str, str]] = None,
) -> tuple[subprocess.Popen, Path]:
    """Start the media server in CPU mode for the given model and capture logs."""

    if not TT_MEDIA_SERVER_DIR.exists():
        raise FileNotFoundError(f"tt-media-server directory not found at {TT_MEDIA_SERVER_DIR}")

    env = os.environ.copy()
    env["MODEL_RUNNER"] = model_runner
    env["RUNS_ON_CPU"] = "true"
    env["DEVICE_IDS"] = "0"
    env["IS_GALAXY"] = "false"
    if extra_env:
        env.update(extra_env)

    command = [
        "uvicorn",
        "main:app",
        "--lifespan",
        "on",
        "--port",
        str(port),
    ]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_model_name(model_runner)
    log_path = LOG_DIR / f"{safe_name}.log"
    log_handle = open(log_path, "w", encoding="utf-8", buffering=1)

    process = subprocess.Popen(
        command,
        cwd=str(TT_MEDIA_SERVER_DIR),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    process._log_handle = log_handle  # type: ignore[attr-defined]
    process._log_path = log_path  # type: ignore[attr-defined]
    return process, log_path


def wait_for_server_ready(
    process: subprocess.Popen,
    timeout: float = SERVER_STARTUP_TIMEOUT,
    interval: float = 1.0,
    log_path: Optional[Path] = None,
) -> None:
    """Stream server logs until the ready marker is observed."""

    ready_path = log_path or getattr(process, "_log_path", None)
    if ready_path is None:
        raise ValueError("Missing log path for readiness checks.")

    deadline = time.time() + timeout
    last_pos = 0
    buffer = ""
    max_buffer = max(len(READY_LOG_TEXT) * 2, 2048)

    with ready_path.open("r", encoding="utf-8", errors="replace") as log_file:
        while time.time() < deadline:
            if process.poll() is not None:
                log_file.seek(0)
                snippet = log_file.read()
                raise RuntimeError(
                    "Server process exited before becoming ready. Last log output:\n"
                    + snippet[-2000:]
                )

            log_file.seek(last_pos)
            chunk = log_file.read()
            if chunk:
                last_pos = log_file.tell()
                buffer = (buffer + chunk)[-max_buffer:]
                if READY_LOG_TEXT in buffer:
                    return

            time.sleep(interval)

    raise TimeoutError(
        f"Server did not log '{READY_LOG_TEXT}' within {timeout} seconds."
    )


def stop_server(
    process: subprocess.Popen,
    timeout: float = SERVER_SHUTDOWN_TIMEOUT,
) -> None:
    """Terminate the spawned server process and wait for a clean shutdown."""

    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
    finally:
        log_handle = getattr(process, "_log_handle", None)
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()


def sanitize_model_name(name: str) -> str:
    """Produce a filesystem-safe fragment from a model runner name."""

    return name.replace("/", "_").replace(" ", "_")
