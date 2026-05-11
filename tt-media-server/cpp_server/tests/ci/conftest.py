# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Shared fixtures for cpp_server CI integration tests.

Centralizes:
  - the API key (`API_KEY = "your-secret-key"`, reused everywhere)
  - the cpp_server binary location (overridable via TT_CPP_SERVER_BIN)
  - server lifecycle: start a server, wait for /tt-liveness, tear it down
  - vllm bench / guidellm bench wrappers
  - threshold assertions

Tests get a `cpp_server(name, env=..., port=..., ...)` factory plus
`vllm_bench(...)`, `guidellm_bench(...)`, and `assert_thresholds(...)`. All
artifacts (server logs, bench result JSON, bench stdout) land under
`tests/ci/_artifacts/<test-name>/` and are uploaded as a single CI artifact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

# Make this directory importable for sibling modules (`from _server import ...`)
# without requiring an __init__.py — keeps the directory hierarchy flat.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _bench import (  # noqa: E402
    BenchResult,
    GuidellmResult,
    run_guidellm_benchmark,
    run_vllm_bench_serve,
)
from _server import (  # noqa: E402
    API_KEY,
    DEFAULT_HOST,
    DEFAULT_THREADS,
    ServerHandle,
    start_python_runner,
    start_server,
    stop_process,
    stop_server,
    wait_for_ready,
)
from _thresholds import assert_bench_thresholds  # noqa: E402

CPP_SERVER_DIR = _HERE.parents[1]  # tt-media-server/cpp_server
DEFAULT_BINARY = CPP_SERVER_DIR / "build" / "tt_media_server_cpp"
ARTIFACTS_ROOT = _HERE / "_artifacts"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cpp_ci: cpp_server CI integration tests")


@pytest.fixture(scope="session")
def api_key() -> str:
    return API_KEY


@pytest.fixture(scope="session")
def cpp_server_binary() -> Path:
    """Absolute path to the cpp_server binary; skip the test if absent."""
    binary = Path(os.environ.get("TT_CPP_SERVER_BIN") or DEFAULT_BINARY)
    if not binary.exists():
        pytest.skip(
            f"cpp_server binary not found at {binary} "
            "(build it with `cpp_server/build.sh`, or set TT_CPP_SERVER_BIN)"
        )
    if not os.access(binary, os.X_OK):
        pytest.skip(f"cpp_server binary at {binary} is not executable")
    return binary


@pytest.fixture(scope="session")
def cpp_server_dir() -> Path:
    """Absolute path to tt-media-server/cpp_server (default cwd for the binary)."""
    return CPP_SERVER_DIR


def _sanitize_for_path(name: str) -> str:
    """Sanitize a pytest node name for use as a directory."""
    return (
        name.replace("/", "_")
        .replace("[", "_")
        .replace("]", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


@pytest.fixture
def artifacts_dir(request: pytest.FixtureRequest) -> Path:
    """Per-test artifacts directory: tests/ci/_artifacts/<sanitized-test-name>/."""
    path = ARTIFACTS_ROOT / _sanitize_for_path(request.node.name)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def cpp_server(
    cpp_server_binary: Path,
    cpp_server_dir: Path,
    artifacts_dir: Path,
):
    """Factory: start one or more cpp_server instances inside a test.

    Usage:
        handle = cpp_server(
            name="mock",
            env={"LLM_DEVICE_BACKEND": "mock"},
            port=8000,
        )
        # handle.base_url, handle.liveness_url, handle.log_path

    All servers started in the test are torn down automatically. Pass
    `wait=False` to skip the readiness probe (useful when readiness depends
    on a separate handle, e.g. prefill+decode socket pairing).
    """
    handles: list[ServerHandle] = []

    def _start(
        name: str = "server",
        *,
        host: str = DEFAULT_HOST,
        port: int = 8000,
        threads: int = DEFAULT_THREADS,
        env: Optional[Mapping[str, str]] = None,
        extra_args: Optional[list[str]] = None,
        cwd: Optional[Path] = None,
        binary: Optional[Path] = None,
        timeout: float = 30.0,
        require: Optional[Mapping[str, Any]] = None,
        wait: bool = True,
    ) -> ServerHandle:
        log_path = artifacts_dir / f"{name}.log"
        handle = start_server(
            name=name,
            binary=binary or cpp_server_binary,
            log_path=log_path,
            host=host,
            port=port,
            threads=threads,
            extra_args=extra_args,
            env=env,
            cwd=cwd or cpp_server_dir,
        )
        handles.append(handle)
        if wait:
            wait_for_ready(handle, timeout=timeout, require=require)
        return handle

    yield _start

    # Reverse order so dependent processes (e.g. prefill before decode) shut
    # down first.
    for handle in reversed(handles):
        stop_server(handle)


@pytest.fixture
def python_runner(cpp_server_dir: Path, artifacts_dir: Path):
    """Factory for ad-hoc helper python processes (e.g. mock_prefill_runner.py)."""
    processes: list[tuple[str, Any]] = []

    def _start(
        name: str,
        script: Path,
        *,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[Path] = None,
        args: Optional[list[str]] = None,
        settle_sec: float = 2.0,
    ):
        log_path = artifacts_dir / f"{name}.log"
        process = start_python_runner(
            name=name,
            script=script,
            log_path=log_path,
            env=env,
            cwd=cwd or cpp_server_dir,
            args=args,
            settle_sec=settle_sec,
        )
        processes.append((name, process))
        return process

    yield _start

    for name, process in reversed(processes):
        stop_process(process, name=name)


@pytest.fixture
def vllm_bench(api_key: str, artifacts_dir: Path):
    """Factory: run vllm bench serve against a base_url; return BenchResult."""

    def _run(**kwargs) -> BenchResult:
        return run_vllm_bench_serve(
            api_key=api_key,
            artifacts_dir=artifacts_dir,
            **kwargs,
        )

    return _run


@pytest.fixture
def guidellm_bench(api_key: str, artifacts_dir: Path):
    """Factory: run guidellm benchmark run against a target; return GuidellmResult."""

    def _run(**kwargs) -> GuidellmResult:
        return run_guidellm_benchmark(
            api_key=api_key,
            artifacts_dir=artifacts_dir,
            **kwargs,
        )

    return _run


@pytest.fixture
def assert_thresholds():
    """Returns the threshold-checking helper (so tests don't need to import it)."""
    return assert_bench_thresholds
