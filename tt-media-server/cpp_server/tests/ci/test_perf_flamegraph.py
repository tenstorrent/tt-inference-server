# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Capture on-CPU and off-CPU flamegraphs for the cpp_server under load.

Produces four SVGs per run (main+worker × on-CPU+off-CPU), saved under
`tests/ci/_artifacts/test_perf_flamegraph/`. CI uploads that directory as an
artifact so every PR has a before/after view of where the server spends and
loses CPU.

Workload: `LLM_DEVICE_BACKEND=mock_pipeline` (requires `./build.sh --blaze`)
with a fixed concurrency × duration. The point is reproducibility — the
absolute numbers move with hardware, but the SHAPE of the flamegraph stays
comparable across PRs on the same runner.

The pytest test drives the same `flamegraph-capture.sh` and
`flamegraph-capture-offcpu.sh` scripts that developers run locally — no
separate code path to maintain.

Requirements:
  - cpp_server built with `--blaze` (binary at build/tt_media_server_cpp)
  - `perf` installed (linux-tools-$(uname -r))
  - sudo available (perf inside containers / restricted hosts needs root)
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

CPP_SERVER_DIR = Path(__file__).resolve().parents[2]
BINARY = CPP_SERVER_DIR / "build" / "tt_media_server_cpp"
CAPTURE_ON = CPP_SERVER_DIR / "flamegraph-capture.sh"
CAPTURE_OFF = CPP_SERVER_DIR / "flamegraph-capture-offcpu.sh"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "_artifacts" / "test_perf_flamegraph"

API_KEY = "your-secret-key"
PORT = 8000
HOST = "127.0.0.1"
BASE_URL = f"http://{HOST}:{PORT}"

# Workload knobs — keep small enough to fit a CI minute budget but large
# enough that the flamegraph is statistically meaningful.
CONCURRENCY = 16
CAPTURE_SECONDS = 20
LOAD_RAMP_SECONDS = 3
READY_TIMEOUT_SECONDS = 30


def _skip_if_missing(path: Path, what: str) -> None:
    if not path.exists():
        pytest.skip(f"{what} not found at {path} — build with `./build.sh --blaze`")


def _skip_if_no_perf() -> None:
    if shutil.which("perf") is None:
        pytest.skip("perf not installed (apt install linux-tools-$(uname -r))")
    # sudo perf works on most CI runners; if it doesn't, skip rather than fail.
    try:
        subprocess.run(
            ["sudo", "-n", "perf", "stat", "-e", "cycles", "--", "true"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("sudo perf not usable on this host")


@pytest.fixture(scope="module")
def server():
    _skip_if_missing(BINARY, "cpp_server binary")
    _skip_if_missing(CAPTURE_ON, "flamegraph-capture.sh")
    _skip_if_missing(CAPTURE_OFF, "flamegraph-capture-offcpu.sh")
    _skip_if_no_perf()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACTS_DIR / "server.log"

    env = os.environ.copy()
    env["LLM_DEVICE_BACKEND"] = "mock_pipeline"

    log_file = open(log_path, "wb")
    proc = subprocess.Popen(
        [str(BINARY), "-p", str(PORT), "-h", HOST, "-t", "4"],
        cwd=str(CPP_SERVER_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode} during startup "
                f"(log: {log_path})"
            )
        try:
            r = requests.get(f"{BASE_URL}/tt-liveness", timeout=2)
            if r.status_code == 200 and r.json().get("model_ready") is True:
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.0)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        raise TimeoutError(f"server not ready within {READY_TIMEOUT_SECONDS}s (log: {log_path})")

    yield proc

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    log_file.close()


def _send_request() -> int:
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Tell me a story about a robot. " * 4}],
        "max_tokens": 256,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        return r.status_code
    except requests.RequestException:
        return 0


def _drive_load(stop_at: float, concurrency: int) -> tuple[int, int]:
    ok = errs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        inflight = [ex.submit(_send_request) for _ in range(concurrency)]
        while time.time() < stop_at:
            done, pending = concurrent.futures.wait(
                inflight, timeout=0.05, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for f in done:
                if f.result() == 200:
                    ok += 1
                else:
                    errs += 1
            inflight = list(pending) + [ex.submit(_send_request) for _ in range(len(done))]
        for f in inflight:
            try:
                if f.result(timeout=30) == 200:
                    ok += 1
                else:
                    errs += 1
            except Exception:
                errs += 1
    return ok, errs


def _run_capture(script: Path, target: str, seconds: int) -> Path:
    """Run a flamegraph-capture script, return the output directory it created."""
    result = subprocess.run(
        [str(script), target, str(seconds)],
        cwd=str(CPP_SERVER_DIR),
        capture_output=True,
        text=True,
        timeout=seconds + 60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed (rc={result.returncode}):\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    # The script prints the output dir as "Output: <path>"
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Output:"):
            return Path(line.split("Output:", 1)[1].strip())
    raise RuntimeError(f"could not parse output dir from {script.name} stdout:\n{result.stdout}")


def _copy_svgs(src_dir: Path, dest_dir: Path, prefix: str) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for svg in sorted(src_dir.glob("*.svg")):
        dest = dest_dir / f"{prefix}_{svg.name}"
        shutil.copy2(svg, dest)
        copied.append(dest)
    return copied


def test_perf_flamegraph(server):
    """Drive load and capture on-CPU + off-CPU flamegraphs for main + worker."""
    # Start load in a background thread; let it ramp up before capturing.
    stop_at = time.time() + LOAD_RAMP_SECONDS + 2 * CAPTURE_SECONDS + 10
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    load_future = pool.submit(_drive_load, stop_at, CONCURRENCY)

    time.sleep(LOAD_RAMP_SECONDS)

    on_dir = _run_capture(CAPTURE_ON, "all", CAPTURE_SECONDS)
    off_dir = _run_capture(CAPTURE_OFF, "all", CAPTURE_SECONDS)

    ok, errs = load_future.result()
    pool.shutdown(wait=True)

    on_svgs = _copy_svgs(on_dir, ARTIFACTS_DIR, "oncpu")
    off_svgs = _copy_svgs(off_dir, ARTIFACTS_DIR, "offcpu")

    # Write a one-line summary file alongside the artifacts so a reviewer can
    # see at a glance how loaded the system was during capture.
    summary = ARTIFACTS_DIR / "summary.txt"
    summary.write_text(
        f"requests_ok={ok}\nrequests_err={errs}\n"
        f"capture_seconds={CAPTURE_SECONDS}\nconcurrency={CONCURRENCY}\n"
        f"on_cpu_dir={on_dir}\noff_cpu_dir={off_dir}\n"
    )

    assert ok > 0, f"no successful requests during capture (errs={errs}); see {ARTIFACTS_DIR / 'server.log'}"
    assert len(on_svgs) >= 2, f"expected at least 2 on-CPU SVGs (main+worker), got {len(on_svgs)}"
    assert len(off_svgs) >= 2, f"expected at least 2 off-CPU SVGs (main+worker), got {len(off_svgs)}"
