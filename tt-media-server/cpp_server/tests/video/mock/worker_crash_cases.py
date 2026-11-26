#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Worker crash tests of the video service, on the mock runner (no device).

Starts its own tt_media_server_cpp (private shm names and ports, a temporary
output folder) with two DEVICE_IDS groups (two worker processes; one group would
run the runner in the server process), kills a worker process with SIGKILL and
checks what the service does. A dead worker is not restarted: the service is marked failed and
the server process exits with code 1, so a supervisor (systemd, Kubernetes)
starts a fresh one. The job table lives in the server process, so no job is left
waiting for a worker that is gone.

  A. kill -9 of the worker while a job runs and another one waits in the queue:
     the server exits with code 1 and says why in its log.
  B. warmup failure of the device runner (a wrong TT_VIDEO_PYTHON_DIR, so the
     cpp_dit_adapters package is not found and no chip is opened): the worker
     process must exit, so that the service sees the dead worker and shuts the
     server down with code 1. Before the fix the worker
     logged "exiting after fatal error" but hung on exit (the Python GIL was
     still held), and the server kept running with `model_ready=true`. A second
     defect hid behind it: `stop()` called `attr()` on null Python handles and the
     worker died with SIGSEGV on exit instead of exiting with code 1.

Usage (from cpp_server, needs a built ./build/tt_media_server_cpp):
    python3 tests/video/mock/worker_crash_cases.py
"""

import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ROOT / "build" / "tt_media_server_cpp"
KEY = "test"
GENERATIONS = "/v1/videos/generations"

results = []


def note(ok, name, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}".rstrip(), flush=True)


def start(port, tag, workdir, extra_env=None):
    env = dict(
        os.environ,
        MODEL_SERVICE="video",
        MODEL_RUNNER_TYPE="video_mock",
        MODEL_RUNNER="tt-wan2.2",
        OPENAI_API_KEY=KEY,
        TT_VIDEO_MOCK_DELAY_MS="4000",
        TT_VIDEO_OUTPUT_DIR=str(workdir / f"{tag}_videos"),
        TT_MEDIA_TASK_QUEUE=f"wc{tag}_tasks",
        TT_MEDIA_RESULT_QUEUE=f"wc{tag}_results",
        TT_WARMUP_SIGNALS_QUEUE=f"wc{tag}_warm",
        TT_WORKER_METRICS_SHM=f"/wc{tag}_metrics",
        # These cases are about worker processes (kill one, shm queues). One
        # DEVICE_IDS group runs the runner in the server process, so the mock
        # server gets two groups, hence two workers (the mock runner opens no
        # chip). inprocess_cases.py covers one group.
        DEVICE_IDS="(0),(1)",
    )
    for key, value in (extra_env or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    log = workdir / f"{tag}.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [str(BINARY), "-p", str(port)],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    return proc, log


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def call(port, method, path, body=None, auth=True):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        method=method,
    )
    if body is not None:
        req.add_header("content-type", "application/json")
    if auth:
        req.add_header("authorization", f"Bearer {KEY}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read()
            return resp.status, (json.loads(text) if text.strip() else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}
    except OSError:
        return None, {}


def liveness(port):
    return call(port, "GET", "/tt-liveness", auth=False)[1] or {}


def model_ready(port):
    return liveness(port).get("model_ready") is True


def wait_until(predicate, timeout):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.5)
    return None


def workers(port):
    """{worker id: (pid, is_alive)} from /tt-liveness."""
    info = liveness(port).get("worker_info") or {}
    return {k: (v.get("pid"), v.get("is_alive")) for k, v in info.items()}


def worker(port, worker_id="0"):
    """(pid, is_alive) of one worker from /tt-liveness, or (None, None)."""
    return workers(port).get(worker_id, (None, None))


def job(port, jid):
    status, body = call(port, "GET", f"{GENERATIONS}/{jid}")
    return (body.get("status") if status == 200 else f"http{status}"), body


def wait_job(port, jid, wanted, timeout):
    last = {}

    def check():
        nonlocal last
        last = job(port, jid)
        return last[0] in wanted

    wait_until(check, timeout)
    return last


def case_a(workdir):
    print(
        "=== A. kill -9 of one of two workers while both run a job and a third waits ==="
    )
    port = 18150
    proc, log = start(port, "a", workdir)
    try:
        note(bool(wait_until(lambda: model_ready(port), 60)), "server ready")
        pid0, _ = worker(port)
        pid1_other, _ = worker(port, "1")
        note(
            pid0 is not None and pid1_other is not None and pid0 != pid1_other,
            "two worker pids are read from /tt-liveness",
            f"pids={pid0},{pid1_other}",
        )
        id1 = call(port, "POST", GENERATIONS, {"prompt": "job one"})[1]["id"]
        id2 = call(port, "POST", GENERATIONS, {"prompt": "job two"})[1]["id"]
        id3 = call(port, "POST", GENERATIONS, {"prompt": "job three"})[1]["id"]
        s1, _ = wait_job(port, id1, ("in_progress",), 15)
        s2, _ = wait_job(port, id2, ("in_progress",), 15)
        note(
            s1 == "in_progress" and s2 == "in_progress",
            "jobs 1 and 2 are in_progress, one per worker",
            f"status={s1},{s2}",
        )
        note(job(port, id3)[0] == "queued", "job 3 is queued behind them")

        os.kill(pid0, signal.SIGKILL)
        t0 = time.time()
        exited = wait_until(lambda: proc.poll() is not None, 30)
        note(
            bool(exited) and proc.returncode == 1,
            "the server process exits with code 1 (nothing waits for the dead worker)",
            f"exit code={proc.poll()} after {time.time() - t0:.1f}s",
        )
        text = log.read_text(errors="replace")
        note(
            "fatal worker error" in text,
            "the log says the server exits because of a fatal worker error",
        )
    finally:
        stop(proc)


def case_b(workdir):
    print("=== B. the device runner fails its warmup: the worker must exit ===")
    port = 18151
    # MODEL_RUNNER_TYPE is removed so that the real VideoDeviceRunner is used. The
    # runner looks for the cpp_dit_adapters package before importing anything, so
    # with a wrong directory it fails before any device is touched.
    env = dict(TT_VIDEO_PYTHON_DIR="/nonexistent", MODEL_RUNNER_TYPE=None)
    proc, log = start(port, "b", workdir, env)
    try:
        t0 = time.time()
        exited = wait_until(lambda: proc.poll() is not None, 30)
        note(
            bool(exited) and proc.returncode == 1,
            "the server exits with code 1 after the failed warmup",
            f"exit code={proc.poll()} after {time.time() - t0:.1f}s",
        )
        text = log.read_text(errors="replace")
        note("Warmup failed" in text, "the log shows the warmup failure")
        note(
            "killed by signal" not in text,
            "the failed worker exits by itself, it is not killed by a signal (no segfault)",
        )
    finally:
        stop(proc)


def main():
    if not BINARY.exists():
        raise SystemExit(f"{BINARY} is not built")
    with tempfile.TemporaryDirectory(prefix="tt-video-worker-crash.") as tmp:
        workdir = Path(tmp)
        case_a(workdir)
        case_b(workdir)
    failed = results.count(False)
    print(f"\n{len(results) - failed} PASS, {failed} FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
