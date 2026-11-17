#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Worker crash tests of the video service, on the mock runner (no device).

Starts its own tt_media_server_cpp (private shm names and ports, a temporary
output folder), kills the worker process with SIGKILL and checks what the
service does:

  A. default restart limit: the job the dead worker had taken becomes `failed`
     (`processing_error`), a new worker starts, the job that was waiting in the
     queue is served by it, and a new job completes.
  B. TT_VIDEO_WORKER_MAX_RESTARTS=1: the first crash is recovered, the second
     one makes the server exit with code 1.
  C. warmup failure of the device runner (a wrong TT_PYTHON_PATH, no chip is
     opened): the worker process must exit, so that the service sees the dead
     worker and shuts the server down with code 1. Before the fix the worker
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
FINAL = ("completed", "failed", "cancelled")

results = []


def note(ok, name, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}".rstrip(), flush=True)


def start(port, tag, max_restarts, workdir, extra_env=None):
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
        TT_VIDEO_WORKER_MAX_RESTARTS=str(max_restarts),
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


def worker(port):
    """(pid, is_alive) of the first worker from /tt-liveness, or (None, None)."""
    for info in (liveness(port).get("worker_info") or {}).values():
        return info.get("pid"), info.get("is_alive")
    return None, None


def new_worker_ready(port, old_pid):
    pid, alive = worker(port)
    return pid if pid and pid != old_pid and alive and model_ready(port) else None


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
    print("=== A. kill -9 of the worker while a job runs and another one waits ===")
    port = 18150
    proc, _ = start(port, "a", 3, workdir)
    try:
        note(bool(wait_until(lambda: model_ready(port), 60)), "server ready")
        pid0, _ = worker(port)
        note(pid0 is not None, "worker pid is read from /tt-liveness", f"pid={pid0}")
        id1 = call(port, "POST", GENERATIONS, {"prompt": "job one"})[1]["id"]
        id2 = call(port, "POST", GENERATIONS, {"prompt": "job two"})[1]["id"]
        s1, _ = wait_job(port, id1, ("in_progress",), 15)
        note(s1 == "in_progress", "job 1 is in_progress", f"status={s1}")
        note(job(port, id2)[0] == "queued", "job 2 is queued behind it")

        os.kill(pid0, signal.SIGKILL)
        t0 = time.time()
        s1, body1 = wait_job(port, id1, FINAL, 40)
        err = body1.get("error") or {}
        note(
            s1 == "failed" and err.get("code") == "processing_error",
            "job 1 (taken by the dead worker) becomes failed/processing_error",
            f"status={s1} code={err.get('code')} after {time.time() - t0:.1f}s",
        )
        pid1 = wait_until(lambda: new_worker_ready(port, pid0), 60)
        note(
            pid1 is not None,
            "a new worker process is started and the model is ready again",
            f"old pid={pid0} new pid={pid1} after {time.time() - t0:.1f}s",
        )
        s2, _ = wait_job(port, id2, FINAL, 40)
        note(
            s2 == "completed",
            "job 2 (still waiting in the queue) is served by the new worker",
            f"status={s2}",
        )
        id3 = call(port, "POST", GENERATIONS, {"prompt": "job three"})[1]["id"]
        s3, _ = wait_job(port, id3, FINAL, 40)
        note(s3 == "completed", "a new job after the restart completes", f"status={s3}")
    finally:
        stop(proc)


def case_b(workdir):
    print("=== B. restart limit: TT_VIDEO_WORKER_MAX_RESTARTS=1, kill the worker twice ===")
    port = 18151
    proc, log = start(port, "b", 1, workdir)
    try:
        note(bool(wait_until(lambda: model_ready(port), 60)), "server ready")
        pid0, _ = worker(port)
        os.kill(pid0, signal.SIGKILL)
        pid1 = wait_until(lambda: new_worker_ready(port, pid0), 60)
        note(
            pid1 is not None,
            "first crash: the worker is restarted (within the limit)",
            f"old={pid0} new={pid1}",
        )
        os.kill(pid1, signal.SIGKILL)
        t0 = time.time()
        exited = wait_until(lambda: proc.poll() is not None, 60)
        note(
            bool(exited) and proc.returncode == 1,
            "second crash (over the limit): the server process exits with code 1",
            f"exit code={proc.poll()} after {time.time() - t0:.1f}s",
        )
        note("giving up" in log.read_text(errors="replace"), "the log says the service gave up")
    finally:
        stop(proc)


def case_c(workdir):
    print("=== C. the device runner fails its warmup: the worker must exit ===")
    port = 18152
    # MODEL_RUNNER_TYPE is removed so that the real VideoDeviceRunner is used; the
    # import of the Python runner fails first, so no device is touched.
    env = dict(TT_PYTHON_PATH="/nonexistent", MODEL_RUNNER_TYPE=None)
    proc, log = start(port, "c", 0, workdir, env)
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
        case_c(workdir)
    failed = results.count(False)
    print(f"\n{len(results) - failed} PASS, {failed} FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
