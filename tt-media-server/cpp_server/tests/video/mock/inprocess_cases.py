#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""In-process mode of the video service, on the mock runner (no device).

With one DEVICE_IDS group the runner runs on a thread of the server process and the
queue is a deque in memory: no worker process, no shared memory, no request/response
files. Every case starts its own tt_media_server_cpp (private ports, shm names and
output folder). Cases:

  selection  one DEVICE_IDS group -> in process; two groups -> worker processes;
             TT_VIDEO_IN_PROCESS=0 -> a worker process; =1 with two groups -> the
             server refuses to start
  shape      no child process, no shm queue, no ipc folder, liveness shows the
             server pid as the worker
  queue      a cancelled queued job is taken out of the queue (never reaches the
             runner), a running one becomes cancelling -> cancelled, order of the rest
  sync       ?async=false returns the file; the queue limit still answers 429
  shutdown   SIGTERM when idle exits at once with code 0; SIGTERM during a
             generation exits after the grace period with code 1
  warmup     the failure of the device runner's warmup stops the server (code 1)

Usage (from cpp_server, needs a built ./build/tt_media_server_cpp):
    python3 tests/video/mock/inprocess_cases.py [--only selection,shape,...]
"""

import argparse
import glob
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import worker_crash_cases as w

GEN = w.GENERATIONS
note = w.note


def server(port, tag, workdir, **env):
    """Start a mock server in process mode by default; returns (proc, log, out)."""
    out = workdir / f"{tag}_videos"
    base = {
        "TT_VIDEO_MOCK_DELAY_MS": "0",
        "TT_VIDEO_OUTPUT_DIR": str(out),
        "TT_VIDEO_IN_PROCESS": None,  # None: not set, the server decides
    }
    base.update(env)
    proc, log = w.start(
        port, f"ip{tag}", 0, workdir, {k: (None if v is None else str(v)) for k, v in base.items()}
    )
    return proc, log, out


def ready(port, timeout=60):
    return bool(w.wait_until(lambda: w.model_ready(port), timeout))


def create(port, prompt="p", **body):
    return w.call(port, "POST", GEN, {"prompt": prompt, **body})


def children(pid):
    """Pids of the direct child processes of `pid`."""
    found = []
    for stat in glob.glob("/proc/[0-9]*/stat"):
        try:
            fields = Path(stat).read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == pid:
            found.append(int(stat.split("/")[2]))
    return found


def shm_with(prefix):
    return sorted(p for p in os.listdir("/dev/shm") if prefix in p)


def ipc_dirs():
    return set(glob.glob("/tmp/tt-video-ipc-*"))


# ----------------------------------------------------------------------------------
def case_selection(wd):
    print("=== selection: when the runner runs in the server process ===")
    # one group, nothing set: in process
    port = 18170
    proc, log, _ = server(port, "s1", wd)
    try:
        note(ready(port), "one DEVICE_IDS group: ready")
        pid, alive = w.worker(port)
        text = log.read_text(errors="replace")
        note(pid == proc.pid and alive, "the worker in /tt-liveness is the server process", f"pid={pid} server={proc.pid}")
        note("in-process video service" in text, "the log says in-process")
        note(children(proc.pid) == [], "no child process", str(children(proc.pid)))
    finally:
        w.stop(proc)

    # TT_VIDEO_IN_PROCESS=0: a worker process
    port = 18171
    proc, log, _ = server(port, "s2", wd, TT_VIDEO_IN_PROCESS=0)
    try:
        note(ready(port), "TT_VIDEO_IN_PROCESS=0: ready")
        pid, _ = w.worker(port)
        note(pid is not None and pid != proc.pid, "the worker is a separate process", f"pid={pid} server={proc.pid}")
        note(len(children(proc.pid)) == 1, "one child process", str(children(proc.pid)))
    finally:
        w.stop(proc)

    # two groups: worker processes, even when the variable is not set
    port = 18172
    proc, log, _ = server(port, "s3", wd, DEVICE_IDS="(0),(1)")
    try:
        note(ready(port), "two DEVICE_IDS groups: ready")
        info = w.liveness(port).get("worker_info") or {}
        pids = {v.get("pid") for v in info.values()}
        note(len(info) == 2 and proc.pid not in pids, "two worker processes", str(sorted(pids)))
    finally:
        w.stop(proc)

    # =1 with two groups is refused
    port = 18173
    proc, log, _ = server(port, "s4", wd, DEVICE_IDS="(0),(1)", TT_VIDEO_IN_PROCESS=1)
    try:
        exited = w.wait_until(lambda: proc.poll() is not None, 30)
        note(bool(exited) and proc.returncode != 0, "TT_VIDEO_IN_PROCESS=1 with two groups: the server does not start", f"code={proc.poll()}")
        note("needs exactly one DEVICE_IDS group" in log.read_text(errors="replace"), "the log says why")
    finally:
        w.stop(proc)


def case_shape(wd):
    print("=== shape: nothing of the process mode machinery exists in process mode ===")
    port = 18174
    before = ipc_dirs()
    proc, log, _ = server(port, "sh", wd)
    try:
        note(ready(port), "server ready")
        for _ in range(3):
            create(port)
        time.sleep(1)
        # the per-worker metrics segment (WorkerMetricsShm, created by main for every
        # service) is not a task or result queue and stays; the queues must not exist
        queues = [n for n in shm_with("wcipsh_") if not n.endswith("_metrics")]
        note(queues == [], "no shm task or result queue", str(queues))
        note(not (ipc_dirs() - before), "no ipc folder", str(sorted(ipc_dirs() - before)))
        note(children(proc.pid) == [], "no child process")
        st, jobs = w.call(port, "GET", "/v1/videos/jobs")
        done = w.wait_until(lambda: all(j["status"] == "completed" for j in w.call(port, "GET", "/v1/videos/jobs")[1]), 20)
        note(bool(done), "three jobs completed", str([j["status"] for j in w.call(port, "GET", "/v1/videos/jobs")[1]]))
    finally:
        w.stop(proc)


def case_queue(wd):
    print("=== queue: cancel takes a queued job out of the in-memory queue ===")
    port = 18175
    proc, log, out = server(port, "q", wd, TT_VIDEO_MOCK_DELAY_MS=2500, TT_VIDEO_MOCK_WRITE_FILE=1)
    try:
        note(ready(port), "server ready")
        ids = [create(port, p)[1]["id"] for p in ("first", "second", "third")]
        w.wait_until(lambda: w.job(port, ids[0])[0] == "in_progress", 10)
        note(w.liveness(port).get("queue_size") == 3, "queue_size counts the running job and the two waiting", str(w.liveness(port).get("queue_size")))
        st, body = w.call(port, "POST", f"{GEN}/{ids[1]}/cancel")
        note(st == 200 and body.get("status") == "cancelled", "cancel of a queued job: cancelled at once", f"{st} {body.get('status')}")
        note(w.liveness(port).get("queue_size") == 2, "it left the queue", str(w.liveness(port).get("queue_size")))
        # cancel of the running job
        st, body = w.call(port, "POST", f"{GEN}/{ids[0]}/cancel")
        note(st == 200 and body.get("status") == "cancelling", "cancel of the running job: cancelling", f"{st} {body.get('status')}")
        first = w.wait_job(port, ids[0], w.FINAL, 20)
        note(first[0] == "cancelled", "the running job ends cancelled", first[0])
        third = w.wait_job(port, ids[2], w.FINAL, 20)
        note(third[0] == "completed", "the third job runs next and completes", third[0])
        second = w.job(port, ids[1])[0]
        note(second == "cancelled", "the cancelled queued job stays cancelled", second)
        text = log.read_text(errors="replace")
        note(text.count("MockVideoRunner] task") == 2, "the runner ran 2 tasks (the cancelled queued one never reached it)", str(text.count("MockVideoRunner] task")))
        note("removed from the queue" in text, "the log says it was removed from the queue")
        mp4s = sorted(p.name for p in out.glob("*.mp4")) if out.exists() else []
        note(len(mp4s) == 1, "only the completed job left an mp4 (the cancelled running job's file was removed)", str(mp4s))
        st, again = w.call(port, "POST", f"{GEN}/{ids[1]}/cancel")
        note(st == 404, "cancel of an already cancelled job: 404", str(st))
    finally:
        w.stop(proc)


def case_sync(wd):
    print("=== sync: ?async=false and the queue limit in process mode ===")
    port = 18176
    proc, log, out = server(port, "sy", wd, TT_VIDEO_MOCK_DELAY_MS=1500, TT_VIDEO_MOCK_WRITE_FILE=1, MAX_QUEUE_SIZE=2)
    try:
        note(ready(port), "server ready")
        import urllib.request, json

        def sync(prompt):
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{GEN}?async=false",
                data=json.dumps({"prompt": prompt}).encode(),
                method="POST",
                headers={"content-type": "application/json", "authorization": f"Bearer {w.KEY}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.status, resp.headers.get("content-type"), resp.read()
            except urllib.error.HTTPError as e:
                return e.code, e.headers.get("content-type"), e.read()

        st, ctype, data = sync("one")
        note(st == 200 and ctype == "video/mp4" and data[4:8] == b"ftyp", "sync request: 200 and an mp4", f"{st} {ctype} {len(data)} bytes")
        # fill the queue: one running + one waiting = 2 = MAX_QUEUE_SIZE
        create(port, "a")
        create(port, "b")
        st, _ = create(port, "c")
        note(st == 429, "a third request over MAX_QUEUE_SIZE=2: 429", str(st))
        st, _, _ = sync("d")
        note(st == 429, "a sync request over the limit: 429", str(st))
    finally:
        w.stop(proc)


def case_shutdown(wd):
    print("=== shutdown: SIGTERM in process mode ===")
    port = 18177
    proc, log, _ = server(port, "d1", wd)
    try:
        note(ready(port), "idle server ready")
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            code = None
        note(code == 0 and time.time() - t0 < 10, "SIGTERM when idle: exit code 0 at once", f"code={code} after {time.time() - t0:.1f}s")
    finally:
        w.stop(proc)

    port = 18178
    proc, log, _ = server(port, "d2", wd, TT_VIDEO_MOCK_DELAY_MS=30000)
    try:
        note(ready(port), "server ready")
        _, job = create(port)
        w.wait_until(lambda: w.job(port, job["id"])[0] == "in_progress", 10)
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            code = None
        took = time.time() - t0
        note(code is not None, "SIGTERM during a generation: the server exits", f"code={code} after {took:.1f}s")
        note(code == 1 and 8 < took < 20, "after the grace period, with code 1 (a generation cannot be interrupted)", f"code={code} after {took:.1f}s")
        note("did not finish within" in log.read_text(errors="replace"), "the log says why")
        # _Exit runs no destructors, so the service unlinks the metrics segment itself.
        note(not shm_with("wcipd2_"), "the metrics shm segment is removed although the exit skips destructors", str(shm_with("wcipd2_")))
    finally:
        w.stop(proc)


def case_warmup(wd):
    print("=== warmup: the device runner fails its warmup in process mode ===")
    port = 18179
    env = dict(TT_VIDEO_PYTHON_DIR="/nonexistent", MODEL_RUNNER_TYPE=None, TT_VIDEO_IN_PROCESS=1)
    proc, log, _ = server(port, "w", wd, **env)
    try:
        t0 = time.time()
        exited = w.wait_until(lambda: proc.poll() is not None, 40)
        note(bool(exited) and proc.returncode == 1, "the server exits with code 1", f"code={proc.poll()} after {time.time() - t0:.1f}s")
        text = log.read_text(errors="replace")
        note("Video runner failed to start" in text or "Warmup failed" in text, "the log shows the warmup failure")
        note("killed by signal" not in text, "no crash signal")
    finally:
        w.stop(proc)


CASES = {
    "selection": case_selection,
    "shape": case_shape,
    "queue": case_queue,
    "sync": case_sync,
    "shutdown": case_shutdown,
    "warmup": case_warmup,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated case names")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n] or list(CASES)
    if not w.BINARY.exists():
        raise SystemExit(f"{w.BINARY} is not built")
    with tempfile.TemporaryDirectory(prefix="tt-video-inprocess.") as tmp:
        for n in names:
            CASES[n](Path(tmp))
    failed = w.results.count(False)
    print(f"\n{len(w.results) - failed} PASS, {failed} FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
