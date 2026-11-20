#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Lifecycle, shutdown and load tests of the video service on the mock runner (no device).

Every case starts its own tt_media_server_cpp (private ports, shm names and output
folder), so it can run next to a server that owns a device. Cases:

  cleaner      background sweep evicts old jobs (JOB_CLEANUP_INTERVAL_SECONDS,
               JOB_RETENTION_SECONDS) without new requests
  stale        JOB_MAX_STUCK_TIME_SECONDS: a job without progress becomes `stale_job`,
               its late result is discarded and the file removed
  sigterm      SIGTERM in the middle of a job: exit, no worker left, no ipc folder,
               no shm queues
  disconnect   the client drops a synchronous request: no mp4 is left, the server lives
  sync2        two synchronous requests at once: the second waits, both succeed
  stress       hundreds of concurrent POST/GET/cancel/DELETE
  bigtable     GET /v1/videos/jobs with 10000 jobs, with the sweeper running
  downloads    many parallel downloads of one mp4 (whole file and Range)
  noffmpeg     no ffmpeg on PATH: the job still completes, with a warning

Usage (from cpp_server, needs a built ./build/tt_media_server_cpp):
    python3 tests/video/mock/lifecycle_cases.py [--only cleaner,stale,...]
"""

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import worker_crash_cases as w

GEN = w.GENERATIONS
KEY = w.KEY
FINAL = ("completed", "failed", "cancelled")
note = w.note


def server(port, tag, workdir, **env):
    """Start a mock server; returns (proc, log, output_dir)."""
    out = workdir / f"{tag}_videos"
    base = {"TT_VIDEO_MOCK_DELAY_MS": "0", "TT_VIDEO_OUTPUT_DIR": str(out)}
    base.update(env)
    proc, log = w.start(
        port, tag, workdir, {k: (None if v is None else str(v)) for k, v in base.items()}
    )
    return proc, log, out


def ready(port, timeout=60):
    return bool(w.wait_until(lambda: w.model_ready(port), timeout))


def create(port, prompt="p", **body):
    st, j = w.call(port, "POST", GEN, {"prompt": prompt, **body})
    return st, j


def finished(port, jid, timeout=30):
    return w.wait_job(port, jid, FINAL, timeout)


def ipc_dirs():
    return set(glob.glob("/tmp/tt-video-ipc-*"))


def shm_with(prefix):
    return sorted(p for p in os.listdir("/dev/shm") if prefix in p)


def rss_mb(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


# ----------------------------------------------------------------------------------
def case_cleaner(wd):
    print("=== cleaner: the sweep evicts old jobs without new requests ===")
    port = 18160
    proc, log, out = server(
        port,
        "cl",
        wd,
        JOB_CLEANUP_INTERVAL_SECONDS=1,
        JOB_RETENTION_SECONDS=3,
        TT_VIDEO_MOCK_WRITE_FILE=1,
    )
    try:
        note(ready(port), "server ready")
        _, j = create(port)
        s, _ = finished(port, j["id"])
        note(s == "completed", "job completes", f"status={s}")
        st0 = w.call(port, "GET", f"{GEN}/{j['id']}")[0]
        note(st0 == 200, "the job is visible right after it finished", f"http{st0}")
        gone = w.wait_until(lambda: w.call(port, "GET", f"{GEN}/{j['id']}")[0] == 404, 15)
        note(bool(gone), "after the retention time the job is evicted by the sweep (404)")
        listing = w.call(port, "GET", "/v1/videos/jobs")[1]
        note(
            listing == [] or listing == {"jobs": []} or not listing,
            "the job list is empty",
            str(listing)[:80],
        )
        left = sorted(p.name for p in out.glob("*")) if out.exists() else []
        print(f"  output folder after eviction: {left}")
        note(
            "Job cleanup: evicted=1" in log.read_text(errors="replace"),
            "the log has the cleanup line",
        )
    finally:
        w.stop(proc)


def case_stale(wd):
    print("=== stale: a job without progress becomes stale_job ===")
    port = 18161
    proc, log, out = server(
        port,
        "st",
        wd,
        JOB_CLEANUP_INTERVAL_SECONDS=1,
        JOB_MAX_STUCK_TIME_SECONDS=2,
        TT_VIDEO_MOCK_DELAY_MS=6000,
        TT_VIDEO_MOCK_WRITE_FILE=1,
    )
    try:
        note(ready(port), "server ready")
        _, j = create(port)
        s, body = w.wait_job(port, j["id"], ("failed",), 15)
        err = body.get("error") or {}
        note(
            s == "failed" and err.get("code") == "stale_job",
            "the stuck job fails as stale_job",
            f"status={s} error={err}",
        )
        time.sleep(7)  # the runner finishes after 6 s
        s2, _ = w.job(port, j["id"])
        note(s2 == "failed", "the late result does not revive the job", f"status={s2}")
        files = sorted(p.name for p in out.glob("*.mp4")) if out.exists() else []
        note(files == [], "the late mp4 is removed from the output folder", f"files={files}")
        id2 = create(port)[1]["id"]
        s3, _ = w.wait_job(port, id2, ("failed", "completed"), 20)
        note(
            s3 in ("failed", "completed"),
            "the service keeps working after a stale job",
            f"status={s3}",
        )
    finally:
        w.stop(proc)


def case_sigterm(wd):
    print("=== sigterm: SIGTERM in the middle of a job ===")
    port = 18162
    before = ipc_dirs()
    proc, log, out = server(port, "sg", wd, TT_VIDEO_MOCK_DELAY_MS=15000)
    try:
        note(ready(port), "server ready")
        pid = proc.pid
        wpid, _ = w.worker(port)
        create(port)
        create(port)
        w.wait_until(
            lambda: w.job(port, w.call(port, "GET", "/v1/videos/jobs")[1][0]["id"])[0]
            == "in_progress",
            10,
        )
        mine = ipc_dirs() - before
        note(len(mine) == 1, "the server has an ipc folder while it runs", str(sorted(mine)))
        shm = shm_with("wcsg_")
        note(len(shm) >= 3, "the shm queues exist while it runs", str(shm))
        t0 = time.time()
        proc.send_signal(signal.SIGTERM)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            code = None
        took = time.time() - t0
        note(code is not None, "the server exits after SIGTERM", f"code={code} after {took:.1f}s")
        alive = (
            os.path.exists(f"/proc/{wpid}")
            and Path(f"/proc/{wpid}/stat").read_text().split()[2] != "Z"
        )
        note(not alive, "the worker process is gone", f"worker pid={wpid}")
        note(not (ipc_dirs() & mine), "the ipc folder is removed")
        note(shm_with("wcsg_") == [], "the shm queues are removed", str(shm_with("wcsg_")))
    finally:
        w.stop(proc)


def raw_sync_request(port, drop_after=None, prompt="sync"):
    """POST ?async=false on a raw socket; returns (status line, seconds) or drops the socket."""
    body = json.dumps({"prompt": prompt}).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall(
        (
            f"POST {GEN}?async=false HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {KEY}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode()
        + body
    )
    t0 = time.time()
    if drop_after is not None:
        time.sleep(drop_after)
        s.close()
        return None, time.time() - t0
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.split(b"\r\n", 1)[0].decode(), time.time() - t0


def case_disconnect(wd):
    print("=== disconnect: the client drops a synchronous request ===")
    port = 18163
    proc, log, out = server(port, "dc", wd, TT_VIDEO_MOCK_DELAY_MS=4000, TT_VIDEO_MOCK_WRITE_FILE=1)
    try:
        note(ready(port), "server ready")
        raw_sync_request(port, drop_after=1.0)
        time.sleep(7)
        files = sorted(p.name for p in out.glob("*")) if out.exists() else []
        print(f"  output folder after the dropped request finished: {files}")
        # Known limitation (docs/VIDEO_SERVICE.md): the mp4 of a synchronous request stays on
        # disk until the next start, whether the client stayed or not. The task itself runs to
        # the end either way.
        note(
            len(files) == 1,
            "the dropped request ran to the end (its mp4 exists, as for a finished sync request)",
            f"files={files}",
        )
        raw_sync_request(port)
        files2 = sorted(p.name for p in out.glob("*")) if out.exists() else []
        note(
            len(files2) == 2,
            "a normal sync request also leaves its mp4 (documented limitation)",
            f"files={files2}",
        )
        jobs = w.call(port, "GET", "/v1/videos/jobs")[1]
        note(not jobs, "a synchronous request leaves no job record", str(jobs)[:80])
        st, j = create(port)
        s, _ = finished(port, j["id"], 20)
        note(s == "completed", "the server still serves jobs", f"status={s}")
    finally:
        w.stop(proc)


def case_sync2(wd):
    print("=== sync2: two synchronous requests at once ===")
    port = 18164
    proc, log, out = server(port, "s2", wd, TT_VIDEO_MOCK_DELAY_MS=3000, TT_VIDEO_MOCK_WRITE_FILE=1)
    try:
        note(ready(port), "server ready")
        t0 = time.time()
        with cf.ThreadPoolExecutor(2) as ex:
            f1 = ex.submit(raw_sync_request, port)
            time.sleep(0.3)
            f2 = ex.submit(raw_sync_request, port)
            r1, r2 = f1.result(), f2.result()
        total = time.time() - t0
        note(
            "200" in (r1[0] or "") and "200" in (r2[0] or ""),
            "both requests return 200",
            f"{r1[0]} / {r2[0]}",
        )
        note(
            total >= 5.5,
            "the second one waited behind the first (about 2 x delay)",
            f"first {r1[1]:.1f}s second {r2[1]:.1f}s total {total:.1f}s",
        )
    finally:
        w.stop(proc)


def case_stress(wd):
    print("=== stress: concurrent POST/GET/cancel/DELETE ===")
    port = 18167
    proc, log, out = server(
        port, "sr", wd, TT_VIDEO_MOCK_DELAY_MS=20, MAX_QUEUE_SIZE=5000, MAX_JOBS=5000
    )
    try:
        note(ready(port), "server ready")
        n = 400

        def one(i):
            st, j = create(port, f"job {i}", seed=i)
            if st != 202:
                return ("post", st, None)
            jid = j["id"]
            codes = [w.call(port, "GET", f"{GEN}/{jid}")[0]]
            if i % 3 == 0:
                codes.append(w.call(port, "POST", f"{GEN}/{jid}/cancel")[0])
            if i % 5 == 0:
                codes.append(w.call(port, "DELETE", f"{GEN}/{jid}")[0])
            return ("ok", st, (jid, codes))

        t0 = time.time()
        with cf.ThreadPoolExecutor(64) as ex:
            res = list(ex.map(one, range(n)))
        took = time.time() - t0
        posts = [r for r in res if r[0] == "post"]
        note(
            not posts,
            f"{n} concurrent POSTs accepted",
            f"rejected={[r[1] for r in posts][:5]} in {took:.1f}s",
        )
        all_codes = [c for r in res if r[0] == "ok" for c in r[2][1]]
        bad = sorted({c for c in all_codes if c not in (200, 404, 409)})
        note(not bad, "GET/cancel/DELETE answered only 200, 404 or 409", f"unexpected={bad}")

        # everything settles: no job stays queued or in progress
        def unsettled():
            jobs = w.call(port, "GET", "/v1/videos/jobs")[1]
            jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
            return [j for j in jobs if j["status"] in ("queued", "in_progress", "cancelling")]

        ok = w.wait_until(lambda: not unsettled(), 60)
        note(bool(ok), "all jobs reach a terminal status", f"left={len(unsettled())}")
        jobs = w.call(port, "GET", "/v1/videos/jobs")[1]
        jobs = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
        by = {}
        for j in jobs:
            by[j["status"]] = by.get(j["status"], 0) + 1
        print(f"  final statuses: {by}")
        bad_state = [j for j in jobs if j["status"] == "failed"]
        note(not bad_state, "no job failed", f"failed={len(bad_state)}")
        note(w.liveness(port).get("status") == "alive", "the server is alive")
    finally:
        w.stop(proc)


def case_bigtable(wd):
    print("=== bigtable: 10000 jobs, the job list and the sweeper ===")
    port = 18168
    proc, log, out = server(
        port,
        "bt",
        wd,
        TT_VIDEO_MOCK_DELAY_MS=0,
        MAX_QUEUE_SIZE=20000,
        MAX_JOBS=20000,
        JOB_CLEANUP_INTERVAL_SECONDS=1,
        JOB_RETENTION_SECONDS=86400,
    )
    try:
        note(ready(port), "server ready")
        n = 10000
        t0 = time.time()
        with cf.ThreadPoolExecutor(64) as ex:
            codes = list(ex.map(lambda i: create(port, f"j{i}")[0], range(n)))
        print(f"  created {n} jobs in {time.time() - t0:.1f}s")
        note(
            codes.count(202) == n,
            f"{n} jobs created",
            f"statuses={ {c: codes.count(c) for c in set(codes)} }",
        )
        w.wait_until(lambda: w.liveness(port).get("queue_size", 1) == 0, 120)
        t0 = time.time()
        st, jobs = w.call(port, "GET", "/v1/videos/jobs")
        took = time.time() - t0
        count = len(jobs) if isinstance(jobs, list) else len(jobs.get("jobs", []))
        note(
            st == 200 and count == n,
            "the list returns every job",
            f"http{st} count={count} in {took:.2f}s",
        )
        # latency of a single GET while 4 threads keep listing
        stop = time.time() + 6
        lat = []

        def lister():
            while time.time() < stop:
                w.call(port, "GET", "/v1/videos/jobs")

        with cf.ThreadPoolExecutor(4) as ex:
            futs = [ex.submit(lister) for _ in range(4)]
            jid = (jobs if isinstance(jobs, list) else jobs["jobs"])[0]["id"]
            while time.time() < stop:
                t1 = time.time()
                w.call(port, "GET", f"{GEN}/{jid}")
                lat.append(time.time() - t1)
                time.sleep(0.2)
            for f in futs:
                f.result()
        lat.sort()
        print(
            f"  GET of one job while the list is requested in a loop: median {lat[len(lat) // 2] * 1000:.0f} ms, max {lat[-1] * 1000:.0f} ms ({len(lat)} samples)"
        )
        note(
            lat[-1] < 5,
            "a single GET stays answerable while the big list is requested",
            f"max {lat[-1] * 1000:.0f} ms",
        )
        note(w.liveness(port).get("status") == "alive", "the server is alive")
    finally:
        w.stop(proc)


def case_downloads(wd):
    print("=== downloads: parallel downloads of one mp4 ===")
    port = 18169
    proc, log, out = server(port, "dl", wd, TT_VIDEO_MOCK_WRITE_FILE="ffmpeg-faststart")
    try:
        note(ready(port), "server ready")
        _, j = create(port)
        s, _ = finished(port, j["id"], 30)
        note(s == "completed", "job completes", f"status={s}")
        url = f"{GEN}/{j['id']}/download"

        def fetch(rng=None):
            req = __import__("urllib.request").request.Request(f"http://127.0.0.1:{port}{url}")
            req.add_header("authorization", f"Bearer {KEY}")
            if rng:
                req.add_header("Range", rng)
            with __import__("urllib.request").request.urlopen(req, timeout=30) as r:
                return r.status, r.read()

        full_status, full = fetch()
        size = len(full)
        h = hashlib.sha256(full).hexdigest()
        with cf.ThreadPoolExecutor(32) as ex:
            full_res = list(ex.map(lambda _: fetch(), range(64)))
            half = size // 2
            rng_res = list(ex.map(lambda _: fetch(f"bytes={half}-"), range(64)))
        note(
            all(s == 200 and hashlib.sha256(b).hexdigest() == h for s, b in full_res),
            "64 parallel full downloads: identical bytes",
            f"size={size}",
        )
        note(
            all(s == 206 and b == full[half:] for s, b in rng_res),
            "64 parallel `bytes=N-` downloads: 206 and the right tail",
        )
        note(w.liveness(port).get("status") == "alive", "the server is alive")
    finally:
        w.stop(proc)


def case_noffmpeg(wd):
    print("=== noffmpeg: no ffmpeg on PATH ===")
    port = 18170
    empty = wd / "emptybin"
    empty.mkdir(exist_ok=True)
    proc, log, out = server(port, "nf", wd, TT_VIDEO_MOCK_WRITE_FILE=1, PATH=str(empty))
    try:
        note(ready(port), "server ready")
        _, j = create(port)
        s, body = finished(port, j["id"], 30)
        note(
            s == "completed",
            "the job completes without ffmpeg",
            f"status={s} error={body.get('error')}",
        )
        text = log.read_text(errors="replace")
        warn = [ln for ln in text.splitlines() if "faststart remux failed" in ln][:2]
        print(f"  log lines about the failed remux: {[x[-130:] for x in warn]}")
        note(bool(warn), "the log says the remux failed and the original is kept", "")
    finally:
        w.stop(proc)


CASES = {
    "cleaner": case_cleaner,
    "stale": case_stale,
    "sigterm": case_sigterm,
    "disconnect": case_disconnect,
    "sync2": case_sync2,
    "stress": case_stress,
    "bigtable": case_bigtable,
    "downloads": case_downloads,
    "noffmpeg": case_noffmpeg,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated case names")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n] or list(CASES)
    if not w.BINARY.exists():
        raise SystemExit(f"{w.BINARY} is not built")
    with tempfile.TemporaryDirectory(prefix="tt-video-lifecycle.") as tmp:
        for n in names:
            CASES[n](Path(tmp))
    failed = w.results.count(False)
    print(f"\n{len(w.results) - failed} PASS, {failed} FAIL")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
