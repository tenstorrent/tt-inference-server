#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Queue, download and sync mode on a REAL video server.

Report: docs/tests/2026-11-14-video-real-server-queue-download-sync.md

The server must already run with a real runner and report model_ready (start
command: tests/video/real_device/README.md). Jobs use num_inference_steps=--steps
(default 4); one generation takes about 80 s on a T3K.

Phases:
  validation  401 / 422 / 404 answers. No generation, takes seconds.
  all         (default) validation, then: three queued jobs cancelled around one
              that must still run, default steps (20) in the job record, DELETE
              409, download + Range + faststart + ffprobe, same seed -> identical
              file, two simultaneous sync requests. 6 generations, about 7
              minutes.

Optional checks need access to the server host (--server-log, --payload-dir);
without them they are reported as SKIP, not as failures.

Run (from cpp_server):
  python3 tests/video/real_device/jobs/queue_download_sync.py --server-log /path/to/server.log
"""

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c
from common import (
    GEN,
    UUID0,
    err_message,
    ffprobe,
    jreq,
    log_lines,
    log_pos,
    ls,
    markers,
    note,
    p,
    req,
    sha,
    skip,
    status,
    submit,
    wait,
)


def phase_validation():
    p("=== VALIDATION: auth, 422, 404 (no generation) ===")
    st, _ = jreq("POST", GEN, {"prompt": "x"}, token=None)
    note(st == 401, "POST without a token -> 401", f"http{st}")
    st, _ = jreq("POST", GEN, {"prompt": "x"}, token="wrong")
    note(st == 401, "POST with a wrong token -> 401", f"http{st}")
    st, _ = jreq("GET", "/v1/videos/jobs", token=None)
    note(st == 401, "GET jobs without a token -> 401", f"http{st}")
    st, j = jreq("POST", GEN, {"prompt": ""})
    note(st == 422, "empty prompt -> 422", f"http{st} {err_message(j)[:80]!r}")
    st, _ = jreq("POST", GEN, {"prompt": "x", "num_inference_steps": 3})
    note(st == 422, "steps=3 (below 4) -> 422", f"http{st}")
    st, _ = jreq("POST", GEN, {"prompt": "x", "num_inference_steps": 51})
    note(st == 422, "steps=51 (above 50) -> 422", f"http{st}")
    st, _ = jreq("POST", GEN, raw_body=b"{not json")
    note(st == 422, "invalid JSON -> 422", f"http{st}")
    st, _ = jreq("GET", f"{GEN}/{UUID0}")
    note(st == 404, "unknown job -> 404", f"http{st}")
    st, _ = jreq("POST", f"{GEN}/{UUID0}/cancel")
    note(st == 404, "cancel of an unknown job -> 404", f"http{st}")


def phase_queue_download_sync():
    p("=== SUITE: several jobs in the queue, download, repeatability, sync ===")
    pos0 = log_pos()

    p("--- 1. several jobs in the queue: cancel B, D, E; C must still run ---")
    pos = log_pos()
    st, a = submit("a red ball rolling on grass", seed=7)
    A = a["id"]
    note(wait(A, ("in_progress",), 600), "A in_progress")
    B = submit("job B (will be cancelled)", seed=11)[1]["id"]
    C = submit("a blue cube spinning slowly", seed=7)[1]["id"]
    D = submit("job D (will be cancelled)", seed=12)[1]["id"]
    E = submit("default steps", steps=None)[1]["id"]  # without num_inference_steps
    st, je = jreq("GET", f"{GEN}/{E}")
    steps = (je.get("request_parameters") or {}).get("num_inference_steps")
    note(
        steps == 20,
        "default num_inference_steps = 20 in the job record",
        f"steps={steps}",
    )
    note(
        all(status(x) == "queued" for x in (B, C, D, E)),
        "B, C, D, E are queued while A runs",
    )
    st, _ = jreq("DELETE", f"{GEN}/{A}")
    note(st == 409, "DELETE of a running job -> 409", f"http{st}")
    st, _ = jreq("DELETE", f"{GEN}/{C}")
    note(st == 409, "DELETE of a queued job -> 409", f"http{st}")
    for jid, n in ((B, "B"), (D, "D"), (E, "E")):
        st, j = jreq("POST", f"{GEN}/{jid}/cancel")
        note(
            st == 200 and j.get("status") == "cancelled",
            f"cancel {n} in the queue -> cancelled",
            f"http{st} status={j.get('status')}",
        )
    if c.PAYLOAD:
        note(len(markers()) == 3, "3 markers in the payload dir", str(ls(c.PAYLOAD)))
    note(status(C) == "queued", "C is still queued (untouched)")
    p(
        "waiting for A and then C (C must run even though B and D around it were cancelled)..."
    )
    wait(A, ("completed", "failed", "cancelled"), 3600)
    note(status(A) == "completed", "A completed")
    wait(C, ("completed", "failed", "cancelled"), 3600)
    note(
        status(C) == "completed",
        "C completed (a job that was not cancelled, between cancelled ones)",
    )
    time.sleep(3)
    if c.LOG:
        skips = log_lines(pos, "skipping cancelled task")
        note(
            len(skips) == 3 and all("before claim" in ln for ln in skips),
            "worker skipped exactly 3 tasks (before claim)",
            f"n={len(skips)}",
        )
    else:
        skip("worker skipped exactly 3 tasks", "no --server-log")
    for jid, n in ((B, "B"), (D, "D"), (E, "E")):
        st, j = jreq("GET", f"{GEN}/{jid}")
        note(
            j.get("status") == "cancelled" and "error" not in j,
            f"{n} cancelled without an error field",
        )
    if c.PAYLOAD:
        note(ls(c.PAYLOAD) == [], "payload dir empty", str(ls(c.PAYLOAD)))

    p("--- 2. download, Range, faststart, ffprobe (job A) ---")
    st, h, abytes = req("GET", f"{GEN}/{A}/download")
    size = len(abytes)
    note(
        st == 200
        and h.get("Accept-Ranges") == "bytes"
        and h.get("Content-Type", "").startswith("video/mp4")
        and int(h.get("Content-Length", -1)) == size
        and "attachment" in h.get("Content-Disposition", ""),
        "download 200 + headers",
        f"bytes={size}",
    )
    st, h, part = req("GET", f"{GEN}/{A}/download", headers={"Range": "bytes=0-99"})
    note(
        st == 206
        and part == abytes[:100]
        and h.get("Content-Range") == f"bytes 0-99/{size}",
        "Range 0-99 -> 206",
        f"http{st} {h.get('Content-Range')}",
    )
    st, h, part = req("GET", f"{GEN}/{A}/download", headers={"Range": "bytes=-10"})
    note(
        st == 206 and part == abytes[-10:],
        "Range -10 -> 206 (end of the file)",
        f"http{st}",
    )
    st, h, part = req(
        "GET", f"{GEN}/{A}/download", headers={"Range": f"bytes={size + 5}-"}
    )
    note(
        st == 416 and h.get("Content-Range") == f"bytes */{size}",
        "Range beyond the file -> 416",
        f"http{st} {h.get('Content-Range')}",
    )
    mid = size // 2
    st, h, part = req("GET", f"{GEN}/{A}/download", headers={"Range": f"bytes={mid}-"})
    note(
        st == 206 and part == abytes[mid:],
        "Range from the middle to the end -> 206",
        f"http{st}",
    )
    note(
        b"moov" in abytes and abytes.find(b"moov") < abytes.find(b"mdat"),
        "faststart: moov before mdat",
        f"moov={abytes.find(b'moov')} mdat={abytes.find(b'mdat')}",
    )
    apath = os.path.join(c.DLDIR, "A.mp4")
    open(apath, "wb").write(abytes)
    pr = ffprobe(apath)
    if pr is None:
        skip("ffprobe reads the video stream", "ffprobe is not installed")
    else:
        p("  ffprobe A:", json.dumps(pr)[:300])
        streams = [s for s in pr.get("streams", []) if s.get("codec_name")]
        note(
            bool(streams) and float(pr.get("format", {}).get("duration", 0)) > 0,
            "ffprobe reads the video stream and duration",
        )

    p("--- 3. repeatability: same prompt and seed as C ---")
    F = submit("a blue cube spinning slowly", seed=7)[1]["id"]
    wait(F, ("completed", "failed", "cancelled"), 3600)
    note(status(F) == "completed", "F completed")
    _, _, cbytes = req("GET", f"{GEN}/{C}/download")
    _, _, fbytes = req("GET", f"{GEN}/{F}/download")
    open(os.path.join(c.DLDIR, "C.mp4"), "wb").write(cbytes)
    open(os.path.join(c.DLDIR, "F.mp4"), "wb").write(fbytes)
    p(f"  C: {len(cbytes)} B sha={sha(cbytes)} | F: {len(fbytes)} B sha={sha(fbytes)}")
    note(
        cbytes == fbytes,
        "same seed -> identical file (byte for byte)",
        f"C={sha(cbytes)} F={sha(fbytes)}",
    )
    note(
        abytes != cbytes,
        "different prompt/seed -> different file (sanity)",
        f"A={sha(abytes)} C={sha(cbytes)}",
    )

    p("--- 4. sync mode (?async=false), two simultaneous requests ---")
    _, _, jl0 = req("GET", "/v1/videos/jobs")
    n0 = len(json.loads(jl0))
    out = {}

    def do_sync(tag, seed):
        t = time.time()
        st, h, raw = req(
            "POST",
            GEN + "?async=false",
            {"prompt": f"sync {tag}", "num_inference_steps": c.STEPS, "seed": seed},
            timeout=900,
        )
        out[tag] = (st, h, raw, time.time() - t)

    threads = [
        threading.Thread(target=do_sync, args=(t, s))
        for t, s in (("one", 21), ("two", 22))
    ]
    for t in threads:
        t.start()
        time.sleep(1)
    for t in threads:
        t.join()
    for tag in ("one", "two"):
        st, h, raw, el = out[tag]
        note(
            st == 200
            and h.get("Content-Type", "").startswith("video/mp4")
            and raw[4:8] == b"ftyp",
            f"sync {tag}: 200 + mp4",
            f"bytes={len(raw)} duration={el:.0f}s",
        )
        try:
            ok = float(h.get("X-Generation-Time", "")) > 0
        except ValueError:
            ok = False
        note(ok, f"sync {tag}: X-Generation-Time", f"{h.get('X-Generation-Time')!r}")
        open(os.path.join(c.DLDIR, f"sync_{tag}.mp4"), "wb").write(raw)
    t1, t2 = out["one"][3], out["two"][3]
    note(
        t2 > t1 * 1.5,
        "two sync requests: the second waits behind the first (one device)",
        f"first={t1:.0f}s second={t2:.0f}s",
    )
    _, _, jl1 = req("GET", "/v1/videos/jobs")
    note(len(json.loads(jl1)) == n0, "sync creates no job record")

    p("--- 5. leftovers and health ---")
    if c.PAYLOAD:
        note(ls(c.PAYLOAD) == [], "payload dir empty at the end", str(ls(c.PAYLOAD)))
    if c.LOG:
        bad = log_lines(pos0, "CRASHED") + log_lines(pos0, "TT_FATAL")
        note(not bad, "no CRASHED/TT_FATAL in the log during the test", str(bad[:2]))
    st, lv = jreq("GET", "/tt-liveness")
    note(lv.get("model_ready") is True, "server is still model_ready", str(lv)[:100])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    c.add_common_args(ap)
    ap.add_argument("--phase", default="all", choices=["validation", "all"])
    args = ap.parse_args()
    if not c.setup(args):
        return 1
    phase_validation()
    if args.phase == "all":
        phase_queue_download_sync()
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
