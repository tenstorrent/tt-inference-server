#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Cancel on a REAL video server (tt_media_server_cpp with a device runner).

Report: docs/tests/2026-11-14-video-cancel-real-server.md

The server must already run with a real runner and report model_ready (start
command and device reset notes: tests/video/real_device/README.md). Jobs use
num_inference_steps=--steps (default 4); one generation takes about 80 s on a T3K.

Job A completes, job B is cancelled while queued (the worker must skip it),
job D is cancelled while running (cancelling -> cancelled, result discarded),
job E proves the device still works. 4 generations, about 4 minutes. The
cancel/CLAIMED race has its own test: real_device/cancel/cancel_race.py.

Optional checks need access to the server host (--server-log, --payload-dir,
--output-dir); without them they are reported as SKIP, not as failures.

Run (from cpp_server):
  python3 tests/video/real_device/cancel/cancel_queue_and_running.py --server-log /path/to/server.log \\
      --output-dir /path/to/TT_VIDEO_OUTPUT_DIR
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c
from common import (
    GEN,
    jreq,
    log_lines,
    log_pos,
    ls,
    markers,
    note,
    p,
    req,
    skip,
    status,
    submit,
    wait,
)


def phase_cancel():
    p(
        "=== CANCEL: A runs normally, B is cancelled in the queue, D is cancelled while running, E checks the device ==="
    )
    pos = log_pos()
    st, a = submit("a red ball rolling on grass", seed=1)
    A = a.get("id")
    note(st == 202, "A submit 202")
    note(wait(A, ("in_progress",), 600), "A in_progress")
    st, b = submit("a blue cube", seed=2)
    B = b.get("id")
    note(st == 202 and status(B) == "queued", "B is queued while A runs")
    st, j = jreq("POST", f"{GEN}/{B}/cancel")
    note(
        st == 200 and j.get("status") == "cancelled",
        "cancel B -> cancelled at once",
        f"http{st} status={j.get('status')}",
    )
    if c.PAYLOAD:
        p("  payload dir after the cancel:", ls(c.PAYLOAD))
        note(len(markers()) >= 1, "cancel marker cancel-<id> exists in the payload dir")
    else:
        skip("cancel marker cancel-<id> exists", "no --payload-dir")
    st, _ = jreq("POST", f"{GEN}/{B}/cancel")
    note(st == 404, "second cancel of B is 404", f"http{st}")

    p("waiting for A to finish (generation on the device)...")
    wait(A, ("completed", "failed", "cancelled"), 3600)
    note(status(A) == "completed", "A completed", f"status={status(A)}")
    st, _, raw = req("GET", f"{GEN}/{A}/download")
    note(
        st == 200 and len(raw) > 1000 and raw[4:8] == b"ftyp",
        "A download is an mp4",
        f"http{st} bytes={len(raw)}",
    )
    time.sleep(3)  # worker sada vadi B iz reda
    if c.LOG:
        # Worker process: the worker pops B and logs "skipping cancelled task ... before
        # claim". In-process mode (one DEVICE_IDS group) has no worker: the cancel removes
        # B from the in-memory queue and only the service logs "removed from the queue".
        lines = log_lines(pos, "skipping cancelled task")
        removed = log_lines(pos, "removed from the queue")
        note(
            any("before claim" in ln for ln in lines) or len(removed) >= 1,
            "B was skipped before it was claimed (worker log, or removed from the in-process queue)",
            str((lines or removed)[:1]),
        )
        req_lines = log_lines(pos, "Cancel requested for task")
        note(
            any("is now cancelled" in ln for ln in req_lines),
            "server log: 'Cancel requested ... is now cancelled'",
            str(req_lines[:1]),
        )
    else:
        skip("log: skipping cancelled task / Cancel requested", "no --server-log")
    st, j = jreq("GET", f"{GEN}/{B}")
    note(
        j.get("status") == "cancelled" and "error" not in j,
        "B cancelled without an error field",
    )
    if c.PAYLOAD:
        note(
            ls(c.PAYLOAD) == [],
            "payload dir empty (markers and files removed)",
            str(ls(c.PAYLOAD)),
        )

    p("--- cancel of a job that is generating ---")
    before = set(ls(c.OUTDIR))
    pos = log_pos()
    st, d = submit("a green forest at dawn", seed=3)
    D = d.get("id")
    note(wait(D, ("in_progress",), 600), "D in_progress")
    time.sleep(5)
    st, j = jreq("POST", f"{GEN}/{D}/cancel")
    note(
        st == 200 and j.get("status") == "cancelling",
        "cancel D -> cancelling",
        f"http{st} status={j.get('status')}",
    )
    t1 = time.time()
    wait(D, ("cancelled", "completed", "failed"), 3600)
    note(
        status(D) == "cancelled",
        "D ends as cancelled",
        f"status={status(D)} (generation took another {time.time() - t1:.0f}s)",
    )
    st, _, _ = req("GET", f"{GEN}/{D}/download")
    note(st in (404, 409), "D download is not available", f"http{st}")
    time.sleep(3)
    if c.OUTDIR:
        new = set(ls(c.OUTDIR)) - before
        note(
            not new,
            "no mp4 of the cancelled job left in the output dir",
            f"new files={sorted(new)}",
        )
    else:
        skip("output dir without an mp4 of the cancelled job", "no --output-dir")
    if c.PAYLOAD:
        note(not markers(), "no leftover marker", str(ls(c.PAYLOAD)))
    if c.LOG:
        req_lines = log_lines(pos, "Cancel requested for task")
        note(
            any("is now cancelling" in ln for ln in req_lines),
            "server log: 'Cancel requested ... is now cancelling'",
            str(req_lines[:1]),
        )

    p("--- the device works after the cancel ---")
    st, e = submit("a yellow bird flying", seed=4)
    E = e.get("id")
    wait(E, ("completed", "failed", "cancelled"), 3600)
    note(
        status(E) == "completed", "E completed after the cancel", f"status={status(E)}"
    )
    st, _, raw = req("GET", f"{GEN}/{E}/download")
    note(
        st == 200 and raw[4:8] == b"ftyp",
        "E download is an mp4",
        f"http{st} bytes={len(raw)}",
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    c.add_common_args(ap)
    args = ap.parse_args()
    if not c.setup(args):
        return 1
    phase_cancel()
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
