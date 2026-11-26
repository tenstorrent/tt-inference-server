#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""A dropped synchronous request and a deep queue, on a real device.


Run (server must be model_ready):
    python3 tests/video/real_device/jobs/disconnect_and_queue_depth.py \
        --server-log <server log> --output-dir <TT_VIDEO_OUTPUT_DIR>

  1. the client drops a `?async=false` request after 20 s: the generation runs to the end,
     no job record exists, the mp4 stays in the output folder (documented limitation), and
     the next job is not blocked
  2. 30 jobs are queued: all accepted, listed in order, the queued ones are cancelled at
     once, only the running one uses the device, the queue drains
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402


def drop_sync_request(after_s):
    host, port = c.BASE.replace("http://", "").split(":")
    body = json.dumps(
        {"prompt": "dropped sync request", "num_inference_steps": c.STEPS}
    ).encode()
    s = socket.create_connection((host, int(port)), timeout=30)
    s.sendall(
        (
            f"POST {c.GEN}?async=false HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {c.TOKEN}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode()
        + body
    )
    time.sleep(after_s)
    s.close()


def main():
    ap = argparse.ArgumentParser()
    c.add_common_args(ap)
    args = ap.parse_args()
    if not c.setup(args):
        return 2

    c.p("--- 1. the client drops a synchronous request")
    files0 = c.ls(c.OUTDIR) if c.OUTDIR else None
    pos = c.log_pos()
    drop_sync_request(20)
    st, jobs = c.jreq("GET", "/v1/videos/jobs")
    c.p(
        f"  jobs right after the drop: http{st} {len(jobs) if isinstance(jobs, list) else jobs}"
    )
    # the sync task still runs: a normal job queued behind it must wait, then complete
    st, j = c.submit("behind the dropped one", seed=2)
    t0 = time.time()
    ok = c.wait(j["id"], ("completed", "failed"), 600)
    c.note(
        ok and c.status(j["id"]) == "completed",
        "a job queued behind the dropped request completes",
        f"after {time.time() - t0:.0f}s",
    )
    if c.LOG:
        wrote = c.log_lines(pos, " wrote ")
        c.note(
            len(wrote) == 2,
            "the worker finished both tasks (the dropped one was not skipped)",
            f"{len(wrote)} 'wrote' lines",
        )
    else:
        c.skip("worker log check", "no --server-log")
    if c.OUTDIR:
        new = sorted(set(c.ls(c.OUTDIR)) - set(files0))
        c.p("  new files in the output folder:", new)
        c.note(
            len(new) == 1,
            "only the queued job's mp4 is on disk (the dropped sync one is deleted once its result is read)",
            str(new),
        )
    st, jobs = c.jreq("GET", "/v1/videos/jobs")
    mine = [
        x
        for x in (jobs if isinstance(jobs, list) else [])
        if x["request_parameters"].get("prompt") == "dropped sync request"
    ]
    c.note(not mine, "the dropped synchronous request has no job record")

    c.p("--- 2. a deep queue: 30 jobs")
    ids = []
    t0 = time.time()
    for i in range(30):
        st, j = c.jreq(
            "POST",
            c.GEN,
            {"prompt": f"queued {i}", "num_inference_steps": c.STEPS, "seed": i},
        )
        if st != 202:
            c.note(False, f"job {i} accepted", f"http{st}")
            break
        ids.append(j["id"])
    c.note(len(ids) == 30, "30 jobs accepted", f"in {time.time() - t0:.1f}s")
    time.sleep(3)
    st, listing = c.jreq("GET", "/v1/videos/jobs")
    mine = [x for x in listing if x["id"] in set(ids)]
    states = {}
    for x in mine:
        states[x["status"]] = states.get(x["status"], 0) + 1
    c.p("  statuses:", states)
    c.note(
        states.get("in_progress", 0) == 1 and states.get("queued", 0) == 29,
        "one job runs, 29 wait",
        str(states),
    )
    _, lv = c.jreq("GET", "/tt-liveness", token=None)
    c.p("  queue_size in /tt-liveness:", lv.get("queue_size"))
    t0 = time.time()
    for jid in ids[1:]:
        c.jreq("POST", f"{c.GEN}/{jid}/cancel")
    c.p(f"  29 cancels took {time.time() - t0:.1f}s")
    c.note(
        all(c.status(i) == "cancelled" for i in ids[1:]),
        "all 29 queued jobs are cancelled at once",
        str(
            {
                s: [c.status(i) for i in ids[1:]].count(s)
                for s in set(c.status(i) for i in ids[1:])
            }
        ),
    )
    c.wait(ids[0], ("completed", "failed"), 600)
    c.note(
        c.status(ids[0]) == "completed",
        "the running job finishes normally",
        c.status(ids[0]),
    )
    drained = False
    for _ in range(30):
        _, lv = c.jreq("GET", "/tt-liveness", token=None)
        if lv.get("queue_size") == 0:
            drained = True
            break
        time.sleep(2)
    c.note(drained, "the queue drains (queue_size 0)", str(lv.get("queue_size")))
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
