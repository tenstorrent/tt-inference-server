#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Cancel/CLAIMED race on a REAL video server.

A cancel that lands after the worker popped the task from the queue but before
it sent CLAIMED. The service still sees the job as "queued", so it finishes the
job as cancelled at once and writes the cancel marker; the worker's second
marker check (after CLAIMED) must then stop the task, and the model must never
run for it.

The window is only microseconds in production, so the server has to be started
with the test hook TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500: the worker then waits
1.5 s between pop and CLAIMED. The hook is not for production; restart the
server without it afterwards (every job would wait 1.5 s). Everything else in
the start command is in tests/video/real_device/README.md. The server must
already be model_ready.

One generation (the job after the race), about 85 s on a T3K. --server-log is
required: the proof is the worker log line "skipping cancelled task N (after
claim)" and the absence of runner lines for that task.

Run (from cpp_server):
  python3 tests/video/real_device/cancel/cancel_race.py --server-log /path/to/server.log
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
    note,
    p,
    skip,
    status,
    submit,
    wait,
)


def phase_cancel_race():
    p(
        "=== CANCEL-RACE: cancel after the task was taken from the queue, before CLAIMED ==="
    )
    if not c.LOG:
        p(
            "FAIL cancel-race needs --server-log (it checks the '(after claim)' line in the worker log)"
        )
        c.results.append((False, "cancel-race: --server-log is required"))
        return
    pos = log_pos()
    st, j = submit("race", seed=5)
    jid = j.get("id")
    note(st == 202, "race: submit 202")
    time.sleep(0.5)
    before = status(jid)
    if before != "queued":
        note(
            False,
            "race: job is still queued while the worker waits (is the hook active?)",
            f"status={before!r}; the server was not started with TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500",
        )
        jreq("POST", f"{GEN}/{jid}/cancel")
        return
    note(True, "race: job is still queued while the worker waits", f"status={before!r}")
    st, j = jreq("POST", f"{GEN}/{jid}/cancel")
    note(
        st == 200 and j.get("status") == "cancelled",
        "race: cancel -> cancelled",
        f"http{st} status={j.get('status')}",
    )
    note(
        wait(jid, ("cancelled",), 15),
        "race: job stays cancelled",
        f"status={status(jid)!r}",
    )
    time.sleep(
        3.0
    )  # dovoljno da se worker probudi, posalje CLAIMED i (ne)pokrene model
    st, final = jreq("GET", f"{GEN}/{jid}")
    note(
        st == 200 and final.get("status") == "cancelled" and "error" not in final,
        "race: cancelled without an error",
        f"status={final.get('status')!r} error={final.get('error')!r}",
    )

    lines = log_lines(pos, "skipping cancelled task")
    p("  skip log lines:", lines)
    after = [ln for ln in lines if "(after claim)" in ln]
    note(
        len(after) == 1,
        "race: worker log: skipped after claim (after claim)",
        f"n={len(after)}",
    )
    task_id = None
    if after:
        try:
            task_id = after[0].split("skipping cancelled task ")[1].split()[0]
        except Exception:
            pass
    if task_id is not None:
        ran = [
            ln
            for ln in log_lines(pos, f"task {task_id} ")
            if "VideoIpcRunner" in ln or "VideoDevice" in ln
        ]
        note(
            not ran,
            "race: the model was not run for that task (no VideoIpcRunner lines)",
            str(ran[:2]),
        )
    else:
        skip(
            "the model was not run for that task",
            "task id could not be read from the log",
        )
    cl = log_lines(pos, "Cancel requested for task")
    p("  cancel log lines:", cl)

    # sljedeci job mora normalno da radi; da je race job pokrenut na cipu,
    # cekao bi se jos jedan cijeli run (oko 80 s)
    t = time.time()
    st, j2 = submit("after race", seed=6)
    j2id = j2.get("id")
    wait(j2id, ("completed", "failed", "cancelled"), 3600)
    el = time.time() - t
    note(
        status(j2id) == "completed",
        "race: the next job completes",
        f"status={status(j2id)!r}",
    )
    note(
        el < 120,
        "race: the next job did not wait behind the race job (< 120 s)",
        f"duration={el:.0f}s",
    )
    if c.PAYLOAD:
        note(ls(c.PAYLOAD) == [], "race: payload dir empty", str(ls(c.PAYLOAD)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    c.add_common_args(ap)
    args = ap.parse_args()
    if not c.setup(args):
        return 1
    phase_cancel_race()
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
