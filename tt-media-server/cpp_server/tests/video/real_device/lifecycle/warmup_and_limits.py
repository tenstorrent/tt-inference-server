#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Warmup answers and queue limits with the real runner (starts the server itself).


    python3 tests/video/real_device/lifecycle/warmup_and_limits.py \
        --tt-metal <tt-metal> --media-server <path to tt-media-server> \
        --dit-cache <tt_dit weight cache> --api-key <token>

  1. while the model warms up: /tt-liveness `model_ready=false`, POST answers 503 with
     Retry-After (the auth check still comes first)
  2. MAX_QUEUE_SIZE=3: the fourth active job is rejected with 429
  3. MAX_JOBS=5: the sixth job record is rejected with 503 (finished jobs count)
"""

import argparse
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import restart_cases as rc  # noqa: E402


def post_raw(srv, token="default"):
    req = urllib.request.Request(
        srv.base + rc.GEN, data=b'{"prompt":"x","num_inference_steps":4}', method="POST"
    )
    req.add_header("content-type", "application/json")
    tok = srv.args.api_key if token == "default" else token
    if tok:
        req.add_header("authorization", f"Bearer {tok}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except OSError:
        return None, {}, b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-metal", required=True)
    ap.add_argument("--media-server", required=True)
    ap.add_argument("--dit-cache", required=True)
    ap.add_argument("--api-key", default="your-secret-key")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="tt-video-limits-real.")
    out = os.path.join(work, "videos")
    os.makedirs(out)

    rc.p("=== 1. warmup answers")
    srv = rc.Server(
        args,
        out,
        os.path.join(work, "warmup.log"),
        {"MAX_QUEUE_SIZE": "3", "MAX_JOBS": "5"},
    ).start()
    seen = {}
    t0 = time.time()
    while time.time() - t0 < 600:
        if srv.proc.poll() is not None:
            break
        st, j = srv.call("GET", "/tt-liveness", token=None)
        if st == 200 and j.get("model_ready") is True:
            break
        pst, hdr, body = post_raw(srv)
        if pst is not None:
            seen.setdefault(
                (
                    "authorized POST",
                    pst,
                    hdr.get("Retry-After") or hdr.get("retry-after"),
                ),
                0,
            )
            seen[
                (
                    "authorized POST",
                    pst,
                    hdr.get("Retry-After") or hdr.get("retry-after"),
                )
            ] += 1
            ust, _, _ = post_raw(srv, token=None)
            seen.setdefault(("POST without token", ust, None), 0)
            seen[("POST without token", ust, None)] += 1
            seen.setdefault(("liveness", st, j.get("model_ready")), 0)
            seen[("liveness", st, j.get("model_ready"))] += 1
        time.sleep(3)
    rc.p("  answers during the warmup (what, http status, Retry-After): count")
    for k, v in sorted(seen.items(), key=str):
        rc.p("   ", k, v)
    post_503 = [k for k in seen if k[0] == "authorized POST" and k[1] == 503]
    rc.note(bool(post_503), "POST during the warmup answers 503", str(post_503))
    rc.note(
        any(k[2] == "30" for k in post_503),
        "503 carries Retry-After: 30",
        str(post_503),
    )
    rc.note(
        any(k[0] == "POST without token" and k[1] == 401 for k in seen),
        "the auth check comes first (401 without a token during the warmup)",
    )
    rc.note(
        any(k[0] == "liveness" and k[2] is False for k in seen),
        "/tt-liveness reports model_ready=false during the warmup",
    )
    if not srv.wait_ready(300):
        rc.note(False, "server ready", srv.log_text()[-300:])
        srv.kill_all()
        return rc.finish()

    rc.p("=== 2. MAX_QUEUE_SIZE=3 -> 429")
    ids, codes = [], []
    for i in range(5):
        st, j = srv.call("POST", rc.GEN, {"prompt": f"q{i}", "num_inference_steps": 4})
        codes.append(st)
        if st == 202:
            ids.append(j["id"])
    rc.p("  POST statuses:", codes)
    rc.note(
        codes[:3] == [202, 202, 202] and codes[3:] == [429, 429],
        "3 active jobs are accepted, the 4th and 5th get 429",
        str(codes),
    )
    for jid in ids[1:]:
        srv.call("POST", f"{rc.GEN}/{jid}/cancel")
    srv.wait_job(ids[0], ("completed", "failed"), 300)
    rc.note(srv.job(ids[0]).get("status") == "completed", "the running job completes")

    rc.p("=== 3. MAX_JOBS=5 -> 503")
    codes = []
    for i in range(5):
        st, j = srv.call("POST", rc.GEN, {"prompt": f"m{i}", "num_inference_steps": 4})
        codes.append(st)
        if st == 202:
            srv.call("POST", f"{rc.GEN}/{j['id']}/cancel")
    rc.p("  POST statuses (3 records already exist, all terminal or cancelled):", codes)
    rc.note(
        503 in codes and codes[0] == 202,
        "when the job table is full (MAX_JOBS) POST answers 503",
        str(codes),
    )
    st, jl = srv.call("GET", "/v1/videos/jobs")
    rc.note(
        isinstance(jl, list) and len(jl) <= 5,
        "the table never holds more than MAX_JOBS records",
        str(len(jl) if isinstance(jl, list) else jl),
    )
    srv.terminate()
    return rc.finish()


if __name__ == "__main__":
    sys.exit(main())
