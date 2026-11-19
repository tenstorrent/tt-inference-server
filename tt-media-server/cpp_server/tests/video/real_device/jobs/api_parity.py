#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sends the same probes to a video server and prints the answers, one line per probe.

Works against the C++ server and the Python tt-media-server (same routes), so the two
outputs can be compared with `diff`. It asserts nothing: it records behaviour.

Report: docs/tests/2026-11-14-video-real-device-extended-checks.md

    python3 tests/video/real_device/jobs/api_parity.py --base-url http://127.0.0.1:8000 \
        --api-key <token> > python.txt

Needs a ready server with a device (it runs two 4-step jobs and one synchronous request).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402

ROUTES = [
    ("POST", c.GEN, {"prompt": "x", "num_inference_steps": 4}),
    ("GET", f"{c.GEN}/{c.UUID0}", None),
    ("GET", "/v1/videos/jobs", None),
    ("GET", f"{c.GEN}/{c.UUID0}/download", None),
    ("POST", f"{c.GEN}/{c.UUID0}/cancel", None),
    ("DELETE", f"{c.GEN}/{c.UUID0}", None),
]


def probe(name, value):
    print(f"PROBE {name:<58} | {value}", flush=True)


def short(j):
    if isinstance(j, dict):
        return {k: j[k] for k in sorted(j)}
    return j


def err_shape(j):
    return sorted(j.keys()) if isinstance(j, dict) else type(j).__name__


def main():
    ap = argparse.ArgumentParser()
    c.add_common_args(ap)
    args = ap.parse_args()
    if not c.setup(args):
        return 2
    print("---- auth")
    for m, path, body in ROUTES:
        st, _, _ = c.req(m, path, body, token=None)
        probe(f"no token {m} {path.replace(c.UUID0, '<id>')}", st)
        st, _, _ = c.req(m, path, body, token="wrong-token")
        probe(f"wrong token {m} {path.replace(c.UUID0, '<id>')}", st)

    print("---- validation of POST")
    for label, body in (
        ("empty prompt", {"prompt": ""}),
        ("blank prompt", {"prompt": "  "}),
        ("missing prompt", {}),
        ("steps 3", {"prompt": "x", "num_inference_steps": 3}),
        ("steps 51", {"prompt": "x", "num_inference_steps": 51}),
        ("steps as string", {"prompt": "x", "num_inference_steps": "abc"}),
        ("prompt as number", {"prompt": 5}),
        ("height 0", {"prompt": "x", "height": 0}),
        ("fps -1", {"prompt": "x", "fps": -1}),
        ("seed as string", {"prompt": "x", "seed": "abc"}),
    ):
        st, j = c.jreq("POST", c.GEN, body)
        probe(f"POST {label}", f"{st} keys={err_shape(j)}")
        if st == 202:
            c.jreq("POST", f"{c.GEN}/{j['id']}/cancel")
    st, _, _ = c.req("POST", c.GEN, raw_body=b"{not json")
    probe("POST broken json", st)

    print("---- unknown or malformed ids")
    for m, suffix in (("GET", ""), ("GET", "/download"), ("POST", "/cancel"), ("DELETE", "")):
        for label, jid in (
            ("unknown uuid", c.UUID0),
            ("not a uuid", "not-a-uuid"),
            ("number", "12345"),
        ):
            st, j = c.jreq(m, f"{c.GEN}/{jid}{suffix}")
            probe(f"{m} {label} {suffix or '/<id>'}", f"{st} keys={err_shape(j)}")

    print("---- a job through its life")
    st, j = c.jreq("POST", c.GEN, {"prompt": "parity probe", "num_inference_steps": 4, "seed": 1})
    jid = j.get("id")
    probe("POST create status", st)
    probe("create body keys", sorted(j.keys()))
    probe("create body status", j.get("status"))
    probe("create body job_type", j.get("job_type"))
    probe("create body model", j.get("model"))
    probe("create body request_parameters keys", sorted((j.get("request_parameters") or {}).keys()))
    st2, j2 = c.jreq(
        "POST", c.GEN, {"prompt": "queued behind", "num_inference_steps": 4, "seed": 2}
    )
    id2 = j2.get("id")
    st, _, _ = c.req("GET", f"{c.GEN}/{id2}/download")
    probe("download of a queued job", st)
    st, _ = c.jreq("DELETE", f"{c.GEN}/{id2}")
    probe("DELETE of a queued job", st)
    c.wait(jid, ("in_progress",), 60)
    st, _, _ = c.req("GET", f"{c.GEN}/{jid}/download")
    probe("download of a running job", st)
    st, _ = c.jreq("DELETE", f"{c.GEN}/{jid}")
    probe("DELETE of a running job", st)
    st, cj = c.jreq("POST", f"{c.GEN}/{id2}/cancel")
    probe("cancel of a queued job", f"{st} status={cj.get('status')}")
    st, cj = c.jreq("POST", f"{c.GEN}/{id2}/cancel")
    probe("second cancel of the same job", f"{st} status={cj.get('status')}")
    st, g = c.jreq("GET", f"{c.GEN}/{id2}")
    probe("GET of the cancelled job", f"{st} status={g.get('status')} keys={sorted(g.keys())}")
    st, d = c.jreq("DELETE", f"{c.GEN}/{id2}")
    probe("DELETE of the cancelled job", f"{st} body={d}")
    c.wait(jid, ("completed", "failed"), 600)
    st, g = c.jreq("GET", f"{c.GEN}/{jid}")
    probe("completed job keys", sorted(g.keys()))
    probe("completed job status", g.get("status"))
    st, _, body = c.req("GET", f"{c.GEN}/{jid}/download")
    st, hdr, body = c.req("GET", f"{c.GEN}/{jid}/download")
    probe("download status", st)
    for h in (
        "content-type",
        "content-disposition",
        "accept-ranges",
        "content-length",
        "content-range",
        "etag",
        "last-modified",
    ):
        v = hdr.get(h)
        if h == "content-disposition" and v:
            v = v.split("filename=")[0] + "filename=<name>"
        if h in ("etag", "last-modified"):
            v = "present" if v else None
        if h == "content-length":
            v = "matches body" if v == str(len(body)) else v
        probe(f"download header {h}", v)
    st, hdr, part = c.req("GET", f"{c.GEN}/{jid}/download", headers={"Range": "bytes=0-99"})
    probe(
        "download Range bytes=0-99",
        f"{st} len={len(part)} content-range={hdr.get('content-range')}",
    )
    st, hdr, part = c.req(
        "GET", f"{c.GEN}/{jid}/download", headers={"Range": f"bytes={len(body)}-"}
    )
    probe("download Range past the end", st)
    st, hdr, _ = c.req("HEAD", f"{c.GEN}/{jid}/download")
    probe("download HEAD", st)
    st, lst = c.jreq("GET", "/v1/videos/jobs")
    probe("list is a", type(lst).__name__)
    probe(
        "list contains the job",
        any(x.get("id") == jid for x in (lst if isinstance(lst, list) else lst.get("jobs", []))),
    )
    st, d = c.jreq("DELETE", f"{c.GEN}/{jid}")
    probe("DELETE of the completed job", f"{st} body={d}")
    st, _ = c.jreq("GET", f"{c.GEN}/{jid}")
    probe("GET after DELETE", st)

    print("---- synchronous mode")
    t0 = time.time()
    st, hdr, body = c.req(
        "POST",
        f"{c.GEN}?async=false",
        {"prompt": "sync probe", "num_inference_steps": 4, "seed": 3},
        timeout=900,
    )
    probe("POST ?async=false status", st)
    probe("sync content-type", hdr.get("content-type"))
    probe("sync x-generation-time present", hdr.get("x-generation-time") is not None)
    probe("sync body starts with mp4 ftyp", body[4:8] == b"ftyp")
    st, _ = c.jreq("POST", f"{c.GEN}?async=banana", {"prompt": "x", "num_inference_steps": 4})
    probe("POST ?async=banana", st)
    st, jl = c.jreq("GET", "/v1/videos/jobs")
    probe(
        "sync request left a job record",
        any(
            (x.get("request_parameters") or {}).get("prompt") == "sync probe"
            for x in (jl if isinstance(jl, list) else [])
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
