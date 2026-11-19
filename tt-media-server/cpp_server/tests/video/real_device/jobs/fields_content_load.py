#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request fields, video content, Range resume and server responsiveness on a real device.

Report: docs/tests/2026-11-14-video-real-device-extended-checks.md

Run (server must be model_ready; the output folder and log are optional):
    python3 tests/video/real_device/jobs/fields_content_load.py \
        --server-log <server log> --output-dir <TT_VIDEO_OUTPUT_DIR> [--long-steps 50]

What it does (jobs run one after another on the device):
  1. validation echo: POST responses for all fields, defaults, limits (422), without
     using the device (a blocker job runs first and the echo jobs are cancelled queued)
  2. a job with a Unicode and a very long prompt
  3. the mp4: codec, size, frame count, moov before mdat, "already faststart" in the log
  4. Range: Accept-Ranges, Content-Length, a download resumed with `bytes=N-`
  5. while a job generates: latency of GET, of GET /jobs, parallel downloads
  6. step limits: --long-steps (default 50) runs once and is timed
  7. memory of the server and the worker before and after
"""

import argparse
import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as c  # noqa: E402


def listening_pid(base_url):
    """Pid of the server: the payload folder is named tt-video-ipc-<pid>."""
    m = re.search(r"tt-video-ipc-(\d+)", c.PAYLOAD or "")
    return int(m.group(1)) if m else None


def rss_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def children(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def mem_snapshot(server_pid):
    snap = {"server": rss_mb(server_pid)}
    for i, ch in enumerate(children(server_pid)):
        snap[f"worker{i}"] = rss_mb(ch)
    return snap


def moov_before_mdat(path):
    """True when the moov box comes before the mdat box (faststart)."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    order = []
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8].decode("latin1")
        order.append(kind)
        if size == 1:
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
        if size == 0:
            break
        pos += size
    return (
        order.index("moov") < order.index("mdat") if "moov" in order and "mdat" in order else None
    ), order


def main():
    ap = argparse.ArgumentParser()
    c.add_common_args(ap)
    ap.add_argument("--long-steps", type=int, default=50, help="steps of the long job (0 = skip)")
    args = ap.parse_args()
    if not c.setup(args):
        return 2
    server_pid = listening_pid(c.BASE)
    mem0 = mem_snapshot(server_pid) if server_pid else {}
    c.p("memory before:", mem0)

    # ------------------------------------------------------------------ 1. validation echo
    c.p("--- 1. validation echo")
    _, blocker = c.submit("blocker for the echo checks", seed=1)
    c.wait(blocker["id"], ("in_progress",), 60)
    st, j = c.jreq("POST", c.GEN, {"prompt": "only a prompt"})
    c.note(
        st == 202 and j["request_parameters"].get("num_inference_steps") == 20,
        "default request: job echoes num_inference_steps=20",
        f"http{st} {j.get('request_parameters')}",
    )
    echo_ids = [j["id"]]
    full = {
        "prompt": "all fields",
        "negative_prompt": "blurry",
        "num_inference_steps": 7,
        "seed": 42,
        "aspect_ratio": "16:9",
        "height": 480,
        "width": 832,
        "fps": 16,
        "duration": 5,
        "num_frames": 81,
    }
    st, j = c.jreq("POST", c.GEN, full)
    rp = j.get("request_parameters", {})
    mismatch = {k: (v, rp.get(k)) for k, v in full.items() if k != "prompt" and rp.get(k) != v}
    c.note(
        st == 202 and not mismatch,
        "all optional fields are accepted and echoed",
        f"http{st} mismatch={mismatch}",
    )
    if j.get("id"):
        echo_ids.append(j["id"])
    for steps, ok in ((3, False), (51, False), (4, True), (50, True)):
        st, j = c.jreq("POST", c.GEN, {"prompt": "limits", "num_inference_steps": steps})
        c.note(
            (st == 202) == ok,
            f"num_inference_steps={steps}: {'accepted' if ok else '422'}",
            f"http{st}",
        )
        if st == 202:
            echo_ids.append(j["id"])
    for body, why in (
        ({"prompt": ""}, "empty prompt"),
        ({"prompt": "   "}, "blank prompt"),
        ({}, "no prompt"),
        ({"prompt": "x", "height": 0}, "height 0"),
        ({"prompt": "x", "fps": -1}, "negative fps"),
        ({"prompt": 5}, "prompt not a string"),
        ({"prompt": "x", "seed": "abc"}, "seed not a number"),
    ):
        st, j = c.jreq("POST", c.GEN, body)
        c.note(st == 422, f"{why}: 422", f"http{st} {c.err_message(j)[:60]}")
    st, _, _ = c.req("POST", c.GEN, raw_body=b"{not json")
    c.note(st == 422, "broken JSON: 422", f"http{st}")
    for jid in echo_ids:
        c.jreq("POST", f"{c.GEN}/{jid}/cancel")
    c.wait(blocker["id"], ("completed", "failed", "cancelled"), 300)
    c.note(
        all(c.status(i) == "cancelled" for i in echo_ids),
        "the echo jobs were cancelled in the queue (no device time used)",
        str([c.status(i) for i in echo_ids]),
    )

    # ------------------------------------------------------------------ 2. unicode + long prompt
    c.p("--- 2. unicode and a very long prompt")
    prompt = "Lisica trči kroz snijeg: čćžšđ ЖЗИ 日本語 🦊 " + ("snow " * 4000)
    pos = c.log_pos()
    st, j = c.submit(prompt, seed=3)
    uid = j["id"]
    c.note(st == 202, "a Unicode prompt of about 20000 characters is accepted", f"http{st}")
    ok = c.wait(uid, ("completed", "failed"), 600)
    c.note(ok and c.status(uid) == "completed", "it generates a video", f"status={c.status(uid)}")
    st, jj = c.jreq("GET", f"{c.GEN}/{uid}")
    c.note(
        jj.get("request_parameters", {}).get("prompt") == prompt,
        "the job echoes the prompt unchanged (Unicode and length)",
    )

    # ------------------------------------------------------------------ 3. the mp4
    c.p("--- 3. the mp4")
    st, hdr, body = c.req("GET", f"{c.GEN}/{uid}/download")
    path = os.path.join(c.DLDIR, "unicode.mp4")
    open(path, "wb").write(body)
    c.note(
        st == 200 and hdr.get("content-type", "").startswith("video/mp4"),
        "download: 200 video/mp4",
        f"http{st} {hdr.get('content-type')} {len(body)} B",
    )
    info = c.ffprobe(path)
    stream = (info or {}).get("streams", [{}])[0]
    c.note(
        stream.get("codec_name") == "h264" and stream.get("nb_frames") == "81",
        "h264, 81 frames",
        str(stream),
    )
    c.note(
        (stream.get("width"), stream.get("height")) == (832, 480),
        "832x480",
        str((stream.get("width"), stream.get("height"))),
    )
    fs, order = moov_before_mdat(path)
    c.note(fs is True, "moov before mdat (faststart)", str(order))
    lines = c.log_lines(pos, "already faststart")
    if c.LOG:
        c.note(
            bool(lines),
            'the log says "already faststart" (no remux was needed)',
            lines[0][-70:] if lines else "",
        )
    else:
        c.skip("faststart log line", "no --server-log")

    # ------------------------------------------------------------------ 4. Range
    c.p("--- 4. Range")
    size = len(body)
    st, hdr, part = c.req("GET", f"{c.GEN}/{uid}/download", headers={"Range": "bytes=0-99"})
    c.note(
        st == 206 and len(part) == 100 and hdr.get("content-range") == f"bytes 0-99/{size}",
        "bytes=0-99: 206, 100 bytes, Content-Range",
        f"{st} {hdr.get('content-range')}",
    )
    c.note(
        hdr.get("accept-ranges") == "bytes", "Accept-Ranges: bytes", str(hdr.get("accept-ranges"))
    )
    cut = int(size * 0.4)
    _, _, first = c.req("GET", f"{c.GEN}/{uid}/download", headers={"Range": f"bytes=0-{cut - 1}"})
    st, hdr, rest = c.req("GET", f"{c.GEN}/{uid}/download", headers={"Range": f"bytes={cut}-"})
    c.note(
        st == 206 and first + rest == body,
        "an interrupted download resumed with bytes=N- gives the same file",
        f"{len(first)}+{len(rest)}={len(first) + len(rest)} of {size}",
    )
    st, hdr, _ = c.req("GET", f"{c.GEN}/{uid}/download")
    c.note(
        hdr.get("content-length") == str(size),
        "Content-Length equals the file size",
        str(hdr.get("content-length")),
    )
    st, hdr, _ = c.req("HEAD", f"{c.GEN}/{uid}/download")
    c.p(f"  HEAD -> http{st} content-length={hdr.get('content-length')}")
    for rng, want in (
        ("bytes=-50", 206),
        (f"bytes={size}-", 416),
        ("bytes=5-2", None),
        ("items=0-1", None),
    ):
        st, hdr, _ = c.req("GET", f"{c.GEN}/{uid}/download", headers={"Range": rng})
        c.p(f"  Range '{rng}' -> http{st}")
        if want:
            c.note(st == want, f"Range '{rng}': {want}", f"http{st}")
        elif rng.startswith("bytes=5-2"):
            c.note(
                st in (200, 416),
                "an invalid range (last < first) is ignored (200) or rejected (416)",
                f"http{st}",
            )

    # ------------------------------------------------------------------ 5. responsiveness
    c.p("--- 5. responsiveness while a job generates")
    st, j = c.submit("load test", seed=5)
    lid = j["id"]
    c.wait(lid, ("in_progress",), 60)
    lat_get, lat_list = [], []
    end = time.time() + 45

    def dl():
        s, _, b = c.req("GET", f"{c.GEN}/{uid}/download")
        return s == 200 and b == body

    with cf.ThreadPoolExecutor(16) as ex:
        dls = [ex.submit(dl) for _ in range(48)]
        while time.time() < end:
            t = time.time()
            c.jreq("GET", f"{c.GEN}/{lid}")
            lat_get.append(time.time() - t)
            t = time.time()
            c.jreq("GET", "/v1/videos/jobs")
            lat_list.append(time.time() - t)
            time.sleep(0.25)
        done = [f.result() for f in dls]
    lat_get.sort()
    lat_list.sort()
    c.p(
        f"  GET job: median {lat_get[len(lat_get) // 2] * 1000:.0f} ms, max {lat_get[-1] * 1000:.0f} ms ({len(lat_get)} samples)"
    )
    c.p(
        f"  GET /jobs: median {lat_list[len(lat_list) // 2] * 1000:.0f} ms, max {lat_list[-1] * 1000:.0f} ms"
    )
    c.note(all(done), "48 parallel downloads during generation: identical bytes")
    c.note(
        lat_get[-1] < 2.0,
        "GET of a job stays fast while the device generates",
        f"max {lat_get[-1] * 1000:.0f} ms",
    )
    t_start = time.time()
    c.wait(lid, ("completed", "failed"), 600)
    c.note(c.status(lid) == "completed", "the job under load completes", f"status={c.status(lid)}")

    # ------------------------------------------------------------------ 6. long job
    if args.long_steps:
        c.p(f"--- 6. {args.long_steps} steps")
        t0 = time.time()
        st, j = c.submit("a long generation", seed=9, steps=args.long_steps)
        ok = c.wait(j["id"], ("completed", "failed"), 3600)
        took = time.time() - t0
        c.note(
            ok and c.status(j["id"]) == "completed",
            f"{args.long_steps} steps complete",
            f"{took:.0f}s",
        )
        c.p(f"  {args.long_steps} steps took {took:.0f} s")

    # ------------------------------------------------------------------ 7. memory
    c.p("--- 7. memory")
    mem1 = mem_snapshot(server_pid) if server_pid else {}
    c.p("memory before:", mem0)
    c.p("memory after: ", mem1)
    for k in mem0:
        if mem0[k] and mem1.get(k):
            c.note(
                mem1[k] < mem0[k] * 1.5 + 200,
                f"{k} RSS did not grow without bound",
                f"{mem0[k]:.0f} MB -> {mem1[k]:.0f} MB",
            )
    if c.OUTDIR:
        c.p(f"  output folder: {len(c.ls(c.OUTDIR))} files")
    return c.finish()


if __name__ == "__main__":
    sys.exit(main())
