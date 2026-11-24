#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared helpers of the real-device video tests (the scripts in the folders next to this file).

Not a test itself. The tests talk to an already running tt_media_server_cpp with
a real runner over HTTP; the helpers here send the requests, wait for job
statuses, read the server log and keep the PASS/FAIL list.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

GEN = "/v1/videos/generations"
UUID0 = "00000000-0000-4000-8000-000000000000"

# set in setup()
BASE = TOKEN = LOG = PAYLOAD = OUTDIR = DLDIR = None
STEPS = 4
T0 = time.time()
results = []  # (ok, name)


def checked_base_url(value):
    """Accept only an http URL of this machine (127.0.0.1 or localhost).

    These tests read the payload directory and the log of the server and look at the
    process that listens on the port, so the server has to run on this machine. The
    URL is rebuilt from its parts, so a path, query or credentials are dropped.
    """
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        raise argparse.ArgumentTypeError(
            f"{value!r}: only http://127.0.0.1:<port> or http://localhost:<port> is accepted"
        )
    return f"http://{parsed.netloc}"


def p(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def note(ok, name, detail=""):
    results.append((ok, name))
    p("PASS" if ok else "FAIL", name, detail)


def skip(name, why):
    p("SKIP", name, f"({why})")


# ---------------------------------------------------------------- http helpers
def req(
    method, path, body=None, token="default", headers=None, raw_body=None, timeout=120
):
    data = (
        raw_body
        if raw_body is not None
        else (None if body is None else json.dumps(body).encode())
    )
    r = urllib.request.Request(BASE + path, data=data, method=method)
    tok = TOKEN if token == "default" else token
    if tok:
        r.add_header("authorization", f"Bearer {tok}")
    if data is not None:
        r.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        r.add_header(k, v)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def jreq(*a, **k):
    st, _, raw = req(*a, **k)
    try:
        return st, (json.loads(raw) if raw.strip() else {})
    except Exception:
        return st, {"raw": raw[:200].decode(errors="replace")}


def err_message(j):
    e = j.get("error")
    if isinstance(e, dict):
        return e.get("message", "")
    return str(e) if e else (j.get("detail") or j.get("raw") or "")


def status(jid):
    st, j = jreq("GET", f"{GEN}/{jid}")
    return j.get("status") if st == 200 else f"http{st}"


def wait(jid, wanted, timeout):
    end = time.time() + timeout
    last = None
    while time.time() < end:
        s = status(jid)
        if s != last:
            p(f"  job {jid[:8]} status={s}")
            last = s
        if s in wanted:
            return True
        time.sleep(1)
    return False


def submit(prompt, seed=None, steps="default"):
    body = {"prompt": prompt}
    n = STEPS if steps == "default" else steps
    if n is not None:
        body["num_inference_steps"] = n
    if seed is not None:
        body["seed"] = seed
    st, j = jreq("POST", GEN, body)
    p(f"submit '{prompt}' -> http{st} id={j.get('id')} status={j.get('status')}")
    return st, j


# ---------------------------------------------------------------- local helpers
def log_pos():
    return os.path.getsize(LOG) if LOG and os.path.exists(LOG) else 0


def log_lines(pos, pattern):
    if not LOG or not os.path.exists(LOG):
        return []
    with open(LOG, "rb") as f:
        f.seek(pos)
        text = f.read().decode(errors="replace")
    return [ln[:220] for ln in text.splitlines() if pattern in ln]


def ls(d):
    return sorted(os.listdir(d)) if d and os.path.isdir(d) else []


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def ffprobe(path):
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"error": out.stderr[:200]}


def markers():
    return [f for f in ls(PAYLOAD) if f.startswith("cancel-")]


# ---------------------------------------------------------------- setup / summary
def add_common_args(ap):
    ap.add_argument(
        "--base-url", default="http://127.0.0.1:8000", type=checked_base_url
    )
    ap.add_argument(
        "--api-key", default=os.environ.get("VIDEO_API_KEY", "your-secret-key")
    )
    ap.add_argument(
        "--steps", type=int, default=4, help="num_inference_steps of the jobs (4-50)"
    )
    ap.add_argument(
        "--server-log",
        default="",
        help="server stdout/stderr log; enables the log checks (else they are SKIP)",
    )
    ap.add_argument(
        "--payload-dir",
        default="",
        help="/tmp/tt-video-ipc-<pid> (default: newest); enables the marker checks",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="TT_VIDEO_OUTPUT_DIR of the server; enables the leftover-mp4 check",
    )
    ap.add_argument(
        "--download-dir",
        default="",
        help="where downloaded mp4 files are kept (default: new temp dir)",
    )


def find_payload_dir(base_url):
    """Payload directory of the server that listens at base_url.

    The server keeps its request and response files in /tmp/tt-video-ipc-<pid>.
    The pid is taken from the process that listens on the port of base_url, so a
    second server on the same machine (for example a mock server of another test)
    is not picked by mistake. Falls back to the newest such directory.
    """
    port = urllib.parse.urlparse(base_url).port or 80
    try:
        out = subprocess.run(
            ["ss", "-ltnpH"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            if re.search(rf":{port}\s", line):
                m = re.search(r"pid=(\d+)", line)
                if m and os.path.isdir(f"/tmp/tt-video-ipc-{m.group(1)}"):
                    return f"/tmp/tt-video-ipc-{m.group(1)}"
    except (OSError, subprocess.SubprocessError):
        pass
    found = sorted(glob.glob("/tmp/tt-video-ipc-*"), key=os.path.getmtime)
    return found[-1] if found else None


def setup(args):
    """Fill the shared state from the arguments. Returns False if the server is not ready."""
    global BASE, TOKEN, LOG, PAYLOAD, OUTDIR, DLDIR, STEPS
    BASE = args.base_url.rstrip("/")
    TOKEN = args.api_key
    STEPS = args.steps
    LOG = args.server_log or None
    OUTDIR = args.output_dir or None
    PAYLOAD = args.payload_dir or None
    if not PAYLOAD:
        PAYLOAD = find_payload_dir(BASE)
    DLDIR = args.download_dir or tempfile.mkdtemp(prefix="tt-video-real-")
    os.makedirs(DLDIR, exist_ok=True)
    st, lv = jreq("GET", "/tt-liveness", token=None)
    if not (st == 200 and lv.get("model_ready") is True):
        print(f"server is not ready at {BASE}: http{st} {str(lv)[:120]}")
        return False
    p(
        f"server: {BASE} | payload: {PAYLOAD} | output: {OUTDIR} | log: {LOG} | downloads: {DLDIR}"
    )
    p(f"ffprobe: {bool(shutil.which('ffprobe'))} | steps: {STEPS}")
    return True


def finish():
    """Print the summary, return the process exit code."""
    p("=== SUMMARY ===")
    for ok, n in results:
        print(("PASS " if ok else "FAIL ") + n)
    passed = sum(1 for ok, _ in results if ok)
    failed = len(results) - passed
    print(f"{passed} PASS, {failed} FAIL", flush=True)
    return 0 if failed == 0 else 1
