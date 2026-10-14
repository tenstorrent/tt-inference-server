#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""MiniMax-H3 FL2VA latency benchmark against a running tt-media-server endpoint.

Sends FL2VA (first/last keyframe + prompt -> video with audio) requests one at a time, for 5,
10 and 15 second clips, and reports end-to-end latency per duration. Python 3.9+, standard
library only. Nothing else to install or download: by default the keyframe is generated here.

    export H3_URL=http://<host>:8000  H3_API_KEY=<key>
    python3 scripts/minimax_h3_fl2va_bench.py                          # 1 warmup + 3 timed runs per duration
    python3 scripts/minimax_h3_fl2va_bench.py --runs 5 --durations 5,10 --first-frame my.jpg --save-videos

Each request is checked: HTTP success, the MP4 has a video and an audio track, and its length
is within 0.6 s of what was asked. Results go to results/<timestamp>/ (summary.md,
summary.json, results.csv) and a summary table is printed at the end.

Timings
  e2e_s      POST sent -> last byte of the MP4 received. Includes upload, queueing, generation,
             encoding and download. This is the headline number.
  server_s   the server's own generation time: the X-Generation-Time header (sync servers), or
             submit -> "completed" seen by polling (async servers, accurate to --poll-interval).
  queue_s    async servers only: submit -> first status other than queued.
Requests are sequential, so nothing queues behind this script's own requests; if anyone else
uses the endpoint during the run, their jobs will show up as extra latency.
"""

import argparse
import base64
import csv
import json
import math
import os
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from http.client import HTTPException

FL2VA_PATH = "/v1/videos/generations/i2v"
TERMINAL = ("completed", "failed", "cancelled")
NOT_STARTED = (None, "queued", "pending", "submitted")
DEFAULT_PROMPT = (
    "A wide aerial shot of a coastal town at sunrise. The camera drifts slowly forward over "
    "rooftops toward the harbour as fishing boats leave, gulls circle overhead, and soft waves "
    "break against the sea wall. Warm golden light, gentle wind, distant church bells."
)
DURATION_TOLERANCE_S = 0.6


# ------------------------------------------------------------------------------------- http


def http(method, url, key, body=None, timeout=60):
    """-> (status, data, elapsed_s, headers). status 0 = transport failure, data = reason.
    data is parsed JSON for JSON responses, else raw bytes."""
    headers = {"Authorization": f"Bearer {key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            if hdrs.get("content-type", "").startswith("application/json"):
                try:
                    data = json.loads(data or b"{}")
                except ValueError:
                    pass
            return r.status, data, time.perf_counter() - t0, hdrs
    except urllib.error.HTTPError as e:
        raw = e.read()[:4000]
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw.decode("utf-8", "replace")
        return e.code, data, time.perf_counter() - t0, {}
    except (
        urllib.error.URLError,
        OSError,
        HTTPException,
    ) as e:  # timeouts, resets, DNS
        return 0, f"{type(e).__name__}: {e}", time.perf_counter() - t0, {}


def short(x, n=300):
    if isinstance(x, (bytes, bytearray)):
        x = x[:n].decode("utf-8", "replace")
    s = x if isinstance(x, str) else json.dumps(x)
    return s if len(s) <= n else s[:n] + "..."


# -------------------------------------------------------------------------------- inputs


def synthetic_keyframe_png(width=1344, height=768, variant=0):
    """A smooth sky/sea gradient with a sun disc, as a PNG, using only zlib.

    variant 0 is a dawn palette, 1 a midday palette (handy as a distinct last keyframe)."""
    top, horizon, sea = [
        ((255, 170, 110), (255, 220, 160), (40, 70, 120)),
        ((90, 160, 235), (200, 230, 250), (20, 90, 150)),
    ][variant % 2]
    sun_x, sun_y, sun_r = (
        width * (0.3 if variant == 0 else 0.65),
        height * (0.45 if variant == 0 else 0.2),
        height * 0.07,
    )
    horizon_y = int(height * 0.55)
    rows = []
    for y in range(height):
        if y < horizon_y:
            t = y / horizon_y
            base = [int(top[c] + (horizon[c] - top[c]) * t) for c in range(3)]
        else:
            t = (y - horizon_y) / (height - horizon_y)
            base = [int(sea[c] * (1 - 0.5 * t)) for c in range(3)]
        row = bytearray(base * width)
        dy = y - sun_y
        if abs(dy) < sun_r:
            half = math.sqrt(sun_r * sun_r - dy * dy)
            x0, x1 = max(0, int(sun_x - half)), min(width, int(sun_x + half))
            row[x0 * 3 : x1 * 3] = bytes((255, 245, 220)) * (x1 - x0)
        rows.append(b"\x00" + bytes(row))
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", raw)
        + chunk(b"IEND", b"")
    )


def load_b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


# ----------------------------------------------------------------------------- mp4 check


def mp4_info(data):
    """(duration_s, has_video, has_audio) read from the MP4 container, no ffmpeg needed."""
    dur = None
    i = data.find(b"mvhd")
    if i > 0:
        try:
            if data[i + 4] == 1:
                (ts,) = struct.unpack(">I", data[i + 24 : i + 28])
                (d,) = struct.unpack(">Q", data[i + 28 : i + 36])
            else:
                (ts,) = struct.unpack(">I", data[i + 16 : i + 20])
                (d,) = struct.unpack(">I", data[i + 20 : i + 24])
            dur = d / ts if ts else None
        except (struct.error, IndexError):
            dur = None
    return (
        dur,
        (b"hdlr" in data and b"vide" in data),
        (b"hdlr" in data and b"soun" in data),
    )


# --------------------------------------------------------------------------------- run


def check_server(args):
    """Wait (up to --wait-ready s) for /tt-liveness to report a ready model. Returns its body."""
    deadline = time.time() + args.wait_ready
    while True:
        code, body, _, _ = http(
            "GET", args.url + "/tt-liveness", args.api_key, timeout=30
        )
        ready = code == 200 and not (
            isinstance(body, dict) and body.get("model_ready") is False
        )
        if ready:
            return body if isinstance(body, dict) else {}
        if time.time() >= deadline:
            sys.exit(
                f"server not ready: GET /tt-liveness -> HTTP {code} {short(body, 200)}\n"
                "(405 'Model is not ready' = still warming up; pass --wait-ready SECONDS to wait)"
            )
        print(f"  waiting for the model to be ready (HTTP {code}) ...", flush=True)
        time.sleep(30)


def generate(args, payload, duration):
    """One FL2VA request, followed to the finished MP4. Returns a result dict."""
    rec = {
        "duration_req_s": duration,
        "http_code": None,
        "mode": None,
        "e2e_s": None,
        "server_s": None,
        "queue_s": None,
        "download_s": None,
        "mp4_bytes": None,
        "mp4_duration_s": None,
        "has_video": None,
        "has_audio": None,
        "ok": False,
        "error": None,
        "job_id": None,
    }
    body = json.dumps(payload).encode()
    t0 = time.perf_counter()
    code, data, _, hdrs = http(
        "POST", args.url + FL2VA_PATH, args.api_key, body=body, timeout=args.timeout
    )
    rec["http_code"] = code
    video = None
    if code == 200 and isinstance(data, (bytes, bytearray)):
        # sync server: the POST response is the MP4
        rec["mode"] = "sync"
        video = data
        gen = hdrs.get("x-generation-time")
        rec["server_s"] = float(gen) if gen else None
    elif code in (200, 201, 202) and isinstance(data, dict) and data.get("id"):
        # async server: poll the job, then download the MP4
        rec["mode"], job_id = "async", data["id"]
        rec["job_id"] = job_id
        t_submit, t_started, status, failures = time.perf_counter(), None, None, 0
        while True:
            time.sleep(args.poll_interval)
            c, st, _, _ = http(
                "GET",
                f"{args.url}/v1/videos/generations/{job_id}",
                args.api_key,
                timeout=30,
            )
            if c == 200 and isinstance(st, dict):
                status = st.get("status")
                if status not in NOT_STARTED and t_started is None:
                    t_started = time.perf_counter()
                if status in TERMINAL:
                    break
            else:
                failures += 1
                if failures >= 20:
                    rec["error"] = f"status endpoint failing: HTTP {c} {short(st, 160)}"
                    break
            if time.perf_counter() - t0 > args.timeout:
                rec["error"] = (
                    f"timed out after {args.timeout:.0f} s (last status {status})"
                )
                http(
                    "POST",
                    f"{args.url}/v1/videos/generations/{job_id}/cancel",
                    args.api_key,
                    timeout=30,
                )
                break
        t_done = time.perf_counter()
        rec["queue_s"] = round((t_started or t_done) - t_submit, 3)
        rec["server_s"] = round(t_done - t_submit, 3)
        if rec["error"] is None and status != "completed":
            rec["error"] = (
                f"job {status}: {short(st.get('error') if isinstance(st, dict) else st, 200)}"
            )
        if rec["error"] is None:
            c, data, dl, _ = http(
                "GET",
                f"{args.url}/v1/videos/generations/{job_id}/download",
                args.api_key,
                timeout=args.timeout,
            )
            rec["download_s"] = round(dl, 3)
            if c == 200 and isinstance(data, (bytes, bytearray)):
                video = data
            else:
                rec["error"] = f"download HTTP {c}: {short(data, 160)}"
        # finished jobs keep their base64 inputs on the server; remove ours
        http(
            "DELETE",
            f"{args.url}/v1/videos/generations/{job_id}",
            args.api_key,
            timeout=60,
        )
    else:
        rec["error"] = f"submit HTTP {code}: {short(data)}"
    rec["e2e_s"] = round(time.perf_counter() - t0, 3)
    if rec["server_s"] is not None:
        rec["server_s"] = round(rec["server_s"], 3)

    if video is not None:
        dur, has_v, has_a = mp4_info(video)
        rec.update(
            mp4_bytes=len(video),
            mp4_duration_s=round(dur, 3) if dur else None,
            has_video=has_v,
            has_audio=has_a,
        )
        problems = []
        if not has_v:
            problems.append("no video track")
        if not has_a:
            problems.append("no audio track")
        if dur is None:
            problems.append("MP4 has no duration")
        elif abs(dur - duration) > DURATION_TOLERANCE_S:
            problems.append(f"MP4 is {dur:.2f} s, asked for {duration} s")
        rec["error"] = "; ".join(problems) or None
        rec["ok"] = not problems
        rec["_video"] = video
    return rec


def summarize(results, durations):
    rows = []
    for d in durations:
        timed = [r for r in results if r["duration_req_s"] == d and not r["warmup"]]
        good = [r for r in timed if r["ok"]]
        e2e = [r["e2e_s"] for r in good]
        srv = [r["server_s"] for r in good if r["server_s"] is not None]
        rows.append(
            {
                "duration_s": d,
                "runs": len(timed),
                "ok": len(good),
                "e2e_median_s": round(statistics.median(e2e), 2) if e2e else None,
                "e2e_min_s": round(min(e2e), 2) if e2e else None,
                "e2e_max_s": round(max(e2e), 2) if e2e else None,
                "e2e_mean_s": round(statistics.mean(e2e), 2) if e2e else None,
                "server_median_s": round(statistics.median(srv), 2) if srv else None,
                "video_s_per_wall_s": round(d / statistics.median(e2e), 3)
                if e2e
                else None,
            }
        )
    return rows


def render_table(rows):
    head = "| Clip | Runs OK | E2E median (s) | E2E min - max (s) | Server gen median (s) | Video s / wall s |"
    lines = [head, "|---|---|---|---|---|---|"]
    for r in rows:
        rng = (
            f"{r['e2e_min_s']} - {r['e2e_max_s']}"
            if r["e2e_min_s"] is not None
            else "-"
        )
        lines.append(
            f"| {r['duration_s']} s | {r['ok']}/{r['runs']} | {r['e2e_median_s'] or '-'} | {rng} | "
            f"{r['server_median_s'] or '-'} | {r['video_s_per_wall_s'] or '-'} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--url",
        default=os.environ.get("H3_URL", "http://localhost:8000"),
        help="server base URL (env H3_URL; default http://localhost:8000)",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("H3_API_KEY"),
        help="bearer key (env H3_API_KEY)",
    )
    ap.add_argument(
        "--durations",
        default="5,10,15",
        help="clip lengths in seconds (4 to 15), comma separated",
    )
    ap.add_argument(
        "--runs", type=int, default=3, help="timed runs per duration (default 3)"
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed runs per duration before the timed ones (default 1)",
    )
    ap.add_argument(
        "--aspect-ratio",
        default="16:9",
        help="21:9, 16:9, 4:3, 1:1, 3:4 or 9:16 (default 16:9)",
    )
    ap.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="text prompt, the same for every request",
    )
    ap.add_argument("--prompt-file", help="read the prompt from this file instead")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--first-frame",
        help="first keyframe image (PNG/JPEG/WebP). Default: a generated 1344x768 PNG",
    )
    ap.add_argument(
        "--last-frame", help="also send a last keyframe (frame_pos -1) from this image"
    )
    ap.add_argument(
        "--first-and-last",
        action="store_true",
        help="send generated first and last keyframes (ignored if --first-frame/--last-frame are given)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="per-request limit in seconds (default 1800)",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="async servers: seconds between status polls",
    )
    ap.add_argument(
        "--wait-ready",
        type=float,
        default=0,
        help="wait up to this many seconds for the model to be ready",
    )
    ap.add_argument(
        "--save-videos",
        action="store_true",
        help="keep every MP4 in the results folder",
    )
    ap.add_argument(
        "--out", default="results", help="parent folder for results (default ./results)"
    )
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("set the API key: --api-key KEY or export H3_API_KEY=KEY")
    args.url = args.url.rstrip("/")
    durations = [int(x) for x in args.durations.split(",") if x.strip()]
    if any(not 4 <= d <= 15 for d in durations):
        sys.exit("durations must be whole seconds from 4 to 15")
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read()

    if args.first_frame:
        frames = [{"image": load_b64(args.first_frame), "frame_pos": 0}]
        if args.last_frame:
            frames.append({"image": load_b64(args.last_frame), "frame_pos": -1})
        kf_desc = args.first_frame + (
            f" + {args.last_frame}" if args.last_frame else ""
        )
    elif args.last_frame:
        frames = [{"image": load_b64(args.last_frame), "frame_pos": -1}]
        kf_desc = f"last only: {args.last_frame}"
    else:
        frames = [
            {
                "image": base64.b64encode(synthetic_keyframe_png(variant=0)).decode(),
                "frame_pos": 0,
            }
        ]
        kf_desc = "generated 1344x768 PNG"
        if args.first_and_last:
            frames.append(
                {
                    "image": base64.b64encode(
                        synthetic_keyframe_png(variant=1)
                    ).decode(),
                    "frame_pos": -1,
                }
            )
            kf_desc = "generated 1344x768 PNGs (first + last)"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.out, f"h3_fl2va_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"MiniMax-H3 FL2VA benchmark -> {args.url}")
    liveness = check_server(args)
    print(
        f"  server ready; keyframes: {kf_desc}; aspect {args.aspect_ratio}; seed {args.seed}; "
        f"{args.warmup} warmup + {args.runs} timed per duration; durations {durations}"
    )

    results = []
    for d in durations:
        for i in range(args.warmup + args.runs):
            warm = i < args.warmup
            label = f"{d:>2} s {'warmup' if warm else f'run {i - args.warmup + 1}/{args.runs}'}"
            payload = {
                "prompt": prompt,
                "duration": d,
                "aspect_ratio": args.aspect_ratio,
                "seed": args.seed,
                "image_prompts": frames,
            }
            print(f"  {label:<14} ...", end="", flush=True)
            rec = generate(args, payload, d)
            rec.update(
                warmup=warm,
                index=i,
                started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            video = rec.pop("_video", None)
            if video is not None and args.save_videos:
                name = f"fl2va_{d}s_{'warmup' if warm else 'run'}{i}.mp4"
                with open(os.path.join(out_dir, name), "wb") as fh:
                    fh.write(video)
                rec["saved_as"] = name
            results.append(rec)
            srv = (
                f" server {rec['server_s']:.1f} s"
                if rec["server_s"] is not None
                else ""
            )
            status = "ok " if rec["ok"] else "FAIL"
            print(
                f" {status} e2e {rec['e2e_s']:.1f} s{srv}"
                + (f"  ({rec['error']})" if rec["error"] else ""),
                flush=True,
            )
            if rec["http_code"] in (401, 403):
                sys.exit("the server refused the API key; check --api-key / H3_API_KEY")

    rows = summarize(results, durations)
    table = render_table(rows)
    meta = {
        "url": args.url,
        "started": stamp,
        "aspect_ratio": args.aspect_ratio,
        "seed": args.seed,
        "prompt": prompt,
        "keyframes": kf_desc,
        "warmup_per_duration": args.warmup,
        "runs_per_duration": args.runs,
        "modes_seen": sorted({r["mode"] for r in results if r["mode"]}),
        "liveness": liveness,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"meta": meta, "summary": rows, "results": results}, fh, indent=1)
    fields = [
        "duration_req_s",
        "warmup",
        "index",
        "ok",
        "e2e_s",
        "server_s",
        "queue_s",
        "download_s",
        "http_code",
        "mode",
        "mp4_bytes",
        "mp4_duration_s",
        "has_video",
        "has_audio",
        "error",
        "job_id",
        "started_at",
        "saved_as",
    ]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write(
            f"# MiniMax-H3 FL2VA benchmark, {stamp}\n\n- Endpoint: {args.url}\n- Aspect ratio {args.aspect_ratio}, "
            f"seed {args.seed}, keyframes: {kf_desc}\n- {args.warmup} warmup + {args.runs} timed requests per "
            f"duration, one at a time\n- Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n\n{table}\n\n"
            "E2E = POST sent to last MP4 byte received. Server gen = X-Generation-Time (sync) or submit to "
            "completed (async).\n"
        )

    print("\n" + table)
    print(f"\nresults: {out_dir}/ (summary.md, summary.json, results.csv)")
    failed = [r for r in results if not r["ok"] and not r["warmup"]]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
