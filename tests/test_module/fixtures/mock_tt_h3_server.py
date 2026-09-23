#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""A stand-in for a tt-media-server MiniMax-H3 endpoint, for exercising the tt-h3
adapter and test_h3_bench.py without hardware.  Standard library + ffmpeg.

By default it speaks the contract observed on the hosted endpoints on 2026-09-18:
bearer auth, POST -> 202 + job id, queued -> in_progress -> completed, GET
.../download serves a real mp4 (ffmpeg testsrc2 + sine, 1344x768 at 24 fps,
17n+5 frames), `num_inference_steps` is a hard 422 "unknown field", the
deployment gates refuse the tasks it does not serve with a str `detail` before
the key is checked, cancel of a finished job is 404 (KNOWN cancel-of-finished-404)
and DELETE of a running job is 409.

    ./mock_tt_h3_server.py --port 8931 --serve t2va  --key k1 --gen-seconds 3
    ./mock_tt_h3_server.py --port 8932 --serve fl2va --key k1 --gen-seconds 3 --hang fl2va:10

Knobs for breaking things on purpose:
  --sync                 the 09-14 contract: POST blocks and answers 200 video/mp4, no job
  --hang TASK:DURATION   a job of that shape stays in_progress forever
  --fail TASK            jobs of that task end `failed` with a device-timeout error
  --no-jobs-route        GET /v1/videos/jobs -> 404 (a build without the listing)
  --silent               clips carry no audio track
  --railed               the soundtrack is a full-scale square wave (the corruption signature)
  --short                clips are one frame short (17n+4 -- the known one-frame gap; a note, not a failure)
  --progress             add a `progress` field to job records (the real server has none)
  --no-auth              no API key in front of anything
  --queue-seconds S      time a job spends `queued` before it starts (default 1)
  --flat                 --gen-seconds is not scaled by duration/5
"""

import argparse
import atexit
import base64
import binascii
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _ffmpeg():
    """ffmpeg from PATH, else the binary imageio-ffmpeg ships (the tests venv)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # pyright: ignore[reportMissingImports]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


TASK_OF = {
    "/v1/videos/generations": "t2va",
    "/v1/videos/generations/i2v": "fl2va",
    "/v1/videos/generations/ref2va": "ref2va",
}
ASPECTS = {
    "21:9": (1536, 672),
    "16:9": (1344, 768),
    "4:3": (1024, 768),
    "1:1": (768, 768),
    "3:4": (768, 1024),
    "9:16": (768, 1344),
}
READS = "aspect_ratio, duration_seconds, negative_prompt, prompt, seed"
MAX_IMAGE_B64 = 10_000_000
MAX_MEDIA_B64 = 80_000_000
CANCELLING_S = 1.0  # a cancelled job reads `cancelling` for this long

OPTS = None
JOBS = {}
LOCK = threading.Lock()
CLIPS = {}  # key -> {"event": Event, "path": str | None}
CLIP_DIR = tempfile.mkdtemp(prefix="mock-tt-h3-")
atexit.register(shutil.rmtree, CLIP_DIR, ignore_errors=True)


def frames_for(seconds):
    n = int(round(seconds * 24))
    while (n - 5) % 17:
        n += 1
    return n


def clip(seconds, aspect):
    """A real mp4 for this shape, encoded once (single-flight) and atomically."""
    key = (seconds, aspect, OPTS.silent, OPTS.railed, OPTS.short)
    with LOCK:
        entry = CLIPS.get(key)
        mine = entry is None
        if mine:
            entry = CLIPS[key] = {"event": threading.Event(), "path": None}
    if not mine:
        entry["event"].wait(timeout=600)
        if entry["path"] and os.path.exists(entry["path"]):
            return entry["path"]
        raise RuntimeError("clip encode failed")
    n = frames_for(seconds) - (1 if OPTS.short else 0)
    w, h = ASPECTS[aspect]
    path = os.path.join(CLIP_DIR, f"{seconds}s-{aspect.replace(':', 'x')}-{n}f.mp4")
    part = path + ".part.mp4"
    cmd = [
        _ffmpeg(),
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-t",
        f"{n / 24:.6f}",
        "-i",
        f"testsrc2=size={w}x{h}:rate=24",
    ]
    if not OPTS.silent:
        src = (
            "aevalsrc=sgn(sin(2*PI*440*t)):s=48000:c=stereo"
            if OPTS.railed
            else "sine=frequency=440:sample_rate=48000"
        )
        cmd += ["-f", "lavfi", "-t", f"{n / 24:.6f}", "-i", src]
    cmd += [
        "-frames:v",
        str(n),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "24",
    ]
    if not OPTS.silent:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ac", "2"]
    cmd += ["-movflags", "+faststart", part]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        os.replace(part, path)
        entry["path"] = path
    except Exception:
        with LOCK:  # a failed encode must not poison the shape for good
            CLIPS.pop(key, None)
        raise
    finally:
        entry["event"].set()
    return path


def job_view(job, now=None):
    now = now or time.time()
    el = now - job["t0"]
    if job["cancelled_at"] is not None:
        status = (
            "cancelling" if now - job["cancelled_at"] < CANCELLING_S else "cancelled"
        )
    elif el < OPTS.queue_seconds:
        status = "queued"
    elif job["fail"] and el >= OPTS.queue_seconds + job["gen"] / 2:
        status = "failed"
    elif job["hang"] or el < OPTS.queue_seconds + job["gen"]:
        status = "in_progress"
    else:
        status = "completed"
    out = {
        "id": job["id"],
        "job_type": "video",
        "model": "MiniMax-H3",
        "task": job["task"],
        "status": status,
        "created_at": int(job["t0"]),
        "request_parameters": job["echo"],
    }
    if OPTS.progress and status == "in_progress":
        out["progress"] = (
            0.0
            if job["hang"]
            else round(min(0.99, (el - OPTS.queue_seconds) / job["gen"]), 3)
        )
    if status == "completed":
        out["completed_at"] = int(job["t0"] + OPTS.queue_seconds + job["gen"])
    if status == "failed":
        out["error"] = {
            "message": "TT_THROW: TIMEOUT: device timeout in fetch queue wait, "
            "potential hang detected",
            "code": "device_error",
        }
    return out


def _decodes(b64):
    try:
        return len(base64.b64decode(b64, validate=True)) > 0
    except (binascii.Error, ValueError):
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-tt-h3/1.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if OPTS.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ------------------------------------------------------------------
    def send(self, code, payload=None, raw=None, ctype="application/json", extra=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if self.path.startswith("/video/"):
            self.send_header("Deprecation", "true")
            self.send_header("Sunset", "Wed, 31 Dec 2026 23:59:59 GMT")
            self.send_header(
                "Link",
                f"<{self.path.replace('/video/', '/v1/videos/', 1)}>; "
                f'rel="successor-version"',
            )
        self.end_headers()
        self.wfile.write(body)

    def detail(self, code, text):
        self.send(code, {"detail": text})

    def authed(self):
        """True if the request may proceed; otherwise the 401 has been sent."""
        if OPTS.no_auth:
            return True
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or not auth[7:].strip():
            self.detail(401, "Not authenticated")
            return False
        if auth[7:].strip() != OPTS.key:
            self.detail(401, "Invalid or missing API Key")
            return False
        return True

    def read_body(self):
        """Drain the request body so keep-alive stays in sync; chunked bodies close."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self.close_connection = True
            return b""
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def norm(self):
        p = self.path.split("?", 1)[0].rstrip("/")
        return p.replace("/video/", "/v1/videos/", 1) if p.startswith("/video/") else p

    def job_id_from(self, path, suffix=""):
        prefix = "/v1/videos/generations/"
        if not path.startswith(prefix):
            return None
        rest = path[len(prefix) :]
        if suffix:
            if not rest.endswith(suffix):
                return None
            rest = rest[: -len(suffix)]
        return rest if rest and "/" not in rest else None

    def gate_text(self, task):
        return (
            f"This deployment serves {', '.join(sorted(OPTS.serve))} only; {task} is not "
            f"available here. Use the /v1/videos/generations route of a deployment that serves it."
        )

    # -- GET ----------------------------------------------------------------------
    def do_GET(self):
        self.read_body()
        p = self.norm()
        if p == "/health":
            return self.send(200, {"status": "ok"})
        if p == "/v1/models":
            return self.send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "MiniMax-H3",
                            "object": "model",
                            "owned_by": "tenstorrent",
                        }
                    ],
                },
            )
        if p == "/tt-liveness":
            with LOCK:
                queued = sum(
                    1 for j in JOBS.values() if job_view(j)["status"] == "queued"
                )
            return self.send(
                200,
                {
                    "status": "ok",
                    "model_ready": True,
                    "queue_size": queued,
                    "max_queue_size": 2,
                    "device_mesh_shape": OPTS.mesh,
                    "runner_in_use": OPTS.runner,
                    "device": "mock",
                },
            )
        if p == "/metrics":
            return self.send(
                200,
                raw=b"# TYPE python_gc_objects_collected_total counter\n"
                b'python_gc_objects_collected_total{generation="0"} 1.0\n',
                ctype="text/plain; version=0.0.4",
            )
        if p == "/openapi.json":
            paths = {}
            for route in (
                "/v1/videos/generations",
                "/v1/videos/generations/i2v",
                "/v1/videos/generations/i2v/upload",
                "/v1/videos/generations/ref2va",
                "/video/generations",
                "/video/generations/i2v",
                "/video/generations/ref2va",
            ):
                paths[route] = {"post": {"security": [{"HTTPBearer": []}]}}
            if not OPTS.no_jobs_route:
                for route in ("/v1/videos/jobs", "/video/jobs"):
                    paths[route] = {"get": {"security": [{"HTTPBearer": []}]}}
            for route in (
                "/v1/videos/generations/{job_id}",
                "/video/generations/{job_id}",
            ):
                paths[route] = {"get": {}, "delete": {}}
                paths[route + "/download"] = {"get": {}}
                paths[route + "/cancel"] = {"post": {}}
            for route in ("/health", "/tt-liveness", "/v1/models", "/metrics"):
                paths[route] = {"get": {}}
            return self.send(
                200,
                {
                    "openapi": "3.1.0",
                    "info": {"title": "mock tt-media-server"},
                    "paths": paths,
                },
            )
        if p == "/v1/videos/jobs":
            if OPTS.no_jobs_route:
                return self.detail(404, "Not Found")
            if not self.authed():
                return
            with LOCK:
                jobs = [job_view(j) for j in JOBS.values()]
            for j in jobs:
                pass  # the real listing carries request_parameters (JobManager.get_all_jobs_metadata)
            return self.send(200, jobs)
        jid = self.job_id_from(p, "/download")
        if jid:
            if not self.authed():
                return
            with LOCK:
                job = JOBS.get(jid)
            if not job or job_view(job)["status"] != "completed":
                return self.detail(404, "Video content not available")
            try:
                path = clip(job["duration"], job["aspect"])
            except Exception as exc:  # noqa: BLE001
                return self.detail(500, f"clip encode failed: {exc}")
            with open(path, "rb") as fh:
                data = fh.read()
            return self.send(
                200,
                raw=data,
                ctype="video/mp4",
                extra={"Content-Disposition": f'attachment; filename="{jid}.mp4"'},
            )
        jid = self.job_id_from(p)
        if jid:
            if not self.authed():
                return
            with LOCK:
                job = JOBS.get(jid)
            if not job:
                return self.detail(404, "Video job not found")
            return self.send(200, job_view(job))
        self.detail(404, "Not Found")

    # -- POST ---------------------------------------------------------------------
    def do_POST(self):
        p = self.norm()
        body = self.read_body()
        if p in TASK_OF:
            return self.submit(TASK_OF[p], body)
        if p == "/v1/videos/generations/i2v/upload":
            if "fl2va" not in OPTS.serve:
                return self.detail(422, self.gate_text("fl2va"))
            if not self.authed():
                return
            return self.detail(422, "multipart upload is not implemented by the mock")
        jid = self.job_id_from(p, "/cancel")
        if jid:
            if not self.authed():
                return
            with LOCK:
                job = JOBS.get(jid)
                if not job or job_view(job)["status"] in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    return self.detail(
                        404, "Video job not found"
                    )  # KNOWN cancel-of-finished-404
                if job["cancelled_at"] is None:
                    job["cancelled_at"] = time.time()
                view = job_view(job)
            return self.send(200, view)
        if p == "/v1/models":
            return self.detail(405, "Method Not Allowed")
        self.detail(404, "Not Found")

    def do_DELETE(self):
        self.read_body()
        p = self.norm()
        jid = self.job_id_from(p)
        if not jid:
            return self.detail(404, "Not Found")
        if not self.authed():
            return
        with LOCK:
            job = JOBS.get(jid)
            if not job:
                return self.detail(404, "Video job not found")
            if job_view(job)["status"] in ("queued", "in_progress", "cancelling"):
                return self.detail(409, "job is still running")
            del JOBS[jid]
        self.send(200, {"id": jid, "deleted": True})

    def submit(self, task, body):
        # 1. the deployment gate answers first, before auth and before the body
        if task not in OPTS.serve:
            return self.detail(422, self.gate_text(task))
        # 2. then the key
        if not self.authed():
            return
        # 3. then the schema
        try:
            req = json.loads(body.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError):
            return self.send(
                422,
                {
                    "detail": [
                        {
                            "type": "json_invalid",
                            "loc": ["body"],
                            "msg": "JSON decode error",
                        }
                    ]
                },
            )
        if not isinstance(req, dict):
            return self.send(
                422,
                {
                    "detail": [
                        {
                            "type": "model_attributes_type",
                            "loc": ["body"],
                            "msg": "Input should be a valid dictionary",
                        }
                    ]
                },
            )
        errors = []

        def err(loc, msg, kind="value_error"):
            errors.append({"type": kind, "loc": ["body", *loc], "msg": msg})

        prompt = req.get("prompt")
        if prompt is None:
            err(["prompt"], "Field required", "missing")
        elif not isinstance(prompt, str):
            err(["prompt"], "Input should be a valid string", "string_type")
        dur = req.get("duration_seconds", 5)
        if not isinstance(dur, int) or isinstance(dur, bool) or not 1 <= dur <= 60:
            err(
                ["duration_seconds"],
                "Input should be a valid integer between 1 and 60",
                "int_parsing",
            )
        seed = req.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            err(["seed"], "Input should be a valid integer", "int_parsing")
        aspect = req.get("aspect_ratio", "16:9")
        if not isinstance(aspect, str):
            err(["aspect_ratio"], "Input should be a valid string", "string_type")
        if "num_inference_steps" in req and not isinstance(
            req["num_inference_steps"], int
        ):
            err(
                ["num_inference_steps"],
                "Input should be a valid integer",
                "int_parsing",
            )
        if task == "fl2va":
            ips = req.get("image_prompts")
            if not isinstance(ips, list) or not ips:
                err(["image_prompts"], "Field required (1-2 entries)", "missing")
            else:
                for i, e in enumerate(ips):
                    img = e.get("image") if isinstance(e, dict) else None
                    if not isinstance(img, str) or not img:
                        err(
                            ["image_prompts", i, "image"],
                            "Input should be a valid string",
                            "string_type",
                        )
                    elif len(img) > MAX_IMAGE_B64:
                        err(
                            ["image_prompts", i, "image"],
                            f"String should have at most {MAX_IMAGE_B64} characters",
                            "string_too_long",
                        )
                    elif not img.startswith(("http://", "https://")) and not _decodes(
                        img
                    ):
                        err(["image_prompts", i, "image"], "image could not be decoded")
        if task == "ref2va":
            refs = req.get("references")
            if not isinstance(refs, dict) or not any(
                refs.get(k) for k in ("images", "videos", "audios")
            ):
                err(["references"], "at least one reference is required", "missing")
            else:
                if refs.get("audios") and not (
                    refs.get("images") or refs.get("videos")
                ):
                    err(
                        ["references"],
                        "audio references must be paired with an image or video",
                    )
                for kind, cap in (("images", 9), ("videos", 3), ("audios", 3)):
                    items = refs.get(kind) or []
                    if len(items) > cap:
                        err(["references", kind], f"at most {cap} reference {kind}")
                    for i, e in enumerate(items):
                        b64 = e.get("b64") if isinstance(e, dict) else None
                        url = e.get("url") if isinstance(e, dict) else None
                        if (b64 is None) == (url is None):
                            err(
                                ["references", kind, i],
                                "exactly one of b64 or url is required",
                            )
                        elif b64 is not None and not isinstance(b64, str):
                            err(
                                ["references", kind, i, "b64"],
                                "Input should be a valid string",
                                "string_type",
                            )
                        elif b64 is not None and len(b64) > MAX_MEDIA_B64:
                            err(
                                ["references", kind, i, "b64"],
                                f"String should have at most {MAX_MEDIA_B64} characters",
                                "string_too_long",
                            )
                        elif b64 is not None and not _decodes(b64):
                            err(
                                ["references", kind, i, "b64"],
                                f"{kind}[{i}] could not be decoded",
                            )
                        elif url is not None and not str(url).startswith(
                            ("http://", "https://")
                        ):
                            err(["references", kind, i, "url"], "url must be http(s)")
        if errors:
            return self.send(422, {"detail": errors})
        # 4. then the MiniMax-H3 policy
        allowed = {
            "prompt",
            "seed",
            "duration_seconds",
            "aspect_ratio",
            "negative_prompt",
        }
        allowed |= (
            {"image_prompts"}
            if task == "fl2va"
            else {"references"}
            if task == "ref2va"
            else set()
        )
        unknown = sorted(k for k in req if k not in allowed)
        if unknown:
            return self.detail(
                422,
                f"unknown field(s) for MiniMax-H3: {', '.join(unknown)}. "
                f"This deployment reads: {READS}",
            )
        if not 4 <= dur <= 15:
            return self.detail(
                422, "duration_seconds must be an integer from 4 to 15 for MiniMax-H3"
            )
        if aspect not in ASPECTS:
            return self.detail(
                422,
                f"aspect_ratio {aspect!r} is not served for MiniMax-H3; "
                f"pick from {sorted(ASPECTS)}",
            )
        if task == "fl2va":
            pos = [e.get("frame_pos", 0) for e in req["image_prompts"]]
            if (
                len(pos) > 2
                or any(x not in (0, -1) for x in pos)
                or len(set(pos)) != len(pos)
            ):
                return self.detail(
                    422,
                    "MiniMax-H3 FL2VA accepts at most two image_prompts with "
                    "frame_pos in {0, -1}, no duplicates",
                )
        # 5. accepted
        gen = OPTS.gen_seconds if OPTS.flat else OPTS.gen_seconds * dur / 5.0
        if OPTS.sync:
            # the 09-14 contract: this request IS the generation
            time.sleep(OPTS.queue_seconds + gen)
            try:
                path = clip(dur, aspect)
            except Exception as exc:  # noqa: BLE001
                return self.detail(500, f"clip encode failed: {exc}")
            with open(path, "rb") as fh:
                data = fh.read()
            return self.send(
                200,
                raw=data,
                ctype="video/mp4",
                extra={
                    "Content-Disposition": 'attachment; filename="generation.mp4"',
                    "X-Generation-Time": f"{gen:.2f}",
                },
            )
        jid = str(uuid.uuid4())
        echo = {
            k: v for k, v in req.items() if k not in ("image_prompts", "references")
        }
        echo["num_inference_steps"] = 20  # the schema default, echoed -- not what runs
        if task == "fl2va":
            echo["image_prompts"] = [
                {"frame_pos": e.get("frame_pos", 0), "image": "<elided>"}
                for e in req["image_prompts"]
            ]
        job = {
            "id": jid,
            "task": task,
            "t0": time.time(),
            "duration": dur,
            "aspect": aspect,
            "gen": gen,
            "echo": echo,
            "cancelled_at": None,
            "hang": (task, dur) in OPTS.hang_shapes,
            "fail": task in OPTS.fail_tasks,
        }
        with LOCK:
            JOBS[jid] = job
        # encode the clip in the background so download is quick when the job completes
        if not job["hang"] and not job["fail"]:
            threading.Thread(
                target=lambda: _quiet(clip, dur, aspect), daemon=True
            ).start()
        self.send(202, job_view(job))


def _quiet(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"background encode failed: {exc}\n")


def main(argv=None):
    global OPTS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument(
        "--serve", default="t2va,fl2va,ref2va", help="tasks this deployment serves"
    )
    ap.add_argument("--key", default="mock-key")
    ap.add_argument("--no-auth", action="store_true")
    ap.add_argument("--queue-seconds", type=float, default=1.0)
    ap.add_argument(
        "--gen-seconds",
        type=float,
        default=3.0,
        help="in_progress time for a 5 s clip (scaled by duration/5 unless --flat)",
    )
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--hang", action="append", default=[], metavar="TASK:DURATION")
    ap.add_argument("--fail", action="append", default=[], metavar="TASK")
    ap.add_argument("--no-jobs-route", action="store_true")
    ap.add_argument("--silent", action="store_true")
    ap.add_argument("--railed", action="store_true")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--mesh",
        default="4,8",
        help="device_mesh_shape /tt-liveness reports (rows,cols); a single BH Galaxy is 4,8",
    )
    ap.add_argument(
        "--runner",
        default="tt-minimax-h3-t2va",
        help="runner_in_use /tt-liveness reports",
    )
    OPTS = ap.parse_args(argv)
    OPTS.mesh = [int(x) for x in str(OPTS.mesh).split(",")]
    OPTS.serve = {t.strip() for t in OPTS.serve.split(",") if t.strip()}
    bad = OPTS.serve - {"t2va", "fl2va", "ref2va"}
    if bad:
        ap.error(
            f"--serve: unknown task(s) {sorted(bad)}; pick from t2va, fl2va, ref2va"
        )
    OPTS.hang_shapes = set()
    for spec in OPTS.hang:
        task, _, dur = spec.partition(":")
        if task not in ("t2va", "fl2va", "ref2va") or not dur.isdigit():
            ap.error(f"--hang: expected TASK:DURATION, got {spec!r}")
        OPTS.hang_shapes.add((task, int(dur)))
    OPTS.fail_tasks = set(OPTS.fail)
    if OPTS.fail_tasks - {"t2va", "fl2va", "ref2va"}:
        ap.error(f"--fail: unknown task(s) {sorted(OPTS.fail_tasks)}")
    if not _ffmpeg():
        ap.error("ffmpeg is needed to make the clips (PATH or imageio-ffmpeg)")
    srv = ThreadingHTTPServer((OPTS.bind, OPTS.port), Handler)
    srv.daemon_threads = True
    print(
        f"mock tt-h3 on http://{OPTS.bind}:{OPTS.port}  serve={sorted(OPTS.serve)} "
        f"auth={'none' if OPTS.no_auth else 'bearer'} mode={'SYNC' if OPTS.sync else 'async'} "
        f"gen={OPTS.gen_seconds}s hang={sorted(OPTS.hang_shapes)} fail={sorted(OPTS.fail_tasks)} "
        f"jobs_route={not OPTS.no_jobs_route} silent={OPTS.silent} railed={OPTS.railed} "
        f"short={OPTS.short}",
        flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
