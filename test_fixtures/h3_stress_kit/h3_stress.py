#!/usr/bin/env python3
"""MiniMax-H3 stress test for the FL2VA (D23) and Ref2VA (C12) deployments.

    python3 h3_stress.py --minutes 30            # the standard run
    python3 h3_stress.py --minutes 3 --workers 2,1   # quick shakedown
    python3 report.py results/<run>              # re-render the report later

Both servers are driven at the same time, each by its own set of threads. One run walks
six time-boxed phases (budget fractions of --minutes):

  baseline  serial, one request per "core" case per server     -> unloaded latency
  validate  every negative/probe case + lifecycle checks        -> admission control
  sustain   N concurrent workers per server drawing random positive cases, with a
            negative request injected periodically, a status-poll hammer, and a
            /v1/videos/jobs list probe                          -> steady-state under load
  burst     B jobs submitted within a second per server, then drained -> queueing
  recovery  the baseline cases again, serial                    -> degradation vs baseline
  edge      the riskiest shapes last (7000-char prompts, out-of-ladder durations)

A monitor thread samples /tt-liveness on both servers every 10 s throughout.

Everything is stdlib. Inputs come from ./assets (see make_assets.py) and ./cases.json.
Outputs land in results/<timestamp>/: results.jsonl + results.csv (one row per request),
monitor.jsonl, stress.log, run.json, out/*.mp4 (sampled), report.md.

Auth: H3_API_KEY (default: the shared test key). Endpoints: H3_FL2VA_URL / H3_REF2VA_URL.
"""
import argparse
import base64
import copy
import csv
import hashlib
import json
import os
import random
import signal
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
PROMPTS = os.path.join(ASSETS, "prompts")

SERVERS = {
    "fl2va": os.environ.get("H3_FL2VA_URL", "https://sjc3-h3-fl2va-lightx2v-d23.n.cloud.tenstorrent.com").rstrip("/"),
    "ref2va": os.environ.get("H3_REF2VA_URL", "https://sjc3-h3-ref2va-lightx2v-c12.n.cloud.tenstorrent.com").rstrip("/"),
}
KEY = os.environ.get("H3_API_KEY", "tt_h3_test_key")
PATHS = {"t2va": "/v1/videos/generations", "fl2va": "/v1/videos/generations/i2v", "ref2va": "/v1/videos/generations/ref2va"}
TERMINAL = ("completed", "failed", "cancelled")
NOT_STARTED = (None, "queued", "pending", "submitted")
PHASE_FRACTION = {"baseline": 0.20, "validate": 0.07, "sustain": 0.48, "burst": 0.10, "recovery": 0.10, "edge": 0.05}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------ http


def http(method, url, body=None, headers=None, timeout=60):
    """-> (code, data, elapsed_s, headers). code 0 = transport failure (data is the reason).
    data is parsed JSON when the response is JSON, else raw bytes."""
    h = {"Authorization": f"Bearer {KEY}"}
    if headers:
        h.update(headers)
    if body is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, method=method, data=body, headers=h)
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
        return e.code, data, time.perf_counter() - t0, {k.lower(): v for k, v in e.headers.items()}
    except Exception as e:  # noqa: BLE001 - timeouts, resets, DNS
        return 0, f"{type(e).__name__}: {e}", time.perf_counter() - t0, {}


def short(x, n=240):
    if isinstance(x, (bytes, bytearray)):
        x = x[:n].decode("utf-8", "replace")
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


# ---------------------------------------------------------------------------- assets

_b64_cache, _b64_lock = {}, threading.Lock()


def asset_b64(name):
    with _b64_lock:
        if name not in _b64_cache:
            with open(os.path.join(ASSETS, name), "rb") as fh:
                _b64_cache[name] = base64.b64encode(fh.read()).decode("ascii")
        return _b64_cache[name]


_prompt_cache = {}


def read_text(name):
    if name not in _prompt_cache:
        with open(os.path.join(PROMPTS, name), encoding="utf-8") as fh:
            _prompt_cache[name] = fh.read()
    return _prompt_cache[name]


def prompt_bank():
    if "bank" not in _prompt_cache:
        with open(os.path.join(PROMPTS, "bank.json"), encoding="utf-8") as fh:
            _prompt_cache["bank"] = json.load(fh)
    return _prompt_cache["bank"]


def resolve_assets(obj):
    """Replace '@asset:NAME' strings anywhere inside obj with that file's base64."""
    if isinstance(obj, dict):
        return {k: resolve_assets(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_assets(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("@asset:"):
        return asset_b64(obj[len("@asset:"):])
    return obj


def build_payload(case, rnd):
    """(path, payload, prompt_text) for one case."""
    prompt = rnd.choice(prompt_bank()) if case["prompt"] == "bank" else read_text(case["prompt"])
    payload = {
        "prompt": prompt,
        "seed": case.get("seed", 42),
        "aspect_ratio": case.get("aspect_ratio", "16:9"),
        "duration": case["duration"],
    }
    if case.get("negative_prompt"):
        payload["negative_prompt"] = read_text(case["negative_prompt"])
    task = case["task"]
    if task == "fl2va":
        imgs = case.get("images") or []
        pos = case.get("frame_pos") or ([0, -1][: len(imgs)])
        if imgs:
            payload["image_prompts"] = [{"image": asset_b64(n), "frame_pos": p} for n, p in zip(imgs, pos)]
    elif task == "ref2va":
        refs = {}
        for key in ("images", "videos", "audios"):
            if case.get(key):
                refs[key] = [{"b64": asset_b64(n)} for n in case[key]]
        payload["references"] = refs
    if case.get("extra"):
        payload.update(resolve_assets(copy.deepcopy(case["extra"])))
    return PATHS[task], payload, prompt


# ------------------------------------------------------------------------- mp4 check


def mp4_info(data):
    """(duration_s, has_video, has_audio) from the container, no ffmpeg needed."""
    dur = None
    i = data.find(b"mvhd")
    if i > 0:
        try:
            ver = data[i + 4]
            if ver == 1:
                ts, = struct.unpack(">I", data[i + 24:i + 28])
                d, = struct.unpack(">Q", data[i + 28:i + 36])
            else:
                ts, = struct.unpack(">I", data[i + 16:i + 20])
                d, = struct.unpack(">I", data[i + 20:i + 24])
            dur = d / ts if ts else None
        except struct.error:
            dur = None
    return dur, (b"hdlr" in data and b"vide" in data), (b"hdlr" in data and b"soun" in data)


# ----------------------------------------------------------------------------- state


class Ctx:
    def __init__(self, args, out_dir):
        self.args = args
        self.out = out_dir
        self.stop = False
        self.lock = threading.Lock()
        self.inflight = Counter()
        self.active_jobs = {s: {} for s in SERVERS}   # job_id -> accept time, insertion-ordered (oldest first)
        self.last_liveness = {}
        self.results = open(os.path.join(out_dir, "results.jsonl"), "a")
        self.monitor = open(os.path.join(out_dir, "monitor.jsonl"), "a")
        self.logf = open(os.path.join(out_dir, "stress.log"), "a")
        self.saved_per_case = Counter()
        self.saved_bad = 0
        self.seq = 0
        self.admission_rejected = set()
        self.worker_error_cases = set()   # case ids that produced a processing_error in the worker
        self.error_base = {}               # worker error_count at run start, per server
        self.guard_tripped = set()         # servers where worker errors rose by >= --max-worker-errors

    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.lock:
            self.logf.write(line + "\n")
            self.logf.flush()

    def record(self, rec):
        with self.lock:
            self.seq += 1
            rec["seq"] = self.seq
            self.results.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.results.flush()

    def sample(self, rec):
        with self.lock:
            self.monitor.write(json.dumps(rec) + "\n")
            self.monitor.flush()

    def inflight_inc(self, server):
        with self.lock:
            n = self.inflight[server]
            self.inflight[server] += 1
            return n

    def inflight_dec(self, server):
        with self.lock:
            self.inflight[server] -= 1

    def maybe_save(self, rec, data, good):
        """keep policy: sample = first good output per case + every bad output (capped)."""
        pol = self.args.keep
        if pol == "none":
            return None
        with self.lock:
            if pol == "sample":
                if good:
                    if self.saved_per_case[rec["case_id"]] >= 1:
                        return None
                    self.saved_per_case[rec["case_id"]] += 1
                else:
                    if self.saved_bad >= 25:
                        return None
                    self.saved_bad += 1
        name = f"{rec['phase']}_{rec['case_id']}_{(rec.get('job_id') or 'nojob')[:8]}.mp4"
        path = os.path.join(self.out, "out", name)
        with open(path, "wb") as fh:
            fh.write(data)
        return os.path.join("out", name)


# ---------------------------------------------------------------------- one request


def run_case(case, phase, ctx, worker="", rnd=None):
    rnd = rnd or random.Random()
    server = case["server"]
    base = SERVERS[server]
    a = ctx.args
    path, payload, prompt = build_payload(case, rnd)
    body = json.dumps(payload).encode()
    expect = case.get("expect", "ok")
    rec = {
        "run": os.path.basename(ctx.out), "phase": phase, "worker": worker, "server": server,
        "case_id": case["id"], "task": case["task"], "expect": expect, "kind": "request",
        "requested_duration": case.get("duration"), "aspect_ratio": payload.get("aspect_ratio"),
        "prompt_chars": len(prompt), "request_bytes": len(body), "path": path,
        "submitted_at": now_iso(), "inflight_at_submit": None, "liveness_queue_at_submit": None,
        "submit_s": None, "http_code": None, "job_id": None, "queue_s": None, "gen_s": None,
        "download_s": None, "e2e_s": None, "bytes": None, "mp4_duration_s": None, "has_video": None,
        "has_audio": None, "server_wall_s": None, "polls": 0, "status_final": None, "pass": None,
        "error": None, "detail": None, "note": case.get("note"), "saved": None,
    }
    headers = {"Authorization": "Bearer definitely-the-wrong-key"} if case.get("badkey") else None
    if expect != "ok" and "edge" in case.get("tags", []):
        # these shapes are known to fail inside the worker and count against its error budget:
        # look at the live counter now rather than waiting for the next monitor sample
        apply_guard(ctx, server, liveness(server))
    if server in ctx.guard_tripped and (expect != "ok" or case["id"] in ctx.worker_error_cases):
        rec.update({"status_final": "skipped", "error": "skipped: worker-error guard tripped on this server", "pass": None})
        ctx.record(rec)
        return rec
    if a.pause_when_down:
        waited = 0
        while not ctx.stop and waited < a.job_timeout:
            lv = ctx.last_liveness.get(server)
            if not lv or (lv.get("http_code") == 200 and lv.get("model_ready") is not False):
                break
            if waited == 0:
                ctx.log(f"{phase:10} {server:6} pausing {case['id']}: server not ready (HTTP {lv.get('http_code')} ready={lv.get('model_ready')})")
            time.sleep(5)
            waited += 5
        if waited:
            rec["paused_s"] = waited
        if waited >= a.job_timeout:
            rec.update({"status_final": "skipped", "error": f"skipped: server down for {waited}s", "pass": None})
            ctx.record(rec)
            return rec
    rec["inflight_at_submit"] = ctx.inflight_inc(server)
    rec["liveness_queue_at_submit"] = (ctx.last_liveness.get(server) or {}).get("queue_size")
    t0 = time.perf_counter()
    job_id = None
    try:
        code, data, el, hdrs = http("POST", base + path, body=body, headers=headers, timeout=a.submit_timeout)
        rec["submit_s"], rec["http_code"] = round(el, 3), code
        accepted = code in (200, 201, 202) and isinstance(data, dict) and data.get("id")
        if expect == "reject":
            rec["detail"] = short(data)
            rec["pass"] = 400 <= code < 500
            if rec["pass"]:
                with ctx.lock:
                    ctx.admission_rejected.add(case["id"])
            if not accepted:
                rec["e2e_s"] = round(time.perf_counter() - t0, 3)
                return rec
            rec["note"] = (rec["note"] or "") + " | UNEXPECTEDLY ACCEPTED, followed to completion"
        if not accepted:
            rec["detail"] = short(data)
            if code == 200 and isinstance(data, (bytes, bytearray)) and data[:12].find(b"ftyp") >= 0:
                # synchronous build: the POST returned the video itself
                rec["status_final"] = "sync"
                rec["gen_s"] = round(el, 3)
                return _finish_with_video(rec, data, t0, ctx, case, hdrs)
            rec["error"] = f"submit HTTP {code}: {short(data, 160)}"
            rec["pass"] = None if expect == "probe" else False
            rec["e2e_s"] = round(time.perf_counter() - t0, 3)
            return rec
        job_id = data["id"]
        rec["job_id"] = job_id
        with ctx.lock:
            ctx.active_jobs[server][job_id] = time.time()
        t_accept = time.perf_counter()
        t_start, last, polls, trail, st = None, None, 0, [], {}
        while True:
            time.sleep(a.poll_interval)
            c2, st, el2, _ = http("GET", f"{base}/v1/videos/generations/{job_id}", timeout=30)
            polls += 1
            if c2 == 200 and isinstance(st, dict):
                s = st.get("status")
                if s != last:
                    trail.append([round(time.perf_counter() - t0, 2), s])
                    last = s
                if s not in NOT_STARTED and t_start is None:
                    t_start = time.perf_counter()
                if s in TERMINAL:
                    break
            else:
                rec["poll_errors"] = rec.get("poll_errors", 0) + 1
                if rec["poll_errors"] >= 20:
                    rec["error"] = f"status endpoint failing: HTTP {c2} {short(st, 120)}"
                    rec["status_final"] = "lost"
                    rec["pass"] = False
                    return rec
            if ctx.stop or time.perf_counter() - t_accept > a.job_timeout:
                rec["error"] = f"timeout after {a.job_timeout:.0f}s (last status {last})" if not ctx.stop else "aborted"
                rec["status_final"] = "timeout" if not ctx.stop else "aborted"
                rec["pass"] = False
                http("POST", f"{base}/v1/videos/generations/{job_id}/cancel", timeout=30)
                return rec
        t_done = time.perf_counter()
        rec["polls"], rec["status_trail"], rec["status_final"] = polls, trail, st.get("status")
        rec["queue_s"] = round((t_start or t_done) - t_accept, 3)
        rec["gen_s"] = round(t_done - (t_start or t_accept), 3)
        if st.get("created_at") and st.get("completed_at"):
            rec["server_wall_s"] = st["completed_at"] - st["created_at"]
        if st.get("status") != "completed":
            if "processing_error" in json.dumps(st.get("error") or ""):
                with ctx.lock:
                    ctx.worker_error_cases.add(case["id"])
            rec["error"] = f"job {st.get('status')}: {short(st.get('error'), 200)}"
            rec["detail"] = short(st.get("error"))
            rec["pass"] = None if expect == "probe" else False
            rec["e2e_s"] = round(time.perf_counter() - t0, 3)
            return rec
        c3, vid, el3, h3 = http("GET", f"{base}/v1/videos/generations/{job_id}/download", timeout=a.download_timeout)
        rec["download_s"] = round(el3, 3)
        if c3 != 200 or not isinstance(vid, (bytes, bytearray)):
            rec["error"] = f"download HTTP {c3}: {short(vid, 160)}"
            rec["pass"] = False
            rec["e2e_s"] = round(time.perf_counter() - t0, 3)
            return rec
        rec = _finish_with_video(rec, vid, t0, ctx, case, h3)
        if a.delete_after:
            with ctx.lock:   # drop it from the in-flight list first so a 404 from the status hammer reads as "deleted"
                ctx.active_jobs[server].pop(job_id, None)
            c4, _, el4, _ = http("DELETE", f"{base}/v1/videos/generations/{job_id}", timeout=60)
            rec["delete_http"], rec["delete_s"] = c4, round(el4, 3)
        return rec
    finally:
        rec["e2e_s"] = rec["e2e_s"] or round(time.perf_counter() - t0, 3)
        ctx.inflight_dec(server)
        if job_id:
            with ctx.lock:
                ctx.active_jobs[server].pop(job_id, None)
        ctx.record(rec)
        _log_rec(ctx, rec)


def _finish_with_video(rec, vid, t0, ctx, case, hdrs):
    dur, hv, ha = mp4_info(vid)
    rec.update({"bytes": len(vid), "mp4_duration_s": round(dur, 3) if dur else None, "has_video": hv,
                "has_audio": ha, "sha256": hashlib.sha256(vid).hexdigest()[:16],
                "x_generation_time": hdrs.get("x-generation-time")})
    problems = []
    if len(vid) < 50_000:
        problems.append(f"only {len(vid)} bytes")
    if not hv:
        problems.append("no video track")
    if not ha:
        problems.append("no audio track")
    want = case.get("duration")
    if dur is None:
        problems.append("no mvhd duration")
    elif want and case.get("expect") == "ok" and abs(dur - want) > 0.6:
        problems.append(f"duration {dur:.2f}s vs requested {want}s")
    good = not problems
    if problems:
        rec["error"] = "; ".join(problems)
    expect = case.get("expect", "ok")
    rec["pass"] = None if expect == "probe" else (False if expect == "reject" else good)
    rec["saved"] = ctx.maybe_save(rec, vid, good)
    rec["e2e_s"] = round(time.perf_counter() - t0, 3)
    return rec


def _log_rec(ctx, rec):
    tag = {True: "ok  ", False: "FAIL", None: "info"}[rec["pass"]]
    if rec["expect"] == "reject":
        ctx.log(f"{rec['phase']:10} {rec['server']:6} {tag} {rec['case_id']:24} HTTP {rec['http_code']} {short(rec.get('detail'), 110)}")
    else:
        ctx.log(f"{rec['phase']:10} {rec['server']:6} {tag} {rec['case_id']:24} HTTP {rec['http_code']} "
                f"q={rec['queue_s'] or 0:5.1f}s gen={rec['gen_s'] or 0:5.1f}s e2e={rec['e2e_s'] or 0:6.1f}s "
                f"infl={rec['inflight_at_submit']} dur={rec['mp4_duration_s']} {('ERR ' + rec['error']) if rec['error'] else ''}")


# --------------------------------------------------------------------------- monitor


def liveness(server):
    code, data, el, _ = http("GET", f"{SERVERS[server]}/tt-liveness", timeout=15)
    s = {"t": now_iso(), "server": server, "http_code": code, "latency_s": round(el, 3)}
    if code == 200 and isinstance(data, dict):
        w = (data.get("worker_info") or {}).get("0") or {}
        s.update({"queue_size": data.get("queue_size"), "model_ready": data.get("model_ready"),
                  "canary_state": data.get("canary_state"), "worker_alive": w.get("is_alive"),
                  "error_count": w.get("error_count"), "restart_count": w.get("restart_count"),
                  "worker_start_time": w.get("start_time"), "ready_time": w.get("ready_time")})
    else:
        s["error"] = short(data, 120)
    return s


def apply_guard(ctx, server, s):
    """Trip the per-server guard when the worker error_count has risen by --max-worker-errors since the run began."""
    ec = s.get("error_count")
    if ec is None:
        return
    base = ctx.error_base.setdefault(server, ec)
    if ec - base >= ctx.args.max_worker_errors and server not in ctx.guard_tripped:
        ctx.guard_tripped.add(server)
        ctx.log(f"GUARD {server}: worker error_count rose {base} -> {ec}; no more negative/probe or previously-failing cases will be sent to it")


def monitor_loop(ctx):
    while not ctx.stop:
        for server in SERVERS:
            s = liveness(server)
            s["inflight"] = ctx.inflight[server]
            s["phase"] = ctx.phase
            ctx.last_liveness[server] = s
            ctx.sample(s)
            apply_guard(ctx, server, s)
            if s["http_code"] != 200 or s.get("model_ready") is False or s.get("canary_state") not in (None, "healthy"):
                ctx.log(f"MONITOR {server}: HTTP {s['http_code']} ready={s.get('model_ready')} canary={s.get('canary_state')} {s.get('error') or ''}")
        for _ in range(int(ctx.args.monitor_interval * 10)):
            if ctx.stop:
                break
            time.sleep(0.1)


# ---------------------------------------------------------------------------- phases


def by_server(cases):
    d = {s: [] for s in SERVERS}
    for c in cases:
        d[c["server"]].append(c)
    return d


def phase_serial(ctx, name, cases, budget_s):
    """Each server runs its cases one at a time; the two servers run concurrently."""
    ctx.phase = name
    deadline = time.time() + budget_s
    ctx.log(f"=== phase {name}: {len(cases)} cases, budget {budget_s/60:.1f} min")

    def worker(server, cs):
        rnd = random.Random(hash((name, server)) & 0xFFFF)
        for case in cs:
            if ctx.stop:
                return
            if time.time() > deadline:
                ctx.log(f"{name} {server}: budget exhausted, skipping {case['id']} and the rest")
                return
            reps = 2 if "det" in case.get("tags", []) else 1
            for _ in range(reps):
                run_case(case, name, ctx, worker=server, rnd=rnd)

    ts = [threading.Thread(target=worker, args=(s, cs), daemon=True) for s, cs in by_server(cases).items() if cs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def lifecycle_checks(ctx, server, case):
    """cancel, delete, double download, status of an unknown id."""
    base = SERVERS[server]
    rnd = random.Random(1)
    path, payload, _ = build_payload(case, rnd)
    body = json.dumps(payload).encode()
    out = {"kind": "lifecycle", "phase": "validate", "server": server, "case_id": "LC-" + case["id"], "t": now_iso(), "checks": {}}

    # cancel a fresh job
    code, data, _, _ = http("POST", base + path, body=body, timeout=ctx.args.submit_timeout)
    if code == 202 and isinstance(data, dict):
        jid = data["id"]
        time.sleep(1.0)
        c, d, el, _ = http("POST", f"{base}/v1/videos/generations/{jid}/cancel", timeout=30)
        final = None
        for _ in range(60):
            time.sleep(1)
            c2, st, _, _ = http("GET", f"{base}/v1/videos/generations/{jid}", timeout=30)
            if isinstance(st, dict) and st.get("status") in TERMINAL:
                final = st.get("status")
                break
        out["checks"]["cancel"] = {"cancel_http": c, "cancel_detail": short(d, 120), "final_status": final,
                                   "pass": c in (200, 202, 204) and final in ("cancelled", "completed")}
    else:
        out["checks"]["cancel"] = {"submit_http": code, "pass": False, "detail": short(data, 120)}

    # complete one, download twice at once, then delete it
    code, data, _, _ = http("POST", base + path, body=body, timeout=ctx.args.submit_timeout)
    if code == 202 and isinstance(data, dict):
        jid = data["id"]
        for _ in range(int(ctx.args.job_timeout)):
            time.sleep(1)
            c2, st, _, _ = http("GET", f"{base}/v1/videos/generations/{jid}", timeout=30)
            if isinstance(st, dict) and st.get("status") in TERMINAL:
                break
        res = [None, None]

        def dl(i):
            res[i] = http("GET", f"{base}/v1/videos/generations/{jid}/download", timeout=ctx.args.download_timeout)

        th = [threading.Thread(target=dl, args=(i,)) for i in range(2)]
        [t.start() for t in th]
        [t.join() for t in th]
        same = all(r and r[0] == 200 for r in res) and res[0][1] == res[1][1]
        out["checks"]["double_download"] = {"codes": [r[0] for r in res], "bytes": [len(r[1]) if isinstance(r[1], (bytes, bytearray)) else None for r in res], "identical": same, "pass": same}
        c, d, _, _ = http("DELETE", f"{base}/v1/videos/generations/{jid}", timeout=30)
        c2, d2, _, _ = http("GET", f"{base}/v1/videos/generations/{jid}", timeout=30)
        c3, d3, _, _ = http("GET", f"{base}/v1/videos/generations/{jid}/download", timeout=30)
        out["checks"]["delete"] = {"delete_http": c, "status_after": c2, "download_after": c3,
                                   "pass": c in (200, 202, 204) and c2 == 404 and c3 == 404}
    else:
        out["checks"]["double_download"] = {"pass": False, "detail": short(data, 120)}
    c, d, _, _ = http("GET", f"{base}/v1/videos/generations/00000000-0000-0000-0000-000000000000", timeout=30)
    out["checks"]["unknown_id"] = {"http": c, "pass": c == 404}
    out["pass"] = all(v.get("pass") for v in out["checks"].values())
    ctx.record(out)
    ctx.log(f"validate   {server:6} {'ok  ' if out['pass'] else 'FAIL'} lifecycle: " + ", ".join(f"{k}={'ok' if v.get('pass') else 'FAIL'}" for k, v in out["checks"].items()))


def phase_validate(ctx, negatives, lifecycle_cases, budget_s):
    ctx.phase = "validate"
    deadline = time.time() + budget_s
    ctx.log(f"=== phase validate: {len(negatives)} negative/probe cases + lifecycle, budget {budget_s/60:.1f} min")

    def worker(server, cs):
        rnd = random.Random(2)
        for case in cs:
            if ctx.stop or time.time() > deadline:
                return
            run_case(case, "validate", ctx, worker=server, rnd=rnd)
        if lifecycle_cases.get(server) and not ctx.stop and time.time() < deadline:
            lifecycle_checks(ctx, server, lifecycle_cases[server])

    ts = [threading.Thread(target=worker, args=(s, cs), daemon=True) for s, cs in by_server(negatives).items() if cs]
    [t.start() for t in ts]
    [t.join() for t in ts]


def phase_sustain(ctx, positives, negatives, budget_s, workers):
    ctx.phase = "sustain"
    deadline = time.time() + budget_s
    ctx.log(f"=== phase sustain: budget {budget_s/60:.1f} min, workers {workers}")
    pos = by_server([c for c in positives if c.get("weight", 0) > 0])
    if ctx.args.inject == "all":
        neg = by_server(negatives)   # customer-like: every mistake shape, including ones that fail inside the worker
    else:
        neg = by_server([c for c in negatives if c["id"] in ctx.admission_rejected])
    ctx.log(f"sustain: injecting only negatives that validate saw rejected at admission: { {s: len(v) for s, v in neg.items()} }")
    threads = []

    def worker(server, i):
        rnd = random.Random(1000 + i)
        cs = pos[server]
        w = [c["weight"] for c in cs]
        while not ctx.stop and time.time() < deadline:
            run_case(rnd.choices(cs, weights=w)[0], "sustain", ctx, worker=f"{server}-w{i}", rnd=rnd)

    def injector(server):
        rnd = random.Random(77)
        while not ctx.stop and time.time() < deadline:
            for _ in range(int(ctx.args.neg_every * 10)):
                if ctx.stop or time.time() >= deadline:
                    return
                time.sleep(0.1)
            run_case(rnd.choice(neg[server]), "sustain-neg", ctx, worker=f"{server}-neg", rnd=rnd)

    def hammer(server):
        """Hit the status endpoint of a live job hard: 3 threads x 30 polls at 10 Hz."""
        base = SERVERS[server]
        while not ctx.stop and time.time() < deadline:
            for _ in range(int(ctx.args.hammer_every * 10)):
                if ctx.stop or time.time() >= deadline:
                    return
                time.sleep(0.1)
            with ctx.lock:
                jobs = list(ctx.active_jobs[server])
            if not jobs:
                continue
            jid = jobs[-1]   # newest in-flight job: least likely to finish (and be deleted) mid-hammer
            lat, codes = [], Counter()

            def burst():
                for _ in range(30):
                    c, _, el, _ = http("GET", f"{base}/v1/videos/generations/{jid}", timeout=30)
                    lat.append(el)
                    if c == 404:
                        with ctx.lock:
                            gone = jid not in ctx.active_jobs[server]
                        # the harness deletes a job once its MP4 is downloaded; 404 after that is the right answer
                        c = "404-after-delete" if gone else 404
                    codes[c] += 1
                    time.sleep(0.1)

            th = [threading.Thread(target=burst) for _ in range(3)]
            [t.start() for t in th]
            [t.join() for t in th]
            lat.sort()
            rec = {"kind": "hammer", "phase": "sustain", "server": server, "t": now_iso(), "job_id": jid, "n": len(lat),
                   "codes": dict(codes), "p50_s": round(lat[len(lat) // 2], 3), "p95_s": round(lat[int(len(lat) * 0.95) - 1], 3),
                   "max_s": round(lat[-1], 3), "pass": codes.get(200, 0) + codes.get("404-after-delete", 0) == len(lat)}
            ctx.record(rec)
            ctx.log(f"sustain    {server:6} {'ok  ' if rec['pass'] else 'FAIL'} status hammer: {len(lat)} polls, codes {dict(codes)}, p50 {rec['p50_s']}s p95 {rec['p95_s']}s")

    def lister(server):
        base = SERVERS[server]
        while not ctx.stop and time.time() < deadline:
            for _ in range(int(ctx.args.list_every * 10)):
                if ctx.stop or time.time() >= deadline:
                    return
                time.sleep(0.1)
            c, d, el, _ = http("GET", f"{base}/v1/videos/jobs", timeout=300)
            n = len(d) if isinstance(d, list) else None
            size = len(json.dumps(d)) if isinstance(d, (list, dict)) else (len(d) if isinstance(d, (bytes, bytearray)) else None)
            rec = {"kind": "jobs-list", "phase": "sustain", "server": server, "t": now_iso(), "http_code": c, "latency_s": round(el, 3),
                   "n_jobs": n, "bytes": size, "pass": c == 200}
            ctx.record(rec)
            ctx.log(f"sustain    {server:6} {'ok  ' if rec['pass'] else 'FAIL'} jobs list: HTTP {c} {n} jobs {size and size/1e6:.1f} MB in {el:.1f}s")

    for server, n in workers.items():
        if not pos.get(server):
            continue
        for i in range(n):
            threads.append(threading.Thread(target=worker, args=(server, i), daemon=True))
        if neg[server]:
            threads.append(threading.Thread(target=injector, args=(server,), daemon=True))
        threads.append(threading.Thread(target=hammer, args=(server,), daemon=True))
        threads.append(threading.Thread(target=lister, args=(server,), daemon=True))
    for t in threads:
        t.start()
        time.sleep(0.3)
    for t in threads:
        t.join()


def phase_burst(ctx, positives, budget_s, sizes):
    ctx.phase = "burst"
    ctx.log(f"=== phase burst: sizes {sizes}, budget {budget_s/60:.1f} min")
    pos = by_server([c for c in positives if c.get("weight", 0) > 0 and c.get("duration", 5) <= 10])
    threads = []
    for server, n in sizes.items():
        cs = pos.get(server)
        if not cs:
            continue
        rnd = random.Random(5)
        picks = [cs[i % len(cs)] for i in range(n)]
        rnd.shuffle(picks)
        go = threading.Barrier(n)

        def one(case, i, go=go, server=server):
            try:
                go.wait(timeout=30)
            except threading.BrokenBarrierError:
                pass
            run_case(case, "burst", ctx, worker=f"{server}-b{i}", rnd=random.Random(i))

        threads += [threading.Thread(target=one, args=(c, i), daemon=True) for i, c in enumerate(picks)]
    for t in threads:
        t.start()
    deadline = time.time() + budget_s
    for t in threads:
        t.join(timeout=max(1, deadline - time.time()))
    alive = [t for t in threads if t.is_alive()]
    if alive:
        ctx.log(f"burst: {len(alive)} requests still in flight after the budget; they finish in the background")


# ------------------------------------------------------------------------------ main


def server_info(server):
    base = SERVERS[server]
    c, m, _, _ = http("GET", f"{base}/v1/models", timeout=20)
    return {"url": base, "models_http": c, "model": (m.get("data") or [{}])[0].get("id") if isinstance(m, dict) else None,
            "liveness": liveness(server)}


def write_csv(out_dir):
    rows = []
    with open(os.path.join(out_dir, "results.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("kind") == "request":
                r.pop("status_trail", None)
                rows.append(r)
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30, help="total wall-clock budget (default 30)")
    ap.add_argument("--workers", default="4,3", help="concurrent workers in the sustain phase: fl2va,ref2va (default 4,3)")
    ap.add_argument("--burst", default="12,8", help="burst sizes: fl2va,ref2va (default 12,8)")
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--only", help="comma list of case ids (or prefixes) to restrict every phase to")
    ap.add_argument("--servers", default="fl2va,ref2va", help="which servers to drive (default both)")
    ap.add_argument("--skip", default="", help="comma list of phases to skip (baseline,validate,sustain,burst,recovery,edge)")
    ap.add_argument("--poll-interval", type=float, default=0.5)
    ap.add_argument("--job-timeout", type=float, default=600, help="seconds to wait for one job (default 600)")
    ap.add_argument("--submit-timeout", type=float, default=300)
    ap.add_argument("--download-timeout", type=float, default=300)
    ap.add_argument("--monitor-interval", type=float, default=10)
    ap.add_argument("--neg-every", type=float, default=25, help="seconds between injected negative requests (sustain)")
    ap.add_argument("--hammer-every", type=float, default=60, help="seconds between status-poll hammers (sustain)")
    ap.add_argument("--list-every", type=float, default=120, help="seconds between /v1/videos/jobs probes (sustain)")
    ap.add_argument("--keep", choices=["sample", "all", "none"], default="sample", help="which output videos to keep")
    ap.add_argument("--inject", choices=["safe", "all"], default="safe",
                    help="which negative shapes the sustain injector may send: 'safe' = only ones already refused at admission "
                         "(default); 'all' = every reject/probe shape incl. ones known to fail inside the worker (customer-like)")
    ap.add_argument("--max-worker-errors", type=int, default=2,
                    help="once a server's worker error_count has risen by this much, stop sending it negative/probe cases "
                         "and cases that already failed in its worker (observed: ~6 worker errors kill the deployment)")
    ap.add_argument("--no-pause-when-down", dest="pause_when_down", action="store_false",
                    help="keep submitting while a server reports not-ready (default: pause until it is back, up to --job-timeout)")
    ap.add_argument("--no-delete-after", dest="delete_after", action="store_false",
                    help="leave finished jobs on the server (default: DELETE each job after its MP4 is downloaded)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    with open(args.cases) as fh:
        cases = json.load(fh)["cases"]
    servers = [s for s in args.servers.split(",") if s]
    for s in list(SERVERS):
        if s not in servers:
            del SERVERS[s]
    cases = [c for c in cases if c["server"] in SERVERS]
    if args.only:
        pref = [p.strip() for p in args.only.split(",") if p.strip()]
        cases = [c for c in cases if any(c["id"] == p or c["id"].startswith(p) for p in pref)]
    for c in cases:
        for key in ("images", "videos", "audios"):
            for n in c.get(key) or []:
                if not os.path.exists(os.path.join(ASSETS, n)):
                    sys.exit(f"missing asset {n} for case {c['id']} -- run make_assets.py --seed-videos")
    positives = [c for c in cases if c.get("expect") == "ok" and "edge" not in c.get("tags", [])]
    negatives = [c for c in cases if c.get("expect") in ("reject", "probe") and "edge" not in c.get("tags", [])]
    core = [c for c in positives if "core" in c.get("tags", []) or "det" in c.get("tags", [])]
    edge = [c for c in cases if "edge" in c.get("tags", [])]
    skip = set(p.strip() for p in args.skip.split(",") if p.strip())
    workers = dict(zip(("fl2va", "ref2va"), (int(x) for x in args.workers.split(","))))
    bursts = dict(zip(("fl2va", "ref2va"), (int(x) for x in args.burst.split(","))))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    ctx = Ctx(args, out_dir)
    ctx.phase = "preflight"
    signal.signal(signal.SIGINT, lambda *_: setattr(ctx, "stop", True) or ctx.log("SIGINT: stopping after in-flight requests"))
    signal.signal(signal.SIGTERM, lambda *_: setattr(ctx, "stop", True))

    ctx.log(f"run {run_id}: {args.minutes:g} min, servers {list(SERVERS)}, {len(cases)} cases "
            f"({len(positives)} positive, {len(negatives)} negative/probe, {len(core)} core, {len(edge)} edge)")
    info_start = {s: server_info(s) for s in SERVERS}
    for s, i in info_start.items():
        ctx.log(f"  {s}: {i['url']} model={i['model']} liveness={ {k: i['liveness'].get(k) for k in ('http_code','queue_size','model_ready','canary_state','error_count','restart_count')} }")
        if i["liveness"].get("http_code") != 200 or i["liveness"].get("model_ready") is False:
            ctx.log(f"  WARNING: {s} is not ready; continuing anyway (results will show it)")
    run_meta = {"run": run_id, "started_at": now_iso(), "args": vars(args), "servers_start": info_start,
                "cases": [c["id"] for c in cases], "phase_fraction": PHASE_FRACTION, "key_fingerprint": hashlib.sha256(KEY.encode()).hexdigest()[:8]}
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(run_meta, fh, indent=1)

    mon = threading.Thread(target=monitor_loop, args=(ctx,), daemon=True)
    mon.start()
    t_start = time.time()
    total = args.minutes * 60
    end = t_start + total
    budget = {p: total * f for p, f in PHASE_FRACTION.items()}
    try:
        if "baseline" not in skip:
            phase_serial(ctx, "baseline", core, budget["baseline"])
        if "validate" not in skip and not ctx.stop:
            lc = {s: next((c for c in core if c["server"] == s and c.get("duration", 5) >= 10), None) or next((c for c in core if c["server"] == s), None) for s in SERVERS}
            phase_validate(ctx, negatives, lc, budget["validate"])
        if "sustain" not in skip and not ctx.stop:
            reserve = sum(budget[p] for p in ("burst", "recovery", "edge") if p not in skip)
            inject_pool = negatives + ([c for c in edge if c.get("expect") != "ok"] if args.inject == "all" else [])
            phase_sustain(ctx, positives, inject_pool, max(60, end - time.time() - reserve), workers)
        if "burst" not in skip and not ctx.stop:
            phase_burst(ctx, positives, budget["burst"], bursts)
        if "recovery" not in skip and not ctx.stop:
            phase_serial(ctx, "recovery", [c for c in core if "det" not in c.get("tags", [])], max(60, end - time.time() - (budget["edge"] if "edge" not in skip else 0)))
        if "edge" not in skip and not ctx.stop:
            phase_serial(ctx, "edge", edge, max(60, budget["edge"]))
    finally:
        ctx.phase = "done"
        ctx.stop = True
        mon.join(timeout=20)
        run_meta["finished_at"] = now_iso()
        run_meta["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        run_meta["servers_end"] = {s: server_info(s) for s in SERVERS}
        run_meta["guard_tripped"] = sorted(ctx.guard_tripped)
        run_meta["worker_error_cases"] = sorted(ctx.worker_error_cases)
        with open(os.path.join(out_dir, "run.json"), "w") as fh:
            json.dump(run_meta, fh, indent=1)
        write_csv(out_dir)
        ctx.log(f"finished in {run_meta['elapsed_min']} min -> {out_dir}")
        try:
            sys.path.insert(0, HERE)
            import report

            text = report.render(out_dir)
            with open(os.path.join(out_dir, "report.md"), "w") as fh:
                fh.write(text)
            print("\n" + text)
        except Exception as e:  # noqa: BLE001
            ctx.log(f"report generation failed: {type(e).__name__}: {e} (run: python3 report.py {out_dir})")


if __name__ == "__main__":
    main()
