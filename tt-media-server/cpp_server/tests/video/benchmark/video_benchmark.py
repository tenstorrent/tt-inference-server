#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Benchmark of the video service over HTTP (works against the C++ and the Python server).

It follows the shape of test_module/benchmark_tests/video_benchmark_tests.py: submit a
job (202), poll it until `completed`, download the mp4, and report the elapsed time and
`inference_steps_per_second`. On top of that it measures the server itself:

  sequential  N videos one after another (single user, like the Python benchmark)
  burst       K videos submitted at once: queue wait, total throughput
  control     latency of the cheap endpoints (POST create, GET status, GET list,
              /tt-liveness) with many jobs in the store; needs a fast runner
              (the mock runner) so the device time does not hide the server cost
  throughput  M jobs through the whole pipeline with the mock runner: jobs per second

When the server exposes /metrics (the C++ server does), the deltas of the tt_video_*
histograms before and after each phase are added to the output.

Usage:
  python3 video_benchmark.py --phase sequential --calls 5 --steps 40 --out result.json
"""

import argparse
import http.client
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.parse

GEN = "/v1/videos/generations"


# ---------------------------------------------------------------- http helpers
class Client:
    """HTTP client of a server on this machine (the benchmark runs next to it).

    The host is one of two constants, chosen by comparison, so the address that was
    passed in never reaches the connection.
    """

    def __init__(self, base, key):
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise SystemExit(
                f"{base!r}: only http://127.0.0.1:<port> or http://localhost:<port> is accepted"
            )
        self.host = "127.0.0.1" if parsed.hostname == "127.0.0.1" else "localhost"
        self.port = int(parsed.port or 80)
        self.key = key

    def call(self, method, path, body=None, auth=True, timeout=120, read=True):
        """Returns (status, headers, bytes, seconds), for any status."""
        data = None if body is None else json.dumps(body).encode()
        headers = {}
        if auth and self.key:
            headers["authorization"] = f"Bearer {self.key}"
        if data is not None:
            headers["content-type"] = "application/json"
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        t0 = time.perf_counter()
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read() if read else b""
            return resp.status, resp.headers, raw, time.perf_counter() - t0
        finally:
            conn.close()

    def json(self, *a, **k):
        st, h, raw, dt = self.call(*a, **k)
        try:
            return st, (json.loads(raw) if raw.strip() else {}), dt
        except ValueError:
            return st, {}, dt


def pct(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": pct(values, 0.50),
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "max": max(values),
    }


# ---------------------------------------------------------------- /metrics
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$")


def scrape(client):
    """tt_video_* samples as {(name, labels): value}; {} if /metrics is not there."""
    st, _, raw, _ = client.call("GET", "/metrics", auth=False, timeout=10)
    if st != 200:
        st, _, raw, _ = client.call("GET", "/metrics", timeout=10)
    if st != 200:
        return {}
    out = {}
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith(("tt_video_", "tt_media_server_video_")):
            continue
        m = _LINE.match(line.strip())
        if m:
            try:
                out[(m.group(1), m.group(2) or "")] = float(m.group(3))
            except ValueError:
                pass
    return out


def hist_delta(before, after, name, label_filter=""):
    """mean and bucket-interpolated quantiles of the observations between two scrapes."""

    def pick(snap, suffix):
        return {
            lbl: v
            for (n, lbl), v in snap.items()
            if n == name + suffix and label_filter in lbl
        }

    cnt = sum(pick(after, "_count").values()) - sum(pick(before, "_count").values())
    tot = sum(pick(after, "_sum").values()) - sum(pick(before, "_sum").values())
    if cnt <= 0:
        return {"n": 0}
    # cumulative buckets summed over matching label sets
    def buckets(snap):
        acc = {}
        for lbl, v in pick(snap, "_bucket").items():
            m = re.search(r'le="([^"]+)"', lbl)
            if m:
                le = float("inf") if m.group(1) == "+Inf" else float(m.group(1))
                acc[le] = acc.get(le, 0.0) + v
        return acc

    b0, b1 = buckets(before), buckets(after)
    edges = sorted(b1)
    cum = [(le, b1[le] - b0.get(le, 0.0)) for le in edges]

    def quant(q):
        target = q * cnt
        prev_le, prev_c = 0.0, 0.0
        for le, c in cum:
            if c >= target:
                if math.isinf(le):
                    return prev_le
                span = c - prev_c
                frac = (target - prev_c) / span if span > 0 else 1.0
                return prev_le + (le - prev_le) * frac
            prev_le, prev_c = le, c
        return prev_le

    return {"n": int(cnt), "mean": tot / cnt, "p50": quant(0.5), "p95": quant(0.95)}


def generic_deltas(before, after):
    """Every video series that changed between two scrapes (both servers).

    Histograms/summaries: count, sum and mean of the new observations. Counters and
    gauges: the change, and the value at the end. Buckets are left out.
    """
    out = {}
    names = {(n, l) for (n, l) in list(before) + list(after)}
    for (n, l) in sorted(names):
        if n.endswith("_bucket") or n.endswith("_created") or n.endswith("_timestamp"):
            continue
        b, a = before.get((n, l), 0.0), after.get((n, l), 0.0)
        key = n + l
        if n.endswith("_sum"):
            base = n[:-4] + "_count" + l
            cnt = after.get((n[:-4] + "_count", l), 0.0) - before.get((n[:-4] + "_count", l), 0.0)
            if cnt > 0:
                out[n[:-4] + l] = {"count": cnt, "sum": a - b, "mean": (a - b) / cnt}
        elif n.endswith("_count"):
            continue
        elif a != b:
            out[key] = {"delta": a - b, "value": a}
    return out


def server_side(before, after):
    if not before or not after:
        return {}
    res = {
        "queue_wait_s": hist_delta(before, after, "tt_video_queue_wait_seconds"),
        "generation_s": hist_delta(before, after, "tt_video_generation_seconds"),
        "e2e_ok_s": hist_delta(
            before, after, "tt_video_e2e_seconds", 'outcome="ok"'
        ),
    }
    for phase in ("request_write", "shm_enqueue", "response_handling"):
        res[f"overhead_{phase}_s"] = hist_delta(
            before, after, "tt_video_server_overhead_seconds", f'phase="{phase}"'
        )
    return res


# ---------------------------------------------------------------- video flow
def job_body(prompt, steps, extra=None):
    body = {"prompt": prompt}
    if steps is not None:
        body["num_inference_steps"] = steps
    body.update(extra or {})
    return body


def run_job(client, prompt, steps, poll, timeout, download=True, extra=None):
    """One full job. Returns a dict of client-side timings (seconds)."""
    t_start = time.perf_counter()
    st, j, t_submit = client.json("POST", GEN, job_body(prompt, steps, extra))
    rec = {"submit_http": st, "submit_s": t_submit, "ok": False}
    if st != 202 or "id" not in j:
        rec["elapsed_s"] = time.perf_counter() - t_start
        return rec
    jid = rec["id"] = j["id"]
    t_claim = None
    last = j.get("status")
    polls = 0
    deadline = t_start + timeout
    status = last
    while time.perf_counter() < deadline:
        time.sleep(poll)
        st, g, _ = client.json("GET", f"{GEN}/{jid}")
        polls += 1
        status = g.get("status") if st == 200 else f"http{st}"
        if status == "in_progress" and t_claim is None:
            t_claim = time.perf_counter()
        if status in ("completed", "failed", "cancelled"):
            break
    t_done = time.perf_counter()
    rec["polls"] = polls
    rec["final_status"] = status
    rec["elapsed_s"] = t_done - t_start  # submit -> terminal status seen
    if t_claim is not None:
        rec["seen_in_progress_s"] = t_claim - t_start
        rec["run_s"] = t_done - t_claim
    if status != "completed":
        return rec
    if download:
        st, h, raw, dt = client.call("GET", f"{GEN}/{jid}/download", timeout=300)
        rec["download_http"] = st
        rec["download_s"] = dt
        rec["size_bytes"] = len(raw)
        rec["download_mb_s"] = (len(raw) / 1e6) / dt if dt > 0 else 0
        rec["x_generation_time"] = h.get("X-Generation-Time") if h else None
        rec["ok"] = st == 200 and len(raw) > 0
    else:
        rec["ok"] = True
    rec["total_s"] = time.perf_counter() - t_start
    return rec


# ---------------------------------------------------------------- phases
def phase_sequential(c, a):
    before = scrape(c)
    recs = []
    for i in range(a.calls):
        r = run_job(c, f"Test video generation {i + 1}", a.steps, a.poll, a.timeout)
        r["index"] = i + 1
        # same formula as the Python benchmark: steps / elapsed (elapsed = until completed)
        r["steps_per_second"] = (a.steps / r["elapsed_s"]) if r.get("ok") else 0
        print(
            f"  video {i + 1}/{a.calls}: ok={r['ok']} elapsed={r['elapsed_s']:.2f}s "
            f"submit={r['submit_s'] * 1000:.1f}ms download={r.get('download_s', 0):.2f}s",
            flush=True,
        )
        recs.append(r)
    after = scrape(c)
    ok = [r for r in recs if r["ok"]]
    return {
        "calls": a.calls,
        "ok": len(ok),
        "steps": a.steps,
        "elapsed_s": summarize([r["elapsed_s"] for r in ok]),
        "submit_ms": summarize([r["submit_s"] * 1000 for r in ok]),
        "download_s": summarize([r["download_s"] for r in ok]),
        "steps_per_second": summarize([r["steps_per_second"] for r in ok]),
        "ttft_python_def_s": statistics.fmean([r["elapsed_s"] for r in ok]) if ok else 0,
        "size_mb": summarize([r["size_bytes"] / 1e6 for r in ok]),
        "server": server_side(before, after),
        "metrics_delta": generic_deltas(before, after),
        "records": recs,
    }


def phase_burst(c, a):
    before = scrape(c)
    recs = [None] * a.burst
    t0 = time.perf_counter()

    def worker(i):
        recs[i] = run_job(c, f"Burst video {i + 1}", a.steps, a.poll, a.timeout)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.burst)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    after = scrape(c)
    ok = [r for r in recs if r and r["ok"]]
    done_at = sorted(r["elapsed_s"] for r in ok)
    return {
        "burst": a.burst,
        "ok": len(ok),
        "steps": a.steps,
        "wall_s": wall,
        "videos_per_minute": (len(ok) / wall * 60) if wall > 0 else 0,
        "completion_s_sorted": done_at,
        "inter_completion_s": [
            done_at[i] - done_at[i - 1] for i in range(1, len(done_at))
        ],
        "elapsed_s": summarize([r["elapsed_s"] for r in ok]),
        "submit_ms": summarize([r["submit_s"] * 1000 for r in ok]),
        "server": server_side(before, after),
        "metrics_delta": generic_deltas(before, after),
        "records": recs,
    }


def timed_loop(fn, n):
    xs = []
    for _ in range(n):
        xs.append(fn() * 1000)
    return summarize(xs)


def phase_control(c, a):
    """Latency of the cheap endpoints while many finished jobs are in the store."""
    ids = []
    curve = []
    sizes = [int(x) for x in a.fill.split(",")]

    def fill_to(n):
        t0 = time.perf_counter()
        while len(ids) < n:
            st, j, _ = c.json("POST", GEN, job_body(f"fill {len(ids)}", a.steps))
            if st != 202:
                break
            ids.append(j["id"])
        wall = time.perf_counter() - t0
        pending = set(ids)
        deadline = time.time() + a.timeout
        while pending and time.time() < deadline:
            for jid in list(pending):
                st, g, _ = c.json("GET", f"{GEN}/{jid}")
                if st == 200 and g.get("status") in ("completed", "failed", "cancelled"):
                    pending.discard(jid)
            time.sleep(0.2)
        return wall, len(pending)

    for n in sizes:
        wall, unfinished = fill_to(n)
        st, _, raw, _ = c.call("GET", "/v1/videos/jobs")
        lst = timed_loop(lambda: c.call("GET", "/v1/videos/jobs")[3], 10)
        stat = timed_loop(lambda: c.call("GET", f"{GEN}/{ids[0]}")[3], 50)
        total = sum(
            v for (nm, _), v in scrape(c).items() if nm == "tt_video_jobs"
        )
        row = {
            "jobs_in_store": int(total) or len(ids),
            "unfinished": unfinished,
            "list_bytes": len(raw),
            "list_ms_p50": lst["p50"],
            "list_ms_p95": lst["p95"],
            "get_status_ms_p50": stat["p50"],
        }
        curve.append(row)
        print(f"  store {row['jobs_in_store']} jobs: GET /jobs {row['list_ms_p50']:.1f} ms, "
              f"{row['list_bytes'] / 1e3:.0f} KB", flush=True)
    jid = ids[0]
    out = {
        "list_scaling": curve,
        "liveness_ms": timed_loop(
            lambda: c.call("GET", "/tt-liveness", auth=False)[3], a.samples
        ),
        "get_status_ms": timed_loop(
            lambda: c.call("GET", f"{GEN}/{jid}")[3], a.samples
        ),
        "get_unknown_404_ms": timed_loop(
            lambda: c.call("GET", f"{GEN}/00000000-0000-4000-8000-000000000000")[3],
            a.samples,
        ),
        "metrics_scrape_ms": timed_loop(
            lambda: c.call("GET", "/metrics", auth=False)[3], 50
        ),
    }
    post = []
    for i in range(a.samples // 5):
        st, _, _, dt = c.call("POST", GEN, job_body("create latency", a.steps))
        if st == 202:
            post.append(dt * 1000)
    out["post_create_ms"] = summarize(post)
    return out


def phase_throughput(c, a):
    """M jobs through the whole pipeline with a fast runner, from several client threads."""
    before = scrape(c)
    lock = threading.Lock()
    counter = {"next": 0}
    recs = []

    def worker():
        while True:
            with lock:
                i = counter["next"]
                if i >= a.jobs:
                    return
                counter["next"] += 1
            r = run_job(c, f"throughput {i}", a.steps, a.poll, a.timeout, download=a.download)
            with lock:
                recs.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(a.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    after = scrape(c)
    ok = [r for r in recs if r["ok"]]
    return {
        "jobs": a.jobs,
        "clients": a.clients,
        "ok": len(ok),
        "rejected_429": sum(1 for r in recs if r.get("submit_http") == 429),
        "wall_s": wall,
        "jobs_per_second": len(ok) / wall if wall > 0 else 0,
        "elapsed_s": summarize([r["elapsed_s"] for r in ok]),
        "submit_ms": summarize([r["submit_s"] * 1000 for r in ok]),
        "server": server_side(before, after),
        "metrics_delta": generic_deltas(before, after),
    }


PHASES = {
    "sequential": phase_sequential,
    "burst": phase_burst,
    "control": phase_control,
    "throughput": phase_throughput,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default="your-secret-key")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--calls", type=int, default=3, help="sequential: videos")
    ap.add_argument("--burst", type=int, default=3, help="burst: videos at once")
    ap.add_argument("--fill", default="100,500,1000,2000", help="control: store sizes to measure at")
    ap.add_argument("--samples", type=int, default=200, help="control: calls per endpoint")
    ap.add_argument("--jobs", type=int, default=300, help="throughput: jobs")
    ap.add_argument("--clients", type=int, default=16, help="throughput: client threads")
    ap.add_argument("--no-download", dest="download", action="store_false")
    ap.add_argument("--poll", type=float, default=1.0, help="poll interval in seconds")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    c = Client(a.base_url, a.api_key)
    st, lv, _ = c.json("GET", "/tt-liveness", auth=False)
    if not (st == 200 and lv.get("model_ready") is True):
        print(f"server not ready at {a.base_url}: http{st} {lv}", file=sys.stderr)
        return 2
    print(f"phase {a.phase} against {a.base_url} (runner: {lv.get('runner_in_use')})", flush=True)
    result = PHASES[a.phase](c, a)
    result["phase"] = a.phase
    result["base_url"] = a.base_url
    result["runner"] = lv.get("runner_in_use")
    result["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    summary = {k: v for k, v in result.items() if k not in ("records",)}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
