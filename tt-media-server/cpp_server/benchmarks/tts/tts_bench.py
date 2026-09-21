#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Concurrency/ISL sweep against the TTS server, with per-chunk timings.

Drives POST /v1/audio/speech and times the arrival of each streamed audio
chunk, which is what the v2 latency contract is written against:

  FC   time from request send to the FIRST audio bytes (TTFC)
  SC   FC -> 2nd chunk
  TC   2nd -> 3rd chunk
  TC4+ every later inter-chunk gap

The server is a server: one bring-up serves the whole sweep. Only the client
concurrency and the input length change per point, so this does NOT need a
board reset between points.

  ./tts_bench.py --host localhost:8010 --concurrency 1,50,200,400,700 \
                 --isl 128,512,1024 --out results/

Writes results.json plus, if matplotlib is available, the two charts.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

# The v2 contract. Defaults mirror cpp_server defaults.hpp; override to match a
# server started with different TTS_*_MS values.
TARGETS = {"fc_p50": 100.0, "fc_p99": 125.0, "sc_p99": 180.0,
           "tc_p99": 400.0, "tc4_p99": 720.0}

# Input throughput is quoted in Mchars/hour, so tokens must be converted to
# characters. This is an ASSUMPTION, not a measurement: state it on every chart
# or the number cannot be compared with anyone else's.
CHARS_PER_TOKEN = 4.0


def synth_text(isl_tokens: int) -> str:
    """Filler text of roughly `isl_tokens` tokens. Audio quality is irrelevant
    here; what matters is that the prompt length is reproducible."""
    word = "the quick brown fox jumps over the lazy dog "
    approx_tokens_per_word = 1.0
    n = max(1, int(isl_tokens / approx_tokens_per_word / 9))
    return (word * n).strip()


def one_request(host: str, text: str, timeout: float) -> dict:
    """Send one request; return chunk arrival offsets in ms."""
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"http://{host}/v1/audio/speech", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    marks: list[float] = []
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                buf = resp.read(4096)
                if not buf:
                    break
                total += len(buf)
                marks.append((time.perf_counter() - t0) * 1000.0)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {"ok": False, "error": str(exc)[:120]}
    if not marks:
        return {"ok": False, "error": "no audio bytes"}
    return {"ok": True, "bytes": total, "marks": marks,
            "total_ms": (time.perf_counter() - t0) * 1000.0}


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_point(host: str, conc: int, isl: int, n: int, timeout: float) -> dict:
    """One (concurrency, ISL) point: n requests at `conc` in flight."""
    text = synth_text(isl)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        results = list(pool.map(lambda _: one_request(host, text, timeout), range(n)))
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    fc = [r["marks"][0] for r in ok]
    sc = [r["marks"][1] - r["marks"][0] for r in ok if len(r["marks"]) > 1]
    tc = [r["marks"][2] - r["marks"][1] for r in ok if len(r["marks"]) > 2]
    tc4 = [b - a for r in ok for a, b in zip(r["marks"][2:], r["marks"][3:])]

    rps = len(ok) / wall if wall > 0 else 0.0
    return {
        "concurrency": conc, "isl": isl, "requests": n, "ok": len(ok),
        "failed": len(results) - len(ok), "wall_s": wall, "rps": rps,
        # Mchars/hour. CHARS_PER_TOKEN is an assumption -- see the module docstring.
        "mchars_per_hour": rps * isl * CHARS_PER_TOKEN * 3600.0 / 1e6,
        "fc": {"p50": pct(fc, 50), "p99": pct(fc, 99), "pmax": pct(fc, 100)},
        "sc": {"p50": pct(sc, 50), "p99": pct(sc, 99), "pmax": pct(sc, 100)},
        "tc": {"p50": pct(tc, 50), "p99": pct(tc, 99), "pmax": pct(tc, 100)},
        "tc4": {"p50": pct(tc4, 50), "p99": pct(tc4, 99), "pmax": pct(tc4, 100)},
        "errors": sorted({r["error"] for r in results if not r["ok"]})[:3],
    }


def verdict(row: dict) -> str:
    """Against the v2 targets. Only meaningful when the server was started with
    matching TTS_*_MS values."""
    bad = []
    if row["fc"]["p50"] > TARGETS["fc_p50"]: bad.append("FC-P50")
    if row["fc"]["p99"] > TARGETS["fc_p99"]: bad.append("FC-P99")
    if row["sc"]["p99"] > TARGETS["sc_p99"]: bad.append("SC-P99")
    if row["tc"]["p99"] > TARGETS["tc_p99"]: bad.append("TC-P99")
    if row["tc4"]["p99"] > TARGETS["tc4_p99"]: bad.append("TC4-P99")
    return "PASS" if not bad else "FAIL:" + ",".join(bad)


def tables(rows: list[dict]) -> str:
    out = []
    hdr = (f"{'ISL':>6} {'Conc':>6} {'ok':>5} {'RPS':>7} {'Mchar/h':>8} "
           f"{'FC p50':>8} {'FC p99':>8} {'FC max':>8} "
           f"{'SC p99':>8} {'TC p99':>8} {'TC4 p99':>9}  verdict")
    out.append(hdr)
    out.append("-" * len(hdr))
    for r in rows:
        out.append(
            f"{r['isl']:>6} {r['concurrency']:>6} {r['ok']:>5} {r['rps']:>7.1f} "
            f"{r['mchars_per_hour']:>8.1f} "
            f"{r['fc']['p50']:>8.1f} {r['fc']['p99']:>8.1f} {r['fc']['pmax']:>8.1f} "
            f"{r['sc']['p99']:>8.1f} {r['tc']['p99']:>8.1f} {r['tc4']['p99']:>9.1f}"
            f"  {verdict(r)}")
    out.append("")
    out.append(f"targets: FC p50<={TARGETS['fc_p50']:.0f} p99<={TARGETS['fc_p99']:.0f}, "
               f"SC p99<={TARGETS['sc_p99']:.0f}, TC p99<={TARGETS['tc_p99']:.0f}, "
               f"TC4+ p99<={TARGETS['tc4_p99']:.0f} ms")
    out.append(f"Mchars/hour assumes {CHARS_PER_TOKEN} chars/token.")
    return "\n".join(out)


def charts(rows: list[dict], outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable - skipping charts, results.json is written)")
        return
    isls = sorted({r["isl"] for r in rows})

    # Chart 1: TTFC vs concurrency, P50 and P99, one series pair per ISL.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for isl in isls:
        pts = sorted((r for r in rows if r["isl"] == isl), key=lambda r: r["concurrency"])
        xs = [p["concurrency"] for p in pts]
        ax.plot(xs, [p["fc"]["p50"] for p in pts], marker="o", label=f"ISL {isl} P50")
        ax.plot(xs, [p["fc"]["p99"] for p in pts], marker="^", linestyle="--",
                label=f"ISL {isl} P99")
    ax.axhline(TARGETS["fc_p50"], color="grey", lw=1, ls=":")
    ax.axhline(TARGETS["fc_p99"], color="red", lw=1, ls=":")
    ax.annotate(f"P50 target {TARGETS['fc_p50']:.0f} ms", (0.01, TARGETS["fc_p50"]),
                xycoords=("axes fraction", "data"), fontsize=8, va="bottom")
    ax.annotate(f"P99 target {TARGETS['fc_p99']:.0f} ms", (0.01, TARGETS["fc_p99"]),
                xycoords=("axes fraction", "data"), fontsize=8, va="bottom", color="red")
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel("time to first chunk (ms)")
    ax.set_title("TTFC vs concurrency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart1_ttfc_vs_concurrency.png"), dpi=130)

    # Chart 2: input throughput.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for isl in isls:
        pts = sorted((r for r in rows if r["isl"] == isl), key=lambda r: r["concurrency"])
        ax.plot([p["concurrency"] for p in pts],
                [p["mchars_per_hour"] for p in pts], marker="o", label=f"ISL {isl}")
    ax.set_xlabel("concurrency (in-flight requests)")
    ax.set_ylabel("input throughput (Mchars/hour)")
    ax.set_title(f"Input throughput vs concurrency  ({CHARS_PER_TOKEN} chars/token)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart2_input_throughput.png"), dpi=130)
    print(f"  charts -> {outdir}/chart1_ttfc_vs_concurrency.png, chart2_input_throughput.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost:8010")
    ap.add_argument("--concurrency", default="1,50,200,400,700",
                    help="comma-separated in-flight request counts")
    ap.add_argument("--isl", default="128,512,1024",
                    help="comma-separated input lengths in tokens")
    ap.add_argument("--requests-per-point", type=int, default=0,
                    help="requests per point (default: 4x concurrency, min 40)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="tts_bench_results")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sweep and exit without contacting the server")
    args = ap.parse_args()

    concs = [int(c) for c in args.concurrency.split(",") if c]
    isls = [int(i) for i in args.isl.split(",") if i]
    os.makedirs(args.out, exist_ok=True)

    if args.dry_run:
        print(f"sweep: {len(concs) * len(isls)} points against {args.host}")
        for isl in isls:
            for c in concs:
                n = args.requests_per_point or max(40, 4 * c)
                print(f"  ISL {isl:>5}  conc {c:>4}  {n:>5} requests")
        return 0

    rows = []
    for isl in isls:
        for c in concs:
            n = args.requests_per_point or max(40, 4 * c)
            print(f"  ISL {isl} conc {c} ({n} requests)...", flush=True)
            row = run_point(args.host, c, isl, n, args.timeout)
            if row["failed"]:
                print(f"    {row['failed']} failed: {row['errors']}")
            rows.append(row)

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump({"targets": TARGETS, "chars_per_token": CHARS_PER_TOKEN,
                   "rows": rows}, fh, indent=1)
    table = tables(rows)
    with open(os.path.join(args.out, "results.txt"), "w") as fh:
        fh.write(table + "\n")
    print()
    print(table)
    charts(rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
