# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Overlay several staged runs (e.g. direct vs ngrok vs console) on one set of charts.

    python compare_runs.py --run direct=staged-direct/staged.json \\
        --run ngrok=staged-ngrok/staged.json --out cmp

writes, next to the --out prefix:

    <out>_1_throughput_vs_fc.png     ladder: requests/s vs first-chunk P50/P95, per run
    <out>_2_fc_vs_concurrency.png    ladder: first-chunk P50/P95 vs concurrency, per run
    <out>_3_fc_vs_prompt_size.png    probe: mean first-chunk latency per prompt size, per run

and prints one table with FC P50/P95 per concurrency level for every run.
P50 is a solid line, P95 dashed; each run gets its own color.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_staged_harness as staged  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8c1d04", "#7a4fd6"]


def load(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    if not path:
        label, path = os.path.basename(os.path.dirname(os.path.abspath(spec))), spec
    with open(path) as f:
        return label, json.load(f)


def probe_means(doc: dict) -> list:
    groups = collections.defaultdict(list)
    for r in doc["records"]:
        if r["stage"] == "probe" and r.get("ttfb_s"):
            groups[r["isl"]].append(r["ttfb_s"] * 1e3)
    return sorted((k, sum(v) / len(v)) for k, v in groups.items())


def table(runs: list) -> str:
    levels = sorted({c for _, d in runs for c, _, _ in staged._ladder_points(d)})
    head = ["conc"] + [f"{lab} {k}" for lab, _ in runs for k in ("req/s", "FC p50", "FC p95")]
    rows = [head]
    for c in levels:
        row = [str(c)]
        for _, d in runs:
            pts = {cc: s for cc, s, _ in staged._ladder_points(d)}
            s = pts.get(c)
            row += (
                [f"{s['req_per_s']:.2f}", f"{s['fc_p50_ms']:.0f}", f"{s['fc_p95_ms']:.0f}"]
                if s
                else ["-", "-", "-"]
            )
        rows.append(row)
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    return "\n".join("  ".join(v.rjust(w[i]) for i, v in enumerate(r)) for r in rows)


def charts(runs: list, prefix: str) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    paths = []

    def save(fig, name):
        fig.tight_layout()
        paths.append(f"{prefix}_{name}.png")
        fig.savefig(paths[-1], dpi=110)
        plt.close(fig)

    # 1. throughput vs first chunk
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for i, (label, doc) in enumerate(runs):
        pts = staged._ladder_points(doc)
        color = COLORS[i % len(COLORS)]
        for key, ls, mk in (("fc_p50_ms", "-", "o"), ("fc_p95_ms", "--", "o")):
            ax.plot(
                [s[key] for _, s, _ in pts],
                [s["req_per_s"] for _, s, _ in pts],
                ls=ls, marker=mk, color=color, lw=2,
                mfc=color if ls == "-" else "white", mec=color,
                label=f"{label} {key[3:6].upper()}",
            )
    staged._style(ax, "Throughput vs first-chunk latency",
                  "first-chunk latency (ms, log)", "throughput (requests/s, log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=9)
    save(fig, "1_throughput_vs_fc")

    # 2. first chunk vs concurrency
    fig, ax = plt.subplots(figsize=(10, 6.2))
    concs = sorted({c for _, d in runs for c, _, _ in staged._ladder_points(d)})
    for i, (label, doc) in enumerate(runs):
        pts = staged._ladder_points(doc)
        color = COLORS[i % len(COLORS)]
        for key, ls in (("fc_p50_ms", "-"), ("fc_p95_ms", "--")):
            ax.plot(
                [c for c, _, _ in pts], [s[key] for _, s, _ in pts],
                ls=ls, marker="o", color=color, lw=2,
                mfc=color if ls == "-" else "white", mec=color,
                label=f"{label} {key[3:6].upper()}",
            )
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(concs))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(c) for c in concs])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    staged._style(ax, "First-chunk latency vs concurrency",
                  "concurrent clients (closed loop)", "first-chunk latency (ms, log)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "2_fc_vs_concurrency")

    # 3. probe: first chunk vs prompt size
    fig, ax = plt.subplots(figsize=(10, 6.2))
    any_probe = False
    for i, (label, doc) in enumerate(runs):
        pts = probe_means(doc)
        if pts:
            any_probe = True
            ax.plot(*zip(*pts), marker="o", color=COLORS[i % len(COLORS)], lw=2, label=label)
    if any_probe:
        staged._style(ax, "First-chunk latency vs prompt size (one request at a time)",
                      "prompt tokens (text + voice)", "first-chunk latency (ms)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        save(fig, "3_fc_vs_prompt_size")
    else:
        plt.close(fig)
    return paths


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="label=path/to/staged.json; repeat once per run")
    ap.add_argument("--out", default="compare", help="chart file prefix")
    args = ap.parse_args(argv)
    runs = [load(s) for s in args.run]
    print(table(runs))
    for p in charts(runs, args.out):
        print("wrote", p)


if __name__ == "__main__":
    main()
