# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pure metric aggregation and presentation for tts_load_harness.

No requests or load generation occur here. JSON field names remain compatible
with earlier harness results. Charts import matplotlib only when requested.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000

CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "30"))

CHUNK_AUDIO_S = CHUNK_TOKENS * 960 / SAMPLE_RATE

CHUNK_AUDIO_MS = CHUNK_AUDIO_S * 1000.0

FC_P50_MS, FC_P95_MS, SC_DEADLINE_MS = 150.0, 350.0, 270.0

TC_DEADLINE_MS = CHUNK_AUDIO_MS - 30.0

TARGETS = {
    # key: (label, axis name, per-request field, p50 target ms, p95 target ms)
    # The second threshold is the P95 acceptance target (FC p95 <= 350 ms), not P90.
    "ttfb": ("TTFB", "TTFB — time to first audio chunk (ms)", "ttfb_s", 150.0, 350.0),
    "ttfs": ("TTFS", "TTFS — first->second chunk gap (ms)", "sc_s", 270.0, 270.0),
    "ttft": (
        "TTFT",
        "TTFT — steady-state inter-chunk gap (ms)",
        # "tc_gaps" is the SERIALIZED name (_serializable renames gaps -> tc_gaps),
        # and aggregate() always reads serialized records. _metric_samples also
        # accepts the in-memory "gaps" so either shape works.
        "tc_gaps",
        570.0,
        570.0,
    ),
}


def pct(vals, p: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * p / 100.0
    lo = int(k // 1)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _metric_samples(ok_recs: list[dict], key: str) -> list[float]:
    """Milliseconds for one metric. ``tc_gaps`` is a LIST per request -- every gap is a
    sample, matching how the TC deadline was always evaluated (per gap, not per
    request), so one long request cannot hide many late chunks."""
    field_name = TARGETS[key][2]
    aliases = {"tc_gaps": ("tc_gaps", "gaps"), "gaps": ("gaps", "tc_gaps")}
    names = aliases.get(field_name, (field_name,))
    out = []
    for r in ok_recs:
        v = next((r[n] for n in names if r.get(n) is not None), None)
        if v is None:
            continue
        if isinstance(v, list):
            out.extend(x * 1000.0 for x in v)
        else:
            out.append(v * 1000.0)
    return out


def _arrival_cohort(recs: list[dict], w0: float | None, w1: float | None) -> list[dict]:
    """Requests submitted in the window; missing bounds retain legacy records."""
    return [r for r in recs if w0 is None or w1 is None or w0 <= r["t_send"] < w1]


def _overlaps(recs: list[dict], w0: float, w1: float):
    """Yield request intervals clipped to the window, including unfinished requests."""
    for r in recs:
        start = max(r["t_send"], w0)
        end = min(r.get("t_end") or w1, w1)
        if end > start:
            yield start, end


def _occupancy(recs: list[dict], w0: float, w1: float) -> float:
    """Average client occupancy, clipping all request lifetimes to the window."""
    if not (w0 and w1 and w1 > w0):
        return float("nan")
    events = []
    for start, end in _overlaps(recs, w0, w1):
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    area, cur, prev = 0.0, 0, w0
    for t, delta in events:
        area += cur * (t - prev)
        prev = t
        cur += delta
    area += cur * (w1 - prev)
    return area / (w1 - w0)


def aggregate(doc: dict) -> tuple[list[dict], int]:
    """Aggregate client occupancy and arrival-cohort latency/throughput per level."""
    chars = doc["meta"]["chars_per_request"]
    by_conc: dict[int, list[dict]] = {}
    for r in doc["records"]:
        by_conc.setdefault(r["conc"], []).append(r)
    level_meta = {lvl["conc"]: lvl for lvl in doc.get("levels", [])}

    rows = []
    for conc in sorted(by_conc):
        recs = by_conc[conc]
        meta = level_meta.get(conc, {})
        w0, w1 = meta.get("window_start"), meta.get("window_end")
        # Occupancy uses all records; throughput and latency use window arrivals.
        in_win = _arrival_cohort(recs, w0, w1)
        ok = [r for r in in_win if r.get("ok")]
        window = (meta.get("window_end", 0) - meta.get("window_start", 0)) or float(
            "nan"
        )
        rps = len(ok) / window if window and window == window else float("nan")
        full = [r for r in ok if not r.get("capped")]
        gen = [r["gen_s"] for r in ok if r.get("gen_s")]
        audio = [r["audio_s"] for r in ok if r.get("audio_s")]
        rtf = [
            r["gen_s"] / r["audio_s"] for r in ok if r.get("audio_s") and r.get("gen_s")
        ]
        nchunks = [r["nchunks"] for r in ok if r.get("nchunks")]
        errors: dict[str, int] = {}
        for r in in_win:
            if not r.get("ok"):
                key = r.get("error") or "?"
                errors[key] = errors.get(key, 0) + 1

        row = {
            "conc": conc,
            "n_ok": len(ok),
            "n_err": len(in_win) - len(ok),
            "rps": rps,
            "gen_p50": pct(gen, 50) if gen else float("nan"),
            "audio_p50": pct(audio, 50) if audio else float("nan"),
            "rtf_p50": pct(rtf, 50) if rtf else float("nan"),
            # Characters may only be credited for responses that delivered the WHOLE
            # text. A client-capped response stopped early; crediting its full input
            # inflates characters/hour by whatever it never spoke.
            "mchar_h": (
                chars * (len(full) / window) * 3600.0 / 1e6
                if window and window == window
                else float("nan")
            ),
            "n_capped": sum(1 for r in ok if r.get("capped")),
            "C": _occupancy(recs, w0, w1),
            "chunks": (sum(nchunks) / len(nchunks)) if nchunks else float("nan"),
            "errors": errors,
        }
        for key in TARGETS:
            samples = _metric_samples(ok, key)
            for percentile in (50, 90, 95, 99):
                row[f"{key}_p{percentile}"] = pct(samples, percentile)
            row[f"{key}_n"] = len(samples)
        rows.append(row)
    return rows, chars


SUMMARY_COLS = [
    ("conc", "users", 5),
    ("rps", "RPS", 7),
    ("mchar_h", "Mchar/h", 8),
    ("ttfb_p50", "TTFB p50", 9),
    ("ttfb_p90", "TTFB p90", 9),
    ("ttfs_p50", "TTFS p50", 9),
    ("ttfs_p90", "TTFS p90", 9),
    ("ttft_p50", "TTFT p50", 9),
    ("ttft_p90", "TTFT p90", 9),
    ("rtf_p50", "RTF", 6),
    ("n_ok", "ok", 6),
    ("n_err", "err", 5),
]


def summary_table(rows: list[dict], chars: int, closed_loop) -> str:
    """Display arrival-cohort throughput; character credit excludes capped requests."""
    out = [
        f"\nchars/request = {chars}   closed-loop = {closed_loop}   throughput = arrival-cohort estimate"
    ]
    out.append("".join(h.rjust(w) for _, h, w in SUMMARY_COLS))
    out.append("-" * sum(w for _, _, w in SUMMARY_COLS))
    for r in rows:
        line = ""
        for key, _, w in SUMMARY_COLS:
            v = r[key]
            line += (f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(w)
        out.append(line)
    return "\n".join(out)


def box_table(rows: list[dict]) -> str:
    """Compact percentile/target comparison; passing P95 is not zero violations."""
    out = ["\nLatency targets (ms): FC P50/P95=150/350, SC=270, TC=570"]
    for row in rows:
        out.append(f"users={row['conc']}  C={row['C']:.1f}  chunks={row['chunks']:.1f}")
        for key, label in (("ttfb", "FC"), ("ttfs", "SC"), ("ttft", "TC")):
            cells = []
            for percentile, target in ((50, TARGETS[key][3]), (95, TARGETS[key][4])):
                value = row[f"{key}_p{percentile}"]
                status = (
                    "n/a" if value != value else "PASS" if value <= target else "MISS"
                )
                cells.append(f"P{percentile} {value:.0f}/{target:.0f} {status}")
            out.append(f"  {label}: " + ", ".join(cells))
        if row["n_ok"] < 20:
            out.append(
                f"  Only {row['n_ok']} successful requests; tail percentiles are uncertain."
            )
    return "\n".join(out) + "\n"


def print_report(doc: dict, args: argparse.Namespace) -> list[dict]:
    rows, chars = aggregate(doc)
    if not rows:
        sys.exit("no records")

    if getattr(args, "format", "table") == "csv":
        print(",".join(k for k, _, _ in SUMMARY_COLS))
        for r in rows:
            print(
                ",".join(
                    f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])
                    for k, _, _ in SUMMARY_COLS
                )
            )
    else:
        print(summary_table(rows, chars, doc["meta"].get("closed_loop")))
        print()
        print(box_table(rows), end="")
        bad = [r for r in rows if r["n_err"]]
        if bad:
            print("\nERRORS (excluded from RPS):")
            for r in bad:
                for err, n in sorted(r["errors"].items(), key=lambda x: -x[1]):
                    print(f"  conc={r['conc']:<5} {n:>5} x {err}")

    if getattr(args, "chart", None):
        _render_charts(rows, chars, args)
    return rows


def _render_charts(rows: list[dict], chars: int, args: argparse.Namespace) -> None:
    """Render individual latency curves and an optional combined chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(TARGETS) if args.target == "all" else [args.target]
    available = [key for key in keys if any(row[f"{key}_n"] for row in rows)]
    base, ext = os.path.splitext(args.chart)
    ext = ext or ".png"
    plots = [
        (key, [key], args.chart if len(keys) == 1 else f"{base}_{key}{ext}")
        for key in available
    ]
    if args.target == "all":
        plots.append(("combined", available, f"{base}_combined{ext}"))
    for key in set(keys) - set(available):
        logger.warning("skipping %s: no samples", key)

    for name, metrics, path in plots:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        for key in metrics:
            label, axis, _, t50, t95 = TARGETS[key]
            for percentile, style in ((50, "-o"), (95, "--s")):
                ax.plot(
                    [row[f"{key}_p{percentile}"] for row in rows],
                    [row["mchar_h"] for row in rows],
                    style,
                    label=f"{label} P{percentile}",
                )
            for row in rows:
                point = (row[f"{key}_p50"], row["mchar_h"])
                ax.annotate(
                    str(row["conc"]),
                    point,
                    xytext=(6, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
                if args.mark_errors and row["n_err"]:
                    ax.scatter(*point, s=140, facecolors="none", edgecolors="red")
            targets = (
                [(50, t50)]
                if name == "combined" or t50 == t95
                else [(50, t50), (95, t95)]
            )
            for percentile, target in targets:
                ax.axvline(
                    target,
                    ls=":",
                    lw=1,
                    label=f"{label} P{percentile} target {target:.0f} ms",
                )
        axis = "latency (ms) — FC / SC / TC" if name == "combined" else TARGETS[name][1]
        title = args.title
        if not args.no_subtitle:
            subtitle = args.subtitle or f"{chars} chars/request"
            title += f"\n{subtitle}; point label = configured concurrency"
        ax.set(
            xlabel=axis,
            ylabel="Arrival-cohort throughput (million characters / hour)",
            title=title,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("wrote %s", path)


def _dump_user_rows(ok: list[dict], users: int) -> list[dict]:
    rows = []
    for uid in range(users):
        mine = [r for r in ok if r["uid"] == uid]
        if not mine:
            continue
        gaps = [g for r in mine for g in r["gaps_ms"]]
        bubbles = [g for r in mine for g in r["bubbles"]]
        rows.append(
            {
                "u": uid,
                "streams": len(mine),
                "chunks": sum(r["nchunks"] for r in mine),
                "admit_p50": pct([r["admit_ms"] for r in mine], 50),
                "fc_p50": pct([r["fc_ms"] for r in mine], 50),
                "fc_p90": pct([r["fc_ms"] for r in mine], 90),
                "sc_p50": pct([r["sc_ms"] for r in mine], 50),
                "tc_p50": pct(gaps, 50),
                "tc_p90": pct(gaps, 90),
                "tc_max": max(gaps) if gaps else float("nan"),
                "bubbles": len(bubbles),
                "bub_pct": 100.0 * len(bubbles) / len(gaps) if gaps else 0.0,
                "stall_ms": sum(r["stall_ms"] for r in mine),
                "rtf_p50": pct([r["rtf"] for r in mine], 50),
            }
        )
    return rows


def _write_dump_tsvs(
    ok: list[dict],
    rows: list[dict],
    outdir: str,
    users: int,
    t_start: float,
    allrec: list[dict] | None = None,
    w0: float | None = None,
    w1: float | None = None,
) -> str:
    header = [
        "u",
        "streams",
        "chunks",
        "admit_p50",
        "fc_p50",
        "fc_p90",
        "sc_p50",
        "tc_p50",
        "tc_p90",
        "tc_max",
        "bubbles",
        "bub_pct",
        "stall_ms",
        "rtf_p50",
    ]
    tsv = os.path.join(outdir, f"pipeline_dump_u0_u{users - 1}.tsv")
    with open(tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h])
                    for h in header
                )
                + "\n"
            )

    with open(os.path.join(outdir, "timings.tsv"), "w") as f:
        f.write("t_send\tt_end\tgen_s\twall_s\ttail_s\tnchunks\n")
        for r in ok:
            f.write(
                f"{r['t_send']:.4f}\t{r['t_end']:.4f}\t{r['audio_end_s']:.4f}\t"
                f"{r['wall_s']:.4f}\t{r['tail_s']:.4f}\t{r['nchunks']}\n"
            )

    # Use the same population and window as C. Rows show pre-event occupancy.
    src = allrec if allrec is not None else ok
    lo = w0 if w0 is not None else t_start
    hi = w1 if w1 is not None else max((r["t_end"] or lo) for r in src)
    events = []
    for start, end in _overlaps(src, lo, hi):
        events.append((start, 1))
        events.append((end, -1))
    events.sort()
    with open(os.path.join(outdir, "occupancy.tsv"), "w") as f:
        f.write("t_rel_s\tin_flight\n")
        cur = 0
        for t, delta in events:
            f.write(f"{t - t_start:.3f}\t{cur}\n")
            cur += delta
    return tsv


def _pct_over(vals: list[float], deadline: float) -> float:
    return 100.0 * sum(1 for v in vals if v > deadline) / max(1, len(vals))


def _latency_row(label: str, vals: list[float], deadline: str) -> str:
    """One percentile row of the dump's latency table."""
    cells = "".join(f"{pct(vals, p):10.1f}" for p in (50, 90, 95, 99))
    return f"{label:<11}{cells}{max(vals):11.1f}   {deadline}"


def _dump_text(s: dict) -> str:
    fc, sc, tc, admit = s["fc"], s["sc"], s["tc"], s["admit"]
    stall_s = sum(g - CHUNK_AUDIO_MS for g in s["bubbles"]) / 1000.0
    latency = "\n".join(
        [
            _latency_row("admission", admit, "--"),
            _latency_row("FC", fc, f"{FC_P50_MS:.0f}/{FC_P95_MS:.0f}"),
            _latency_row("SC", sc, f"{SC_DEADLINE_MS:.0f}"),
            _latency_row("TC", tc, f"{TC_DEADLINE_MS:.0f}"),
        ]
    )
    header = f"pipeline dump: {s['users']} users, {s['elapsed_s']:.0f}s scored"
    fc_p50 = "MISS" if pct(fc, 50) > FC_P50_MS else "ok"
    fc_p95 = "MISS" if pct(fc, 95) > FC_P95_MS else "ok"
    return f"""
================ {header} ================
streams scored {s["scored"]}   chunks {s["chunks"]}   \
chunks/s {s["chunks"] / s["elapsed_s"]:.1f}   chars/request {s["chars"]}
requests recorded {s["recorded"]}   scored {s["scored"]}   failed {s["failed"]}   \
short(1-2ch) {s["short"]}   empty {s["empty"]}
arrival window {s["elapsed_s"]:.1f}s   C window {s["window_s"]:.1f}s   \
(drain excluded; with drain it would be {s["drain_window_s"]:.1f}s)
C (time-avg in-flight)  {s["C"]:.1f}          RPS {s["rps"]:.2f}
  Little's law  C = RPS x W
     W = gen_s  (to last audio chunk) {s["mean_gen_s"]:6.2f}s -> C \
{s["rps"] * s["mean_gen_s"]:7.1f}
     W = wall_s (to socket close)     {s["mean_wall_s"]:6.2f}s -> C \
{s["rps"] * s["mean_wall_s"]:7.1f}
     post-audio tail                  {s["mean_tail_s"]:6.2f}s

                  p50       p90       p95       p99        max      deadline
{latency}

LONG GAPS (inter-chunk gap > {CHUNK_AUDIO_MS:.0f} ms; not playback stalls)
  count {len(s["bubbles"])} / {len(tc)} gaps = \
{100.0 * len(s["bubbles"]) / max(1, len(tc)):.1f}%
  summed gap excess {stall_s:.1f} s across {s["scored"]} streams
  worst gap {max(tc):.0f} ms = {max(tc) / CHUNK_AUDIO_MS:.2f}x real time
  users with >=1 bubble: {s["users_with_bubbles"]} / {s["n_users"]}
  RTF p50 {pct(s["rtf"], 50):.3f}  p90 {pct(s["rtf"], 90):.3f}  (must stay < 1.0)

DEADLINE VIOLATIONS
  FC p50 {fc_p50}   FC p95 {fc_p95}
  SC over {SC_DEADLINE_MS:.0f}ms: {_pct_over(sc, SC_DEADLINE_MS):.1f}%
  TC over {TC_DEADLINE_MS:.0f}ms: {_pct_over(tc, TC_DEADLINE_MS):.1f}%

wrote {s["tsv"]}
      {os.path.join(s["outdir"], "occupancy.tsv")}
"""


LADDER_COLS = [
    ("users", 6),
    ("scored", 8),
    ("RPS", 8),
    ("C", 8),
    ("FC p50", 9),
    ("FC p90", 9),
    ("SC p50", 9),
    ("TC p50", 9),
    ("TC p90", 9),
    ("bubble%", 9),
    ("RTF p50", 9),
    ("failed", 8),
]


def ladder_table(results: list[dict]) -> str:
    out = [
        "",
        "".join(h.rjust(w) for h, w in LADDER_COLS),
        "-" * sum(w for _, w in LADDER_COLS),
    ]
    for s in results:
        tc = s["tc"]
        cells = [
            str(s["users"]),
            str(s["scored"]),
            f"{s['rps']:.2f}",
            f"{s['C']:.1f}",
            f"{pct(s['fc'], 50):.0f}",
            f"{pct(s['fc'], 90):.0f}",
            f"{pct(s['sc'], 50):.0f}",
            f"{pct(tc, 50):.0f}",
            f"{pct(tc, 90):.0f}",
            f"{100.0 * len(s['bubbles']) / max(1, len(tc)):.1f}",
            f"{pct(s['rtf'], 50):.3f}",
            str(s["failed"]),
        ]
        out.append("".join(c.rjust(w) for c, (_, w) in zip(cells, LADDER_COLS)))
    return "\n".join(out)
