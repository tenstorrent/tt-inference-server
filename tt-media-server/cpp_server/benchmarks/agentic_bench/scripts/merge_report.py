#!/usr/bin/env python3
"""
merge_report.py - aggregate all per-chunk GuideLLM benchmarks.json files under
a run directory and produce a single self-contained `report.html`.

Idempotent and safe to re-run while the soak is still in progress.

Usage:
    merge_report.py <run_dir>

The HTML embeds:
  - config snapshot (from config.txt)
  - aggregate TTFT / ITL / TPOT / E2E / throughput percentiles, overall and
    bucketed by wall-clock hour and by prompt-tokens decile
  - per-chunk summary table
  - timeline (SVG rendered from 60-second buckets)
  - per-request table, paginated client-side
  - raw merged JSON embedded as <script type="application/json" id="data">
"""

from __future__ import annotations

import html
import json
import math
import statistics
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from guidellm.benchmark import GenerativeBenchmarksReport
except ImportError as exc:
    print(
        "error: `guidellm` is not installed. Install with:\n"
        "   pip install <path>/cpp_server/guidellm",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chunks(run_dir: Path) -> list[tuple[Path, GenerativeBenchmarksReport]]:
    """Load every chunk's benchmarks.json from the run directory."""
    chunks: list[tuple[Path, GenerativeBenchmarksReport]] = []
    for chunk_dir in sorted(run_dir.glob("chunk_*")):
        bj = chunk_dir / "benchmarks.json"
        if not bj.is_file():
            continue
        try:
            rep = GenerativeBenchmarksReport.load_file(bj)
        except Exception as e:  # noqa: BLE001
            print(f"warn: failed to load {bj}: {e}", file=sys.stderr)
            continue
        chunks.append((chunk_dir, rep))
    return chunks


# ---------------------------------------------------------------------------
# Row extraction (one row per request, from every chunk)
# ---------------------------------------------------------------------------

def extract_rows(chunks: list[tuple[Path, GenerativeBenchmarksReport]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_dir, rep in chunks:
        chunk_idx = int(chunk_dir.name.rsplit("_", 1)[-1])
        for bench in rep.benchmarks:
            for status in ("successful", "incomplete", "errored"):
                for req in getattr(bench.requests, status) or []:
                    rows.append(_row_from_request(req, status, chunk_idx))
    return rows


def _row_from_request(req: Any, status: str, chunk_idx: int) -> dict[str, Any]:
    timings = req.info.timings
    return {
        "chunk": chunk_idx,
        "status": status,
        "request_id": req.request_id,
        "response_id": req.response_id,
        "start": timings.request_start,
        "first_token": timings.first_token_iteration,
        "last_token": timings.last_token_iteration,
        "end": timings.request_end,
        "prompt_tokens": req.prompt_tokens,
        "output_tokens": req.output_tokens,
        "ttft_ms": req.time_to_first_token_ms,
        "itl_ms": req.inter_token_latency_ms,
        "tpot_ms": req.time_per_output_token_ms,
        "latency_s": req.request_latency,
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

PERCENTILES = [50, 90, 95, 99]


def percentiles(values: list[float], pcts: list[int] = PERCENTILES) -> dict[str, float | None]:
    """Compute percentiles. Returns None for empty input."""
    out: dict[str, float | None] = {f"p{p}": None for p in pcts}
    out["mean"] = None
    out["count"] = 0
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return out
    clean.sort()
    out["count"] = len(clean)
    out["mean"] = sum(clean) / len(clean)
    for p in pcts:
        # Nearest-rank percentile (C=1 from NIST); stable, no interp artifacts.
        k = max(0, min(len(clean) - 1, math.ceil(p / 100 * len(clean)) - 1))
        out[f"p{p}"] = clean[k]
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "successful"]
    if not ok:
        return {"requests": len(rows), "successful": 0}
    start = min(r["start"] for r in ok if r["start"] is not None)
    end = max(r["end"] for r in ok if r["end"] is not None)
    duration = max(end - start, 1e-9)
    total_output_tokens = sum((r["output_tokens"] or 0) for r in ok)
    return {
        "requests": len(rows),
        "successful": len(ok),
        "incomplete": sum(1 for r in rows if r["status"] == "incomplete"),
        "errored":    sum(1 for r in rows if r["status"] == "errored"),
        "start_ts": start,
        "end_ts": end,
        "duration_s": duration,
        "rps": len(ok) / duration,
        "output_tps": total_output_tokens / duration,
        "ttft_ms":  percentiles([r["ttft_ms"]  for r in ok]),
        "itl_ms":   percentiles([r["itl_ms"]   for r in ok]),
        "tpot_ms":  percentiles([r["tpot_ms"]  for r in ok]),
        "latency_s":percentiles([r["latency_s"] for r in ok]),
        "prompt_tokens":  percentiles([r["prompt_tokens"] or 0 for r in ok]),
        "output_tokens":  percentiles([r["output_tokens"] or 0 for r in ok]),
    }


def bucket_by_prompt_decile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in rows if r["status"] == "successful" and r["prompt_tokens"]]
    if len(ok) < 10:
        return []
    sorted_pt = sorted(r["prompt_tokens"] for r in ok)
    # decile boundaries (9 edges -> 10 buckets)
    edges = [sorted_pt[min(len(sorted_pt) - 1, int(len(sorted_pt) * q / 10))] for q in range(1, 10)]
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for r in ok:
        idx = bisect_right(edges, r["prompt_tokens"])
        buckets[min(idx, 9)].append(r)
    out = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        pts = [r["prompt_tokens"] for r in b]
        out.append({
            "decile": i + 1,
            "count": len(b),
            "prompt_tokens_min": min(pts),
            "prompt_tokens_max": max(pts),
            "ttft_ms":  percentiles([r["ttft_ms"]  for r in b]),
            "itl_ms":   percentiles([r["itl_ms"]   for r in b]),
            "latency_s":percentiles([r["latency_s"] for r in b]),
        })
    return out


def bucket_by_hour(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in rows if r["status"] == "successful" and r["start"] is not None]
    if not ok:
        return []
    start0 = min(r["start"] for r in ok)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for r in ok:
        hour = int((r["start"] - start0) // 3600)
        buckets.setdefault(hour, []).append(r)
    out = []
    for hour in sorted(buckets):
        b = buckets[hour]
        out.append({
            "hour": hour,
            "count": len(b),
            "ttft_ms":  percentiles([r["ttft_ms"]  for r in b]),
            "itl_ms":   percentiles([r["itl_ms"]   for r in b]),
            "latency_s":percentiles([r["latency_s"] for r in b]),
        })
    return out


def timeline_buckets(rows: list[dict[str, Any]], bucket_s: int = 60) -> list[dict[str, Any]]:
    ok = [r for r in rows if r["status"] == "successful" and r["start"] is not None]
    if not ok:
        return []
    start0 = min(r["start"] for r in ok)
    all_reqs = [r for r in rows if r["start"] is not None]
    by_bucket: dict[int, dict[str, list[float]]] = {}
    for r in ok:
        k = int((r["start"] - start0) // bucket_s)
        bd = by_bucket.setdefault(k, {"ttft": [], "itl": [], "latency": []})
        if r["ttft_ms"]   is not None: bd["ttft"].append(r["ttft_ms"])
        if r["itl_ms"]    is not None: bd["itl"].append(r["itl_ms"])
        if r["latency_s"] is not None: bd["latency"].append(r["latency_s"])
    # error counts (same bucket-key basis)
    err_by_bucket: dict[int, int] = {}
    for r in all_reqs:
        if r["status"] != "successful":
            k = int((r["start"] - start0) // bucket_s)
            err_by_bucket[k] = err_by_bucket.get(k, 0) + 1
    out = []
    for k in sorted(set(by_bucket) | set(err_by_bucket)):
        bd = by_bucket.get(k, {"ttft": [], "itl": [], "latency": []})
        out.append({
            "bucket": k,
            "ts": start0 + k * bucket_s,
            "rps": len(bd["ttft"]) / bucket_s,
            "ttft_p50": percentiles(bd["ttft"])["p50"],
            "ttft_p95": percentiles(bd["ttft"])["p95"],
            "ttft_p99": percentiles(bd["ttft"])["p99"],
            "itl_p50": percentiles(bd["itl"])["p50"],
            "latency_p95": percentiles(bd["latency"])["p95"],
            "errors": err_by_bucket.get(k, 0),
        })
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def fmt_float(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_int(v: Any) -> str:
    return "-" if v is None else f"{int(v)}"


def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def render_perc_table(title: str, rows: list[tuple[str, dict[str, float | None]]]) -> str:
    head = ''.join(f"<th>{h}</th>" for h in ["metric", "count", "mean", "p50", "p90", "p95", "p99"])
    body_lines = []
    for name, p in rows:
        body_lines.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{fmt_int(p.get('count'))}</td>"
            f"<td>{fmt_float(p.get('mean'))}</td>"
            f"<td>{fmt_float(p.get('p50'))}</td>"
            f"<td>{fmt_float(p.get('p90'))}</td>"
            f"<td>{fmt_float(p.get('p95'))}</td>"
            f"<td>{fmt_float(p.get('p99'))}</td>"
            "</tr>"
        )
    return (
        f"<h3>{html.escape(title)}</h3>"
        "<table><thead><tr>" + head + "</tr></thead>"
        "<tbody>" + ''.join(body_lines) + "</tbody></table>"
    )


def render_timeline_svg(buckets: list[dict[str, Any]]) -> str:
    if not buckets:
        return "<p><em>no timeline data</em></p>"

    def _sparkline(key: str, title: str, unit: str, color_hint: str) -> str:
        values = [(b["ts"], b.get(key)) for b in buckets]
        clean = [(t, v) for t, v in values if v is not None]
        if not clean:
            return f"<div><strong>{html.escape(title)}</strong>: no data</div>"
        xs = [t for t, _ in clean]
        ys = [v for _, v in clean]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        y_range = max(y_max - y_min, 1e-9)
        x_range = max(x_max - x_min, 1e-9)
        W, H = 900, 90
        pad_l, pad_r, pad_t, pad_b = 70, 10, 10, 18
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b
        pts = []
        for t, v in clean:
            px = pad_l + (t - x_min) / x_range * plot_w
            py = pad_t + (1 - (v - y_min) / y_range) * plot_h
            pts.append(f"{px:.1f},{py:.1f}")
        path = "M " + " L ".join(pts)
        y_axis_min = f'<text x="{pad_l - 5}" y="{pad_t + plot_h:.1f}" class="ylabel" text-anchor="end">{y_min:.0f}</text>'
        y_axis_max = f'<text x="{pad_l - 5}" y="{pad_t + 9:.1f}" class="ylabel" text-anchor="end">{y_max:.0f}</text>'
        x_axis_start = f'<text x="{pad_l}" y="{H - 4}" class="xlabel">{fmt_ts(x_min)}</text>'
        x_axis_end = f'<text x="{W - pad_r}" y="{H - 4}" class="xlabel" text-anchor="end">{fmt_ts(x_max)}</text>'
        return (
            f'<div class="chart"><div class="chart-title">{html.escape(title)} ({html.escape(unit)}) '
            f'&mdash; min {y_min:.2f}, max {y_max:.2f}, mean {sum(ys)/len(ys):.2f}</div>'
            f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" width="100%" height="{H}">'
            f'<path d="{path}" fill="none" stroke="{color_hint}" stroke-width="1.5" />'
            f'{y_axis_min}{y_axis_max}{x_axis_start}{x_axis_end}'
            "</svg></div>"
        )

    return (
        _sparkline("rps",          "requests/sec (60s buckets)",       "req/s",  "currentColor")
        + _sparkline("ttft_p50",   "TTFT p50",                         "ms",     "currentColor")
        + _sparkline("ttft_p95",   "TTFT p95",                         "ms",     "currentColor")
        + _sparkline("ttft_p99",   "TTFT p99",                         "ms",     "currentColor")
        + _sparkline("itl_p50",    "ITL p50",                          "ms",     "currentColor")
        + _sparkline("latency_p95","request latency p95",              "s",      "currentColor")
        + _sparkline("errors",     "errors (count per 60s bucket)",    "count",  "currentColor")
    )


def render_chunk_table(chunks: list[tuple[Path, GenerativeBenchmarksReport]]) -> str:
    rows = []
    for cd, rep in chunks:
        for bench in rep.benchmarks:
            reqs = bench.requests
            s = len(reqs.successful or [])
            i = len(reqs.incomplete or [])
            e = len(reqs.errored or [])
            rows.append(
                "<tr>"
                f"<td>{html.escape(cd.name)}</td>"
                f"<td>{fmt_ts(bench.start_time)}</td>"
                f"<td>{fmt_float(bench.duration, 1)}</td>"
                f"<td>{s}</td>"
                f"<td>{i}</td>"
                f"<td>{e}</td>"
                "</tr>"
            )
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in
        ["chunk", "start (UTC)", "duration (s)", "ok", "incomplete", "errored"]) + "</tr>"
    return "<table><thead>" + head + "</thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_html(
    run_dir: Path,
    config_text: str,
    chunks: list[tuple[Path, GenerativeBenchmarksReport]],
    rows: list[dict[str, Any]],
) -> str:
    agg = aggregate(rows)
    hourly = bucket_by_hour(rows)
    decile = bucket_by_prompt_decile(rows)
    tl = timeline_buckets(rows)

    raw = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": config_text,
        "aggregate": agg,
        "hourly": hourly,
        "prompt_decile": decile,
        "timeline": tl,
        "requests": rows,
    }

    agg_table = render_perc_table("aggregate (all successful requests)", [
        ("TTFT (ms)",                 agg.get("ttft_ms", {}) if agg.get("successful") else {}),
        ("ITL (ms, excl. first tok)", agg.get("itl_ms", {}) if agg.get("successful") else {}),
        ("TPOT (ms)",                 agg.get("tpot_ms", {}) if agg.get("successful") else {}),
        ("latency (s)",               agg.get("latency_s", {}) if agg.get("successful") else {}),
        ("prompt tokens",             agg.get("prompt_tokens", {}) if agg.get("successful") else {}),
        ("output tokens",             agg.get("output_tokens", {}) if agg.get("successful") else {}),
    ])

    hourly_rows = []
    for h in hourly:
        hourly_rows.append(
            "<tr>"
            f"<td>h{h['hour']}</td><td>{h['count']}</td>"
            f"<td>{fmt_float(h['ttft_ms']['p50'])}</td>"
            f"<td>{fmt_float(h['ttft_ms']['p95'])}</td>"
            f"<td>{fmt_float(h['ttft_ms']['p99'])}</td>"
            f"<td>{fmt_float(h['itl_ms']['p95'])}</td>"
            f"<td>{fmt_float(h['latency_s']['p95'])}</td>"
            "</tr>"
        )
    hourly_table = (
        "<table><thead><tr>"
        + "".join(f"<th>{x}</th>" for x in
                  ["hour", "count", "ttft p50", "ttft p95", "ttft p99", "itl p95", "latency p95 (s)"])
        + "</tr></thead><tbody>" + "".join(hourly_rows) + "</tbody></table>"
    ) if hourly else "<p><em>fewer than one hour of data</em></p>"

    decile_rows = []
    for d in decile:
        decile_rows.append(
            "<tr>"
            f"<td>d{d['decile']}</td><td>{d['count']}</td>"
            f"<td>{d['prompt_tokens_min']}&ndash;{d['prompt_tokens_max']}</td>"
            f"<td>{fmt_float(d['ttft_ms']['p50'])}</td>"
            f"<td>{fmt_float(d['ttft_ms']['p95'])}</td>"
            f"<td>{fmt_float(d['itl_ms']['p95'])}</td>"
            f"<td>{fmt_float(d['latency_s']['p95'])}</td>"
            "</tr>"
        )
    decile_table = (
        "<table><thead><tr>"
        + "".join(f"<th>{x}</th>" for x in
                  ["decile", "count", "prompt tokens range",
                   "ttft p50", "ttft p95", "itl p95", "latency p95 (s)"])
        + "</tr></thead><tbody>" + "".join(decile_rows) + "</tbody></table>"
    ) if decile else "<p><em>too few successful requests for decile buckets</em></p>"

    raw_json = json.dumps(raw, default=_json_default)

    agg_summary = (
        f"{agg.get('successful', 0)} ok / {agg.get('errored', 0)} err / {agg.get('incomplete', 0)} incomp "
        f"over {fmt_float(agg.get('duration_s'), 1)}s "
        f"({fmt_float(agg.get('rps'), 3)} req/s, {fmt_float(agg.get('output_tps'), 1)} out_tok/s)"
    ) if agg.get("successful") else "no successful requests yet"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>agentic_bench report - {html.escape(run_dir.name)}</title>
<style>
  :root {{
    color-scheme: light dark;
    --fg: #1a1a1a; --bg: #ffffff; --muted: #666; --border: #d0d0d0; --band: #fafafa;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg: #e6e6e6; --bg: #141414; --muted: #9a9a9a; --border: #2e2e2e; --band: #1a1a1a; }}
  }}
  body {{ font: 14px/1.45 -apple-system, system-ui, "Segoe UI", sans-serif;
          color: var(--fg); background: var(--bg); margin: 1.25rem; max-width: 1100px; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.2rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 0.2rem; }}
  h3 {{ font-size: 0.98rem; margin: 1rem 0 0.3rem; color: var(--muted); font-weight: 600; }}
  .subtitle {{ color: var(--muted); }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1rem; font-size: 0.9rem; }}
  th, td {{ border: 1px solid var(--border); padding: 3px 8px; text-align: left; vertical-align: top; }}
  th {{ background: var(--band); font-weight: 600; }}
  tbody tr:nth-child(even) {{ background: var(--band); }}
  pre {{ background: var(--band); border: 1px solid var(--border); padding: 8px; overflow-x: auto; font-size: 0.82rem; }}
  .chart {{ margin: 0.5rem 0 0.75rem; }}
  .chart-title {{ font-size: 0.85rem; color: var(--muted); margin-bottom: 2px; }}
  .chart svg {{ background: var(--band); border: 1px solid var(--border); color: #3478ff; }}
  .xlabel, .ylabel {{ font-size: 10px; fill: var(--muted); }}
  nav a {{ margin-right: 0.75rem; color: var(--muted); text-decoration: none; }}
  nav a:hover {{ text-decoration: underline; }}
  #req-controls {{ margin-bottom: 0.3rem; display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }}
  #req-controls input, #req-controls select {{ font: inherit; padding: 2px 6px; background: var(--bg); color: var(--fg); border: 1px solid var(--border); }}
  .req-status-successful {{ color: #1a7f37; }}
  .req-status-errored    {{ color: #b42318; font-weight: 600; }}
  .req-status-incomplete {{ color: #b07000; }}
  .footer {{ margin-top: 2rem; color: var(--muted); font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>agentic_bench report</h1>
<p class="subtitle">
  run: <code>{html.escape(str(run_dir))}</code>
  &middot; generated: {html.escape(datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))}
  &middot; chunks: {len(chunks)}
  &middot; {html.escape(agg_summary)}
</p>
<nav>
  <a href="#config">config</a>
  <a href="#aggregate">aggregate</a>
  <a href="#timeline">timeline</a>
  <a href="#hourly">hourly</a>
  <a href="#decile">by prompt size</a>
  <a href="#chunks">chunks</a>
  <a href="#requests">requests</a>
</nav>

<h2 id="config">config</h2>
<pre>{html.escape(config_text)}</pre>

<h2 id="aggregate">aggregate percentiles</h2>
{agg_table}

<h2 id="timeline">timeline (60-second buckets, wall-clock UTC)</h2>
{render_timeline_svg(tl)}

<h2 id="hourly">by wall-clock hour</h2>
{hourly_table}

<h2 id="decile">by prompt-tokens decile</h2>
{decile_table}

<h2 id="chunks">per-chunk summary</h2>
{render_chunk_table(chunks)}

<h2 id="requests">per-request table</h2>
<p class="subtitle">Client-side paginated. Use the filter to grep by
<code>response_id</code> for server-log correlation.</p>
<div id="req-controls">
  <label>filter <input id="req-filter" type="text" placeholder="response_id, status, ..."></label>
  <label>status
    <select id="req-status">
      <option value="">all</option>
      <option value="successful">successful</option>
      <option value="errored">errored</option>
      <option value="incomplete">incomplete</option>
    </select>
  </label>
  <label>page size
    <select id="req-pagesize">
      <option>50</option><option selected>200</option><option>1000</option>
    </select>
  </label>
  <span>page <span id="req-page">1</span> / <span id="req-pages">1</span>
        &middot; <span id="req-total">0</span> rows</span>
  <button id="req-prev">prev</button>
  <button id="req-next">next</button>
</div>
<table id="req-table">
  <thead>
    <tr>
      <th>chunk</th><th>status</th><th>start (UTC)</th>
      <th>ttft (ms)</th><th>itl (ms)</th><th>tpot (ms)</th>
      <th>lat (s)</th><th>in tok</th><th>out tok</th>
      <th>response_id</th>
    </tr>
  </thead>
  <tbody id="req-body"></tbody>
</table>

<p class="footer">
  Raw merged data is embedded below as <code>&lt;script type="application/json" id="data"&gt;</code>.
  Extract with: <code>python -c "import json, re, sys; d=re.search(r'id=\\\"data\\\"&gt;(.*?)&lt;/script&gt;', open(sys.argv[1]).read(), re.S).group(1); json.dump(json.loads(d), sys.stdout)" report.html</code>
</p>

<script type="application/json" id="data">{raw_json}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById("data").textContent);
  const reqs = data.requests || [];
  const $ = (id) => document.getElementById(id);
  const body = $("req-body");
  const state = {{ page: 1, pageSize: 200, filter: "", status: "" }};

  const fmtTs = (ts) => {{
    if (ts == null) return "-";
    const d = new Date(ts * 1000);
    return d.toISOString().replace("T", " ").replace(/\\.\\d+Z$/, "");
  }};
  const fmtF = (v, d=2) => v == null ? "-" : Number(v).toFixed(d);

  const filtered = () => reqs.filter(r => {{
    if (state.status && r.status !== state.status) return false;
    if (!state.filter) return true;
    const q = state.filter.toLowerCase();
    return (r.response_id||"").toLowerCase().includes(q)
        || (r.request_id||"").toLowerCase().includes(q)
        || (r.status||"").toLowerCase().includes(q);
  }});

  const render = () => {{
    const xs = filtered();
    const pages = Math.max(1, Math.ceil(xs.length / state.pageSize));
    if (state.page > pages) state.page = pages;
    const off = (state.page - 1) * state.pageSize;
    const slice = xs.slice(off, off + state.pageSize);
    body.innerHTML = slice.map(r => `
      <tr>
        <td>${{r.chunk}}</td>
        <td class="req-status-${{r.status}}">${{r.status}}</td>
        <td>${{fmtTs(r.start)}}</td>
        <td>${{fmtF(r.ttft_ms)}}</td>
        <td>${{fmtF(r.itl_ms)}}</td>
        <td>${{fmtF(r.tpot_ms)}}</td>
        <td>${{fmtF(r.latency_s)}}</td>
        <td>${{r.prompt_tokens ?? "-"}}</td>
        <td>${{r.output_tokens ?? "-"}}</td>
        <td><code>${{(r.response_id ?? "-").replace(/[<>&]/g, c => ({{
          '<':'&lt;','>':'&gt;','&':'&amp;'
        }}[c]))}}</code></td>
      </tr>`).join("");
    $("req-page").textContent   = state.page;
    $("req-pages").textContent  = pages;
    $("req-total").textContent  = xs.length;
  }};

  $("req-filter").addEventListener("input", e => {{ state.filter = e.target.value; state.page = 1; render(); }});
  $("req-status").addEventListener("change", e => {{ state.status = e.target.value; state.page = 1; render(); }});
  $("req-pagesize").addEventListener("change", e => {{ state.pageSize = +e.target.value; state.page = 1; render(); }});
  $("req-prev").addEventListener("click", () => {{ if (state.page > 1) {{ state.page--; render(); }} }});
  $("req-next").addEventListener("click", () => {{ state.page++; render(); }});
  render();
}})();
</script>
</body>
</html>
"""


def _json_default(obj: Any) -> Any:
    # Handle pydantic BaseModels / Paths / sets
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    chunks = load_chunks(run_dir)
    if not chunks:
        print(f"warn: no chunks found under {run_dir}", file=sys.stderr)
        # still write an empty report so the user sees something
    rows = extract_rows(chunks)

    config_file = run_dir / "config.txt"
    config_text = config_file.read_text() if config_file.is_file() else "(no config.txt)"

    html_doc = render_html(run_dir, config_text, chunks, rows)
    out = run_dir / "report.html"
    out.write_text(html_doc)
    print(f"wrote {out} ({len(rows)} requests across {len(chunks)} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
