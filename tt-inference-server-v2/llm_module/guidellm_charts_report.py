# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from report_module.display import DISPLAY_NAMES

OUT_DIR = Path(__file__).parent / "output"


def _extract_run_meta(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = data.get("Run Configuration") or {}
    shape = str(cfg.get("shape") or "")
    strategy = str(cfg.get("strategy") or "")
    isl = osl = concurrency = None
    m = re.search(r"prompt_tokens=(\d+).*?output_tokens=(\d+)", shape)
    if m:
        isl, osl = int(m.group(1)), int(m.group(2))
    mc = re.search(r"max_concurrency=(\d+)", strategy)
    if mc:
        concurrency = int(mc.group(1))
    return {"isl": isl, "osl": osl, "concurrency": concurrency}


def load_sweeps(out_dir: Path) -> List[Dict[str, Any]]:
    sweeps: List[Dict[str, Any]] = []
    for p in sorted(out_dir.glob("*_parsed.json")):
        block = json.loads(p.read_text())
        data = block.get("data") or {}
        targets = block.get("targets") or {}
        sweeps.append(
            {
                "file": p.name,
                "kind": block.get("kind", ""),
                "model": targets.get("model", ""),
                "timestamp": targets.get("timestamp", ""),
                **_extract_run_meta(data),
                "data": data,
            }
        )
    return sweeps


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GuideLLM Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    color-scheme: dark;
    --carbon:  #0D0F12;
    --violet:  #1F1035;
    --neon:    #9A4BFF;
    --volt:    #BAFF39;
    --grey:    #6E7A8A;
    --white:   #F3F4F6;
    --card-bg: rgba(31, 16, 53, 0.32);
    --border:  rgba(110, 122, 138, 0.22);
    --muted:   #6E7A8A;
  }
  body {
    font-family: -apple-system, ui-sans-serif, system-ui, sans-serif;
    margin: 16px 24px 32px;
    color: var(--white);
    background:
      radial-gradient(900px 600px at 85% -10%, rgba(154, 75, 255, 0.18), transparent 60%),
      radial-gradient(700px 500px at -10% 110%, rgba(31, 16, 53, 0.65), transparent 55%),
      var(--carbon);
    min-height: 100vh;
  }
  h1 { margin: 0 0 4px; font-size: 20px; font-weight: 600; color: var(--white); letter-spacing: .01em; }
  .meta {
    display: flex;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
    background: rgba(31, 16, 53, 0.55);
    border: 1px solid rgba(154, 75, 255, 0.32);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 22px;
    font-size: 14px;
    box-shadow: 0 0 36px -14px rgba(154, 75, 255, 0.45);
  }
  .meta label {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: var(--white);
    text-transform: uppercase;
    letter-spacing: .1em;
    font-size: 12px;
    font-weight: 600;
  }
  .meta select {
    background: rgba(13, 15, 18, 0.92);
    color: var(--white);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 6px;
    font: inherit;
    font-size: 15px;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
    min-width: 96px;
    cursor: pointer;
  }
  .meta select:hover { border-color: var(--neon); }
  .meta select:focus { outline: 1px solid var(--volt); border-color: var(--volt); }
  .meta #sweep-meta { color: var(--muted); font-size: 12px; margin-left: auto; }
  .kpis { display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin-bottom: 16px; }
  .kpi {
    background: var(--card-bg);
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    box-shadow: 0 0 24px -12px rgba(31, 16, 53, 0.9);
  }
  .kpi .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
  .kpi .value { font-size: 19px; font-weight: 600; margin-top: 4px; color: var(--volt); font-variant-numeric: tabular-nums; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 0 32px -16px rgba(31, 16, 53, 0.85);
  }
  .card h3 { margin: 0 0 10px; font-size: 12px; font-weight: 500; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; }
  .full { grid-column: 1 / -1; }
  .chart { width: 100%; height: 320px; }
  .chart.tall { height: 380px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { padding: 5px 8px; border-bottom: 1px solid var(--border); text-align: left; color: var(--white); }
  th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .pass { color: var(--volt); font-weight: 600; }
  .fail { color: #ff5577; font-weight: 600; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 14px; font-size: 12px; }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; font-variant-numeric: tabular-nums; color: var(--white); }
  .empty { color: var(--muted); font-size: 12px; padding: 14px; }
</style>
</head>
<body>
<h1>GuideLLM Benchmark Report</h1>
<div class="meta">
  <label>Model <select id="sel-model"></select></label>
  <label>ISL <select id="sel-isl"></select></label>
  <label>OSL <select id="sel-osl"></select></label>
  <label>Concurrency <select id="sel-c"></select></label>
  <span id="sweep-meta"></span>
</div>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="card"><h3>Latency Percentiles (ms)</h3><div id="chart-percentiles" class="chart"></div></div>
  <div class="card"><h3>Per-Request Latency Breakdown (ms)</h3><div id="chart-breakdown" class="chart"></div></div>
  <div class="card"><h3>Per-Turn Metrics</h3><div id="chart-perturn" class="chart"></div></div>
  <div class="card"><h3>Cold vs Warm TTFT</h3><div id="chart-coldwarm" class="chart"></div></div>
  <div class="card"><h3>Time Accounting</h3><div id="chart-time" class="chart"></div></div>
  <div class="card"><h3>SLO Checks</h3><div id="slo-table"></div></div>
  <div class="card"><h3>Top 3 Latency Outliers</h3><div id="outliers-table"></div></div>
  <div class="card"><h3>Run Configuration</h3><div id="runcfg"></div></div>
  <div class="card full"><h3>Cross-Sweep — Latency &amp; Throughput vs Concurrency</h3><div id="chart-sweeps" class="chart tall"></div></div>
  <div class="card full">
    <h3>Cross-Sweep — ISL × OSL Heatmap (TTFT p95)</h3>
    <div style="margin-bottom: 8px; font-size: 12px; color: var(--muted);">
      Model: <select id="heatmap-model"></select>
      &nbsp;Concurrency: <select id="heatmap-c"></select>
    </div>
    <div id="chart-heatmap" class="chart tall"></div>
  </div>
</div>

<script>
const SWEEPS = __SWEEPS_JSON__;
const DISPLAY_NAMES = __DISPLAY_NAMES_JSON__;
function dn(key) { return (key != null && DISPLAY_NAMES[key]) || key; }

const PALETTE = {
  carbon: "#0D0F12",
  violet: "#1F1035",
  neon:   "#9A4BFF",
  volt:   "#BAFF39",
  grey:   "#6E7A8A",
  white:  "#F3F4F6",
};
const GRID = "rgba(110, 122, 138, 0.18)";
const AXIS = "rgba(110, 122, 138, 0.5)";
const baseLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: PALETTE.white, size: 11 },
  margin: { l: 56, r: 24, t: 10, b: 44 },
  legend: { orientation: "h", y: -0.22, font: { size: 10, color: PALETTE.white } },
  colorway: [
    PALETTE.volt, PALETTE.neon, "#7AE0F2", "#FF8A65",
    "#C77DFF", "#A4F25B", PALETTE.grey, "#FFB454",
  ],
  xaxis: { gridcolor: GRID, zerolinecolor: AXIS, linecolor: AXIS, tickfont: { color: PALETTE.grey } },
  yaxis: { gridcolor: GRID, zerolinecolor: AXIS, linecolor: AXIS, tickfont: { color: PALETTE.grey } },
  hoverlabel: {
    bgcolor: PALETTE.carbon,
    bordercolor: "rgba(154, 75, 255, 0.55)",
    font: { color: PALETTE.white, size: 12, family: "ui-monospace, SFMono-Regular, monospace" },
  },
};
const VAL_HOVER = "%{y}<extra></extra>";

function wrapLabel(s, maxLen = 12) {
  if (!s || s.length <= maxLen) return s;
  const seps = [];
  for (let i = 0; i < s.length; i++) {
    if (/[\s_\-]/.test(s[i])) seps.push(i);
  }
  if (!seps.length) return s;
  const mid = s.length / 2;
  seps.sort((a, b) => Math.abs(a - mid) - Math.abs(b - mid));
  const k = seps[0];
  return s.slice(0, k) + "<br>" + s.slice(k + 1);
}
const baseConfig = { responsive: true, displayModeBar: false };
const HEATMAP_SCALE = [
  [0.0, PALETTE.violet],
  [0.45, PALETTE.neon],
  [1.0, PALETTE.volt],
];
const PCT_COLORS = {
  p50:  "#FF8A65",  // orange
  p90:  "#C77DFF",  // light purple
  p95:  PALETTE.neon, // purple
  p99:  "#7AE0F2",  // light blue
  p999: PALETTE.volt, // green
};
const BRK_COLORS = {
  min:  "#adcaf7",  // light blue
  p50:  "#FF8A65",  // orange (matches percentile chart p50)
  mean: "#4287f5",  // medium blue
  p95:  PALETTE.neon, // purple — matches percentile chart p95
  max:  "#113f85",  // dark blue
};

const selModel = document.getElementById("sel-model");
const selIsl = document.getElementById("sel-isl");
const selOsl = document.getElementById("sel-osl");
const selC = document.getElementById("sel-c");

function uniqSorted(xs) {
  return [...new Set(xs.filter(v => v != null))].sort((a, b) =>
    typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b))
  );
}

function fillSelect(el, values) {
  const prev = el.value;
  el.innerHTML = "";
  values.forEach(v => {
    const o = document.createElement("option");
    o.value = String(v); o.textContent = String(v);
    el.appendChild(o);
  });
  if (values.map(String).includes(prev)) el.value = prev;
}

function repopulateSelectors(changed) {
  const m = selModel.value;
  if (changed === "init" || changed === "model") {
    fillSelect(selIsl, uniqSorted(SWEEPS.filter(s => s.model === m).map(s => s.isl)));
  }
  if (changed === "init" || changed === "model" || changed === "isl") {
    const i = +selIsl.value;
    fillSelect(selOsl, uniqSorted(SWEEPS.filter(s => s.model === m && s.isl === i).map(s => s.osl)));
  }
  const i = +selIsl.value, o = +selOsl.value;
  fillSelect(selC, uniqSorted(SWEEPS.filter(s => s.model === m && s.isl === i && s.osl === o).map(s => s.concurrency)));
}

function refreshFromSelection() {
  const m = selModel.value;
  const i = +selIsl.value, o = +selOsl.value, c = +selC.value;
  const idx = SWEEPS.findIndex(s => s.model === m && s.isl === i && s.osl === o && s.concurrency === c);
  if (idx >= 0) render(idx);
}

selModel.addEventListener("change", () => { repopulateSelectors("model"); refreshFromSelection(); });
selIsl.addEventListener("change", () => { repopulateSelectors("isl"); refreshFromSelection(); });
selOsl.addEventListener("change", () => { repopulateSelectors("osl"); refreshFromSelection(); });
selC.addEventListener("change", refreshFromSelection);

fillSelect(selModel, uniqSorted(SWEEPS.map(s => s.model)));
repopulateSelectors("init");

function fmt(v, d = 2) {
  if (v == null || v === "" || Number.isNaN(v)) return "—";
  if (typeof v !== "number") return String(v);
  return v >= 1000 ? v.toFixed(0) : v.toFixed(d);
}

function render(idx) {
  const s = SWEEPS[idx];
  const d = s.data || {};
  document.getElementById("sweep-meta").textContent =
    `· ${s.model} · ISL=${s.isl} OSL=${s.osl} concurrency=${s.concurrency}`;
  renderKpis(d["Key Takeaways"] || {});
  renderPercentiles(d["Latency Percentiles (ms)"] || []);
  renderBreakdown(d["Per-Request Latency Breakdown (ms)"] || []);
  renderPerTurn(d["Per-Turn Breakdown"] || []);
  renderColdWarm(d["Cold vs. Warm TTFT"] || {});
  renderTime(d["Time Accounting"] || {});
  renderSlo(d["SLO Checks"] || []);
  renderOutliers(d["Top 3 Latency Outliers"] || []);
  renderRunCfg(d["Run Configuration"] || {});
}

function renderKpis(kt) {
  const items = [
    ["TTFT p50 (ms)", kt.TTFT_median_ms],
    ["ITL p50 (ms)", kt.ITL_median_ms],
    ["TPOT p50 (ms)", kt.TPOT_median_ms],
    ["Latency p50 (ms)", kt.request_latency_median_ms],
    ["out tok/s (agg)", kt.output_tokens_per_second_agg],
    ["Concurrency", kt.concurrency_observed],
    ["Per-user tok/s", kt.per_user_out_tps],
  ];
  document.getElementById("kpis").innerHTML = items.map(([l, v]) =>
    `<div class="kpi"><div class="label">${l}</div><div class="value">${fmt(v)}</div></div>`
  ).join("");
}

function renderPercentiles(rows) {
  const ps = ["p50","p90","p95","p99","p999"];
  const normalized = rows.map(r => {
    if (r.metric === "request_latency") {
      const out = { label: "Request Latency (ms)" };
      for (const p of ps) out[p] = (r[p] ?? 0) * 1000;
      return out;
    }
    return { label: dn(r.metric), p50: r.p50, p90: r.p90, p95: r.p95, p99: r.p99, p999: r.p999 };
  });
  const traces = ps.map(p => ({
    type: "bar", name: p,
    x: normalized.map(r => wrapLabel(r.label)),
    y: normalized.map(r => r[p]),
    marker: { color: PCT_COLORS[p] },
    hovertemplate: VAL_HOVER,
  }));
  Plotly.newPlot("chart-percentiles", traces, {
    ...baseLayout,
    barmode: "group",
    margin: { ...baseLayout.margin, b: 56 },
    yaxis: { ...baseLayout.yaxis, type: "log", title: "ms (log scale)" },
    xaxis: { ...baseLayout.xaxis, tickangle: 0 },
  }, baseConfig);
}

function renderBreakdown(rows) {
  if (!rows.length) { document.getElementById("chart-breakdown").innerHTML = '<div class="empty">No breakdown data</div>'; return; }
  const segs = rows.map(r => wrapLabel(r.segment, 14));
  const series = [
    ["min", "min_ms"], ["p50", "p50_ms"], ["mean", "mean_ms"], ["p95", "p95_ms"], ["max", "max_ms"],
  ];
  const traces = series.map(([n, k]) => ({
    type: "bar", name: n, x: segs, y: rows.map(r => r[k]),
    marker: { color: BRK_COLORS[n] },
    hovertemplate: VAL_HOVER,
  }));
  Plotly.newPlot("chart-breakdown", traces, {
    ...baseLayout,
    barmode: "group",
    margin: { ...baseLayout.margin, b: 56 },
    yaxis: { ...baseLayout.yaxis, type: "log", title: "ms (log scale)" },
    xaxis: { ...baseLayout.xaxis, tickangle: 0 },
  }, baseConfig);
}

function renderPerTurn(rows) {
  if (!rows.length) { document.getElementById("chart-perturn").innerHTML = '<div class="empty">No turn data</div>'; return; }
  const x = rows.map(r => r.turn);
  const traces = [
    { name: "TTFT (ms)", x, y: rows.map(r => r.ttft_ms), type: "scatter", mode: "lines+markers", hovertemplate: VAL_HOVER },
    { name: "ITL (ms)", x, y: rows.map(r => r.itl_ms), type: "scatter", mode: "lines+markers", yaxis: "y2", hovertemplate: VAL_HOVER },
    { name: "TPOT (ms)", x, y: rows.map(r => r.tpot_ms), type: "scatter", mode: "lines+markers", yaxis: "y2", hovertemplate: VAL_HOVER },
    { name: "out tok/s", x, y: rows.map(r => r.out_tps), type: "scatter", mode: "lines+markers", yaxis: "y3", line: { dash: "dot" }, hovertemplate: VAL_HOVER },
  ];
  Plotly.newPlot("chart-perturn", traces, {
    ...baseLayout,
    xaxis: { ...baseLayout.xaxis, title: "turn index" },
    yaxis: { ...baseLayout.yaxis, title: "TTFT (ms)" },
    yaxis2: { title: "ITL / TPOT (ms)", overlaying: "y", side: "right", gridcolor: "rgba(0,0,0,0)", tickfont: { color: PALETTE.grey } },
    yaxis3: { overlaying: "y", side: "right", anchor: "free", position: 1, showticklabels: false, showgrid: false },
  }, baseConfig);
}

function renderColdWarm(d) {
  const mean = s => { const m = String(s||"").match(/mean_ttft = ([\d.]+)/); return m ? parseFloat(m[1]) : null; };
  const cold = mean(d.turn_0_cold);
  const warmKey = Object.keys(d).find(k => k.startsWith("turns_"));
  const warm = mean(d[warmKey]);
  Plotly.newPlot("chart-coldwarm", [{
    type: "bar",
    x: ["cold (turn 0)", "warm (turn 1+)"],
    y: [cold, warm],
    marker: { color: [PALETTE.neon, PALETTE.volt] },
    text: [fmt(cold), fmt(warm)],
    textposition: "outside",
    hovertemplate: VAL_HOVER,
  }], { ...baseLayout, yaxis: { ...baseLayout.yaxis, title: "mean TTFT (ms)" } }, baseConfig);
}

function renderTime(d) {
  const labels = ["active", "idle", "warmup", "cooldown"];
  const values = [d.active_s, d.idle_s, d.warmup_s, d.cooldown_s].map(v => Math.max(0, +v || 0));
  Plotly.newPlot("chart-time", [{
    type: "pie", labels, values, hole: 0.5,
    textinfo: "label+percent",
    marker: { colors: [PALETTE.volt, PALETTE.grey, PALETTE.neon, PALETTE.violet] },
  }], { ...baseLayout, showlegend: false }, baseConfig);
}

function renderSlo(rows) {
  if (!rows.length) { document.getElementById("slo-table").innerHTML = '<div class="empty">No SLO data</div>'; return; }
  const html = '<table><thead><tr><th>Target</th><th>Observed</th><th>Result</th></tr></thead><tbody>' +
    rows.map(r => `<tr><td>${r.slo_target}</td><td class="num">${r.observed || ""}</td><td class="${(r.result||"").toLowerCase()}">${r.result || ""}</td></tr>`).join("") +
    '</tbody></table>';
  document.getElementById("slo-table").innerHTML = html;
}

function renderOutliers(rows) {
  if (!rows.length) { document.getElementById("outliers-table").innerHTML = '<div class="empty">No outliers</div>'; return; }
  const html = '<table><thead><tr><th>#</th><th>turn</th><th class="num">ptok</th><th class="num">lat ms</th><th class="num">TTFT ms</th><th class="num">prefill ms</th><th class="num">decode ms</th><th>note</th></tr></thead><tbody>' +
    rows.map(r => `<tr><td>${r.rank}</td><td>${r.turn ?? ""}</td><td class="num">${r.ptok ?? ""}</td><td class="num">${fmt(r.lat_ms)}</td><td class="num">${fmt(r.ttft_ms)}</td><td class="num">${fmt(r.prefill_ms)}</td><td class="num">${fmt(r.decode_ms)}</td><td>${r.note || ""}</td></tr>`).join("") +
    '</tbody></table>';
  document.getElementById("outliers-table").innerHTML = html;
}

function renderRunCfg(c) {
  const order = ["model","backend","strategy","shape","guidellm","python","platform","wall_time_s","warmup_s","cooldown_s"];
  const items = order.filter(k => k in c).map(k => `<dt>${k}</dt><dd>${c[k]}</dd>`).join("");
  document.getElementById("runcfg").innerHTML = `<dl class="kv">${items}</dl>`;
}

function renderSweepsTrend() {
  const groups = new Map();
  SWEEPS.forEach(s => {
    const key = `${s.model} | ISL=${s.isl} OSL=${s.osl}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(s);
  });
  const traces = [];
  let i = 0;
  for (const [key, runs] of groups) {
    runs.sort((a, b) => (a.concurrency ?? 0) - (b.concurrency ?? 0));
    const x = runs.map(r => r.concurrency);
    const ttft = runs.map(r => {
      const lp = (r.data["Latency Percentiles (ms)"] || []).find(p => p.metric === "time_to_first_token_ms");
      return lp ? lp.p95 : null;
    });
    const outTps = runs.map(r => {
      const ss = (r.data["Summary Statistics"] || []).find(p => p.metric === "output_tokens_per_second");
      return ss ? ss.mean : null;
    });
    const color = baseLayout.colorway[i % baseLayout.colorway.length];
    traces.push({ name: `${key} · TTFT p95`, x, y: ttft, type: "scatter", mode: "lines+markers", line: { color }, legendgroup: key, hovertemplate: VAL_HOVER });
    traces.push({ name: `${key} · out tok/s`, x, y: outTps, type: "scatter", mode: "lines+markers", line: { color, dash: "dot" }, yaxis: "y2", legendgroup: key, hovertemplate: VAL_HOVER });
    i++;
  }
  Plotly.newPlot("chart-sweeps", traces, {
    ...baseLayout,
    xaxis: { title: "max_concurrency", type: "log" },
    yaxis: { title: "TTFT p95 (ms)", type: "log" },
    yaxis2: { title: "out tok/s", overlaying: "y", side: "right", type: "log" },
  }, baseConfig);
}

const heatmapModelEl = document.getElementById("heatmap-model");
const heatmapCEl = document.getElementById("heatmap-c");

function populateHeatmapFilters() {
  const models = [...new Set(SWEEPS.filter(s => s.isl != null && s.osl != null).map(s => s.model))].sort();
  models.forEach(m => {
    const o = document.createElement("option"); o.value = m; o.textContent = m;
    heatmapModelEl.appendChild(o);
  });
  refreshHeatmapConcurrencies();
  heatmapModelEl.addEventListener("change", () => { refreshHeatmapConcurrencies(); renderHeatmap(); });
  heatmapCEl.addEventListener("change", renderHeatmap);
}

function refreshHeatmapConcurrencies() {
  const m = heatmapModelEl.value;
  const cs = [...new Set(SWEEPS.filter(s => s.model === m && s.isl != null && s.osl != null).map(s => s.concurrency))]
    .filter(c => c != null).sort((a, b) => a - b);
  heatmapCEl.innerHTML = "";
  cs.forEach(c => {
    const o = document.createElement("option"); o.value = c; o.textContent = c;
    heatmapCEl.appendChild(o);
  });
  // Prefer the most common concurrency (likely the grid sweep level)
  const counts = new Map();
  SWEEPS.filter(s => s.model === m).forEach(s => counts.set(s.concurrency, (counts.get(s.concurrency) || 0) + 1));
  let best = cs[0], n = 0;
  for (const [c, k] of counts) if (cs.includes(c) && k > n) { best = c; n = k; }
  if (best != null) heatmapCEl.value = best;
}

function renderHeatmap() {
  const model = heatmapModelEl.value;
  const c = +heatmapCEl.value;
  const runs = SWEEPS.filter(s => s.model === model && s.concurrency === c && s.isl != null && s.osl != null);
  if (!runs.length) {
    Plotly.purge("chart-heatmap");
    document.getElementById("chart-heatmap").innerHTML = '<div class="empty">No sweeps for that model + concurrency</div>';
    return;
  }
  const isls = [...new Set(runs.map(r => r.isl))].sort((a, b) => a - b);
  const osls = [...new Set(runs.map(r => r.osl))].sort((a, b) => a - b);
  const z = osls.map(o => isls.map(i => {
    const match = runs.find(r => r.isl === i && r.osl === o);
    if (!match) return null;
    const lp = (match.data["Latency Percentiles (ms)"] || []).find(p => p.metric === "time_to_first_token_ms");
    return lp ? lp.p95 : null;
  }));
  Plotly.newPlot("chart-heatmap", [{
    type: "heatmap",
    x: isls.map(String), y: osls.map(String), z,
    colorscale: HEATMAP_SCALE,
    colorbar: {
      title: { text: "TTFT p95 ms", font: { color: PALETTE.grey } },
      tickfont: { color: PALETTE.grey },
      outlinecolor: AXIS,
    },
    hovertemplate: "ISL=%{x} OSL=%{y}<br>TTFT p95 = %{z} ms<extra></extra>",
  }], {
    ...baseLayout,
    xaxis: { ...baseLayout.xaxis, title: "ISL (prompt tokens)" },
    yaxis: { ...baseLayout.yaxis, title: "OSL (output tokens)" },
  }, baseConfig);
}

refreshFromSelection();
renderSweepsTrend();
populateHeatmapFilters();
renderHeatmap();
</script>
</body>
</html>
"""


def build_html(sweeps: List[Dict[str, Any]]) -> str:
    payload = json.dumps(sweeps).replace("</", "<\\/")
    names = json.dumps(DISPLAY_NAMES).replace("</", "<\\/")
    return (
        HTML_TEMPLATE.replace("__SWEEPS_JSON__", payload)
        .replace("__DISPLAY_NAMES_JSON__", names)
    )


def main() -> None:
    sweeps = load_sweeps(OUT_DIR)
    if not sweeps:
        raise SystemExit(f"no *_parsed.json under {OUT_DIR}")
    html = build_html(sweeps)
    out = OUT_DIR / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(sweeps)} sweep(s))")


if __name__ == "__main__":
    main()
