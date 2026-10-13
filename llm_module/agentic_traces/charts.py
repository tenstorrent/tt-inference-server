# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Per-point telemetry charts for one InferenceX/AIPerf agentic-trace run.

Reads the artifacts the AIPerf fork leaves in a run's artifact dir --
``profile_export.jsonl`` (one record per replayed request) and
``profile_export_aiperf_timeslices.csv`` (1 s windows of throughput and load) --
and writes dependency-free SVG files that the markdown report links with
``![alt](path)``. Plain standard library on purpose: this runs in the
orchestrator venv and the reports venv, neither of which carries a plotting
library, and an SVG stays crisp and diffable.

Charts, each a separate file so the report can order and caption them:

* ``isl_osl``        input / output sequence-length histograms
* ``latency``        TTFT and end-to-end latency per request over the run
* ``interactivity``  per-request output tokens/s/user over the run
* ``throughput``     prefill and decode throughput over the run
* ``load``           effective concurrency, prefix-cache hit rate, completed requests

The load chart's cache panel is the prefix-cache hit rate over time, read from
the serving engine's counters in ``server_metrics_export.json`` (hundreds of MB,
so streamed, not parsed whole); without that file it falls back to the client's
tokens-in-flight. KV-cache utilisation and queue depth are in the same file but
are not drawn.
"""

from __future__ import annotations

import bisect
import csv
import html
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROFILE_EXPORT = "profile_export.jsonl"
TIMESLICES = "profile_export_aiperf_timeslices.csv"
SERVER_METRICS = "server_metrics_export.json"

# Each ``<name>.svg`` gets a ``<name>.mmd`` markdown fragment of Mermaid charts:
# the only chart form a GitHub job summary renders (see "Mermaid twins").
MERMAID_SUFFIX = ".mmd"

# Keys under which the engine-derived hit-rate series ride in the same mapping
# as the client timeslices.
HIT_RATE_ROLLING = "Prefix Cache Hit Rate (rolling)"
HIT_RATE_CUMULATIVE = "Prefix Cache Hit Rate (cumulative)"
_HIT_RATE_WINDOW_S = 30

# Payload key prefix for a chart's report-relative link (``<prefix><name>``).
# The report renderer duplicates it: report_module does not import llm_module.
TELEMETRY_CHART_PREFIX = "telemetry_chart_"

# Human titles, keyed by chart name; the renderer uses them as captions and the
# order here is the order they appear in the report.
CHART_TITLES: Dict[str, str] = {
    "isl_osl": "Sequence length distribution",
    "latency": "Latency over the run",
    "interactivity": "Interactivity over the run",
    "throughput": "Prefill and decode throughput",
    "load": "Load over the run",
}

_WIDTH = 900
_MARGIN_L = 64
_MARGIN_R = 24
_PANEL_TITLE_H = 22
_PANEL_AXIS_H = 34

# Colors are CSS variables so one file follows the viewer's light/dark setting.
# Series hues are the first slots of the validated categorical palette.
_STYLE = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--axis:#b9b8b2;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;
--s7:#4a3aa7;--bad:#e34948;--mute:#b9b8b2}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--grid:#34332f;--axis:#6b6a65;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;
--s7:#9085e9;--bad:#e66767;--mute:#6b6a65}}
text{font-family:system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;fill:var(--ink2);font-size:11px}
.title{fill:var(--ink);font-size:13px;font-weight:600}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
"""

_SLICE_METRICS = (
    "Effective Prefill Throughput",
    "Effective Decode Throughput",
    "Effective Concurrency",
    "Tokens In Flight",
)

_SERIES = ("var(--s1)", "var(--s2)", "var(--s3)")
_ROLLING_WINDOW = 50
_MAX_DOTS = 1200
_MAX_LINE_POINTS = 1500


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass
class Request:
    """One replayed request, times in seconds relative to the profiling start."""

    start: float
    end: float
    phase: str  # "warmup" | "profiling"
    ok: bool
    conversation: str
    root: str
    depth: int
    turn: int
    session: int = 0
    isl: Optional[float] = None
    osl: Optional[float] = None
    ttft: Optional[float] = None  # seconds
    e2el: Optional[float] = None  # seconds
    tps_user: Optional[float] = None  # output tokens/s/user


def _metric(metrics: Dict, name: str) -> Optional[float]:
    entry = metrics.get(name)
    value = entry.get("value") if isinstance(entry, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def load_requests(path: Path) -> Tuple[List[Request], int]:
    """Parse the per-request export; also returns the profiling-start clock origin (ns)."""
    raw: List[Tuple[Dict, Dict]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            meta = record.get("metadata") or {}
            if "request_start_ns" in meta and "request_end_ns" in meta:
                raw.append((meta, record.get("metrics") or {}))
    if not raw:
        return [], 0

    profiling_starts = [
        m["request_start_ns"] for m, _ in raw if m.get("benchmark_phase") == "profiling"
    ]
    origin = min(profiling_starts or [m["request_start_ns"] for m, _ in raw])

    requests: List[Request] = []
    for meta, metrics in raw:
        failed = bool(meta.get("was_cancelled")) or "time_to_first_token" not in metrics
        ttft = _metric(metrics, "time_to_first_token")
        e2el = _metric(metrics, "request_latency")
        requests.append(
            Request(
                start=(meta["request_start_ns"] - origin) / 1e9,
                end=(meta["request_end_ns"] - origin) / 1e9,
                phase=str(meta.get("benchmark_phase") or "profiling"),
                ok=not failed,
                conversation=str(meta.get("conversation_id") or ""),
                root=str(
                    meta.get("root_correlation_id")
                    or meta.get("x_correlation_id")
                    or ""
                ),
                depth=int(meta.get("agent_depth") or 0),
                turn=int(meta.get("turn_index") or 0),
                session=int(meta.get("session_num") or 0),
                isl=_metric(metrics, "input_sequence_length"),
                osl=_metric(metrics, "output_sequence_length"),
                ttft=ttft / 1000.0 if ttft is not None else None,
                e2el=e2el / 1000.0 if e2el is not None else None,
                tps_user=_metric(metrics, "output_token_throughput_per_user"),
            )
        )
    return requests, origin


def load_timeslices(
    path: Path, origin_ns: int, names: Iterable[str]
) -> Dict[str, List[Tuple[float, float]]]:
    """Read ``avg`` per window for the named metrics as (window midpoint s, value)."""
    wanted = set(names)
    series: Dict[str, List[Tuple[float, float]]] = {name: [] for name in wanted}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Stat") != "avg" or row.get("Metric") not in wanted:
                continue
            try:
                mid = (int(row["Start_NS"]) + int(row["End_NS"])) / 2
                value = float(row["Value"])
            except (KeyError, ValueError):
                continue
            series[row["Metric"]].append(((mid - origin_ns) / 1e9, value))
    return series


def _stream_metric_blocks(path: Path, names: Iterable[str]) -> Dict[str, Dict]:
    """Pull the named metric blocks out of ``server_metrics_export.json``.

    The export is hundreds of MB (one 1 s timeslice per endpoint per metric), so
    it is scanned line by line rather than parsed whole. It is written with
    ``indent=2``: metrics sit at four spaces under ``"metrics"``, and each
    block closes at the next four-space ``}``. Only the profiling phase
    (``"metrics"``) is read, and the scan stops at ``"warmup_metrics"``.
    A file written any other way yields nothing, which just omits the chart.
    """
    wanted = {f'    "{name}": {{': name for name in names}
    found: Dict[str, Dict] = {}
    in_metrics = False
    current: Optional[str] = None
    body: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not in_metrics:
                in_metrics = line.rstrip() == '  "metrics": {'
                continue
            if current is None:
                if line.startswith('  "warmup_metrics"') or len(found) == len(wanted):
                    break
                name = wanted.get(line.rstrip())
                if name is not None:
                    current, body = name, ["{"]
                continue
            if line.rstrip() in ("    },", "    }"):
                body.append("}")
                try:
                    found[current] = json.loads("".join(body))
                except ValueError:
                    logger.warning("Could not parse metric %s in %s", current, path)
                current = None
            else:
                body.append(line)
    return found


def _summed_windows(block: Dict, origin_ns: int) -> Dict[float, float]:
    """Counter increase per 1 s window, summed over the endpoints' series."""
    out: Dict[float, float] = {}
    for series in block.get("series") or []:
        for window in series.get("timeslices") or []:
            total = window.get("total")
            if not isinstance(total, (int, float)):
                continue
            mid = ((window["start_ns"] + window["end_ns"]) / 2 - origin_ns) / 1e9
            out[mid] = out.get(mid, 0.0) + total
    return out


def load_prefix_cache_hit_rate(
    path: Path, origin_ns: int
) -> Dict[str, List[Tuple[float, float]]]:
    """Prefix-cache hit rate over time (%), from the engines' own counters.

    Two lines: a trailing ``_HIT_RATE_WINDOW_S`` second window (hits / queries
    over it) and the cumulative rate since profiling began. A window with no
    queries has no rate and is skipped rather than drawn as 0%.
    """
    blocks = _stream_metric_blocks(
        path, ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries")
    )
    if len(blocks) < 2:
        return {}
    hits = _summed_windows(blocks["vllm:prefix_cache_hits"], origin_ns)
    queries = _summed_windows(blocks["vllm:prefix_cache_queries"], origin_ns)
    times = sorted(t for t in queries if t >= 0)
    rolling: List[Tuple[float, float]] = []
    cumulative: List[Tuple[float, float]] = []
    window: List[Tuple[float, float]] = []
    total_hits = total_queries = 0.0
    for t in times:
        h, q = hits.get(t, 0.0), queries[t]
        window.append((h, q))
        if len(window) > _HIT_RATE_WINDOW_S:
            window.pop(0)
        total_hits += h
        total_queries += q
        window_queries = sum(w[1] for w in window)
        if window_queries > 0:
            rolling.append((t, 100.0 * sum(w[0] for w in window) / window_queries))
        if total_queries > 0:
            cumulative.append((t, 100.0 * total_hits / total_queries))
    if not rolling:
        return {}
    return {HIT_RATE_ROLLING: rolling, HIT_RATE_CUMULATIVE: cumulative}


# --------------------------------------------------------------------------
# Small SVG toolkit
# --------------------------------------------------------------------------


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _ticks(lo: float, hi: float, target: int = 5) -> List[float]:
    """Round tick values covering [lo, hi]; the last tick is >= hi."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / target
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    ticks = []
    value = math.floor(lo / step) * step
    while True:
        if value >= lo - 1e-9:
            ticks.append(round(value, 10))
        if value >= hi - 1e-9:
            break
        value += step
    return ticks


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{value / 1e6:.3g}M"
    if magnitude >= 1e4:
        return f"{value / 1e3:.3g}k"
    if magnitude >= 1e3:
        return f"{value / 1e3:.2g}k"
    if magnitude >= 100 or value == int(value):
        return f"{value:.0f}"
    return f"{value:.3g}"


class _Canvas:
    def __init__(self, width: int = _WIDTH):
        self.width = width
        self.parts: List[str] = []
        self.height = 0

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def render(self, description: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-label="{_esc(description)}">'
            f"<title>{_esc(description)}</title><style>{_STYLE}</style>"
            f'<rect width="{self.width}" height="{self.height}" fill="var(--bg)"/>'
            + "".join(self.parts)
            + "</svg>\n"
        )


class _Panel:
    """A plot area with a title, light grid and tick labels."""

    def __init__(
        self,
        canvas: _Canvas,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        title: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        xlabel: str = "",
        ylabel: str = "",
        yticks: Optional[List[float]] = None,
        y_labels: bool = True,
        xticks: Optional[List[Tuple[float, str]]] = None,
        title_h: float = _PANEL_TITLE_H,
    ):
        self.c = canvas
        self.left = x + _MARGIN_L
        self.right = x + width - _MARGIN_R
        self.top = y + title_h
        self.bottom = y + height - _PANEL_AXIS_H
        self.xlim = xlim
        self.ylim = ylim
        canvas.add(f'<text class="title" x="{x + 4}" y="{y + 14}">{_esc(title)}</text>')
        if xticks is None:
            xticks = [
                (v, _fmt(v))
                for v in _ticks(
                    *xlim, target=max(3, int((self.right - self.left) / 90))
                )
            ]
        yt = yticks if yticks is not None else _ticks(*ylim, target=4)
        for value in yt:
            if not y_labels:
                break
            py = self.sy(value)
            canvas.add(
                f'<line class="grid" x1="{self.left:.1f}" x2="{self.right:.1f}" '
                f'y1="{py:.1f}" y2="{py:.1f}"/>'
                f'<text x="{self.left - 6:.1f}" y="{py + 3.5:.1f}" text-anchor="end">'
                f"{_fmt(value)}</text>"
            )
        for value, label in xticks:
            if value < xlim[0] - 1e-9 or value > xlim[1] + 1e-9:
                continue
            px = self.sx(value)
            canvas.add(
                f'<text x="{px:.1f}" y="{self.bottom + 14:.1f}" '
                f'text-anchor="middle">{_esc(label)}</text>'
            )
        canvas.add(
            f'<line class="axis" x1="{self.left:.1f}" x2="{self.right:.1f}" '
            f'y1="{self.bottom:.1f}" y2="{self.bottom:.1f}"/>'
        )
        if xlabel:
            canvas.add(
                f'<text x="{(self.left + self.right) / 2:.1f}" '
                f'y="{self.bottom + 28:.1f}" text-anchor="middle">{_esc(xlabel)}</text>'
            )
        if ylabel:
            cy = (self.top + self.bottom) / 2
            canvas.add(
                f'<text transform="translate({x + 12:.1f},{cy:.1f}) rotate(-90)" '
                f'text-anchor="middle">{_esc(ylabel)}</text>'
            )

    def sx(self, value: float) -> float:
        lo, hi = self.xlim
        return self.left + (value - lo) / (hi - lo or 1.0) * (self.right - self.left)

    def sy(self, value: float) -> float:
        lo, hi = self.ylim
        return self.bottom - (value - lo) / (hi - lo or 1.0) * (self.bottom - self.top)

    def clip_y(self, value: float) -> float:
        return min(max(value, self.ylim[0]), self.ylim[1])

    def scatter(
        self,
        points: Sequence[Tuple[float, float]],
        color: str,
        tips: Optional[Sequence[str]] = None,
        radius: float = 2.4,
        opacity: float = 0.5,
    ) -> None:
        for i, (px, py) in enumerate(points):
            tip = f"<title>{_esc(tips[i])}</title>" if tips else ""
            self.c.add(
                f'<circle cx="{self.sx(px):.1f}" cy="{self.sy(self.clip_y(py)):.1f}" '
                f'r="{radius}" fill="{color}" fill-opacity="{opacity}">{tip}</circle>'
            )

    def line(
        self,
        points: Sequence[Tuple[float, float]],
        color: str,
        width: float = 1.5,
        opacity: float = 1.0,
        dash: str = "",
    ) -> None:
        if len(points) < 2:
            return
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{self.sx(px):.1f},{self.sy(self.clip_y(py)):.1f}"
            for i, (px, py) in enumerate(points)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.c.add(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-opacity="{opacity}"{dash_attr} '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def vline(self, value: float, label: str = "") -> None:
        px = self.sx(value)
        self.c.add(
            f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{self.top:.1f}" y2="{self.bottom:.1f}" '
            'stroke="var(--axis)" stroke-dasharray="4 3"/>'
        )
        if label:
            self.c.add(
                f'<text x="{px + 4:.1f}" y="{self.top + 10:.1f}">{_esc(label)}</text>'
            )

    def legend(
        self,
        entries: Sequence[Tuple],
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> None:
        """Entries are ``(label, color[, style])``; style is ``dash``, ``line`` or an opacity."""
        right = self.right if x is None else x
        baseline = self.top - 6 if y is None else y
        cursor = right
        for entry in reversed(entries):
            label, color = entry[0], entry[1]
            style = entry[2] if len(entry) > 2 else 1.0
            cursor -= 32 + 6.2 * len(label) + 14
            if style == "dash":
                swatch = (
                    f'<line x1="{cursor:.1f}" x2="{cursor + 24:.1f}" y1="{baseline - 4:.1f}" '
                    f'y2="{baseline - 4:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="5 3"/>'
                )
            elif style == "line":
                swatch = (
                    f'<line x1="{cursor:.1f}" x2="{cursor + 24:.1f}" y1="{baseline - 4:.1f}" '
                    f'y2="{baseline - 4:.1f}" stroke="{color}" stroke-width="2.4"/>'
                )
            else:
                swatch = (
                    f'<rect x="{cursor + 3:.1f}" y="{baseline - 9:.1f}" width="10" height="10" '
                    f'rx="2" fill="{color}" fill-opacity="{style}"/>'
                )
            self.c.add(
                swatch
                + f'<text x="{cursor + 30:.1f}" y="{baseline:.1f}">{_esc(label)}</text>'
            )


def _time_limit(requests: Sequence[Request]) -> float:
    ends = [r.end for r in requests if r.phase == "profiling"] or [
        r.end for r in requests
    ]
    return max(ends) if ends else 1.0


def _stack_height(panels: int, panel_h: int) -> int:
    return panels * panel_h + 8


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def _duration(seconds: float) -> str:
    """``15m 7s`` / ``1h 5m`` style label."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    if minutes:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    return f"{secs}s"


def _time_ticks(lo: float, hi: float) -> List[Tuple[float, str]]:
    """Duration-labelled ticks on round minute boundaries."""
    span = max(hi - lo, 1.0)
    step = next(
        (
            c
            for c in (10, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400)
            if span / c <= 7
        ),
        14400,
    )
    first = math.ceil(lo / step) * step
    ticks = []
    value = first
    while value <= hi + 1e-9:
        ticks.append((float(value), ("-" if value < 0 else "") + _duration(abs(value))))
        value += step
    return ticks


def _rolling_window(count: int) -> int:
    """Rolling window in requests: InferenceX's 50 once a run has the volume.

    A short CI point (a few dozen requests) under a 50-request window makes the
    rolling line the cumulative one, drawn over each other; a tenth of the run,
    at least 5, keeps it a moving view.
    """
    return max(5, min(_ROLLING_WINDOW, count // 10))


def _rolling_percentile(
    points: Sequence[Tuple[float, float]], q: float, window: int = _ROLLING_WINDOW
) -> List[Tuple[float, float]]:
    """Percentile of the trailing ``window`` values, points ordered by x."""
    out: List[Tuple[float, float]] = []
    for i in range(len(points)):
        chunk = sorted(v for _, v in points[max(0, i - window + 1) : i + 1])
        out.append((points[i][0], _percentile(chunk, q)))
    return out


def _cumulative_percentile(
    points: Sequence[Tuple[float, float]], q: float
) -> List[Tuple[float, float]]:
    seen: List[float] = []
    out: List[Tuple[float, float]] = []
    for x, v in points:
        bisect.insort(seen, v)
        out.append((x, _percentile(seen, q)))
    return out


def _subsample(items: Sequence, limit: int = _MAX_DOTS) -> List:
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _chart_isl_osl(requests: Sequence[Request], _slices) -> Optional[str]:
    profiling = [r for r in requests if r.phase == "profiling" and r.ok]
    if not profiling:
        return None
    canvas = _Canvas()
    panel_h = 290
    canvas.height = _stack_height(1, panel_h)
    half = _WIDTH / 2
    quantiles = (
        ("p50", 0.5, "var(--s1)"),
        ("p75", 0.75, "var(--s3)"),
        ("p90", 0.9, "var(--s4)"),
        ("p95", 0.95, "var(--bad)"),
    )
    for index, (name, values) in enumerate(
        (
            ("Input sequence length", [r.isl for r in profiling]),
            ("Output sequence length", [r.osl for r in profiling]),
        )
    ):
        data = sorted(v for v in values if v is not None and v > 0)
        if not data:
            continue
        # Log-spaced bins over the full range: the lengths are heavy-tailed, and
        # a linear axis clipped at p99 hides exactly the long requests.
        lo, hi = math.log10(data[0]), math.log10(data[-1])
        if hi - lo < 1e-9:
            hi = lo + 1.0
        bins = 40
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in data:
            counts[min(int((math.log10(v) - lo) / width), bins - 1)] += 1
        ymax = _ticks(0, max(counts), 4)[-1]
        decades = [
            (float(e), _fmt(10.0**e)) for e in range(math.ceil(lo), math.floor(hi) + 1)
        ]
        # Endpoints are labelled with the actual min/max; drop decade ticks that
        # would sit on top of them.
        gap = 0.07 * (hi - lo)
        xticks = (
            [(lo, _fmt(data[0]))]
            + [t for t in decades if lo + gap < t[0] < hi - gap]
            + [(hi, _fmt(data[-1]))]
        )
        panel = _Panel(
            canvas,
            index * half,
            4,
            half,
            panel_h - 36,
            title=f"{name} distribution",
            xlim=(lo, hi),
            ylim=(0, ymax),
            xlabel=f"{len(data):,} requests · range {_fmt(data[0])}–{_fmt(data[-1])} tokens · log scale",
            ylabel="count",
            xticks=xticks,
        )
        for i, count in enumerate(counts):
            if not count:
                continue
            x0, x1 = panel.sx(lo + i * width), panel.sx(lo + (i + 1) * width)
            y0 = panel.sy(count)
            canvas.add(
                f'<rect x="{x0 + 0.5:.1f}" y="{y0:.1f}" width="{max(x1 - x0 - 1, 1):.1f}" '
                f'height="{panel.bottom - y0:.1f}" fill="var(--mute)">'
                f"<title>{_fmt(10 ** (lo + i * width))}–{_fmt(10 ** (lo + (i + 1) * width))} "
                f"tokens: {count} requests</title></rect>"
            )
        entries = []
        for label, q, color in quantiles:
            value = _percentile(data, q)
            if value <= 0:
                continue
            px = panel.sx(math.log10(value))
            canvas.add(
                f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{panel.top:.1f}" '
                f'y2="{panel.bottom:.1f}" stroke="{color}" stroke-width="1.6" '
                'stroke-dasharray="4 3"/>'
            )
            entries.append((f"{label} {_fmt(value)}", color, "dash"))
        panel.legend(entries, x=panel.right, y=panel_h - 2)
    return canvas.render("Input and output sequence length histograms, log scale")


def _percentile_stack(
    requests: Sequence[Request],
    specs: Sequence[Tuple[str, str, Callable[[Request], Optional[float]], float, str]],
    description: str,
) -> Optional[str]:
    """Per-request values over time: faint dots, rolling P75/P90, cumulative P90.

    ``q90``/``q75`` per spec pick the tail: latency uses the high percentiles
    directly, while a rate (tokens/s/user) is worst at the *low* end, so its
    "P90" is the 10th percentile -- the speed 90% of requests beat.
    """
    profiling = sorted(
        (r for r in requests if r.phase == "profiling" and r.ok), key=lambda r: r.end
    )
    if not profiling:
        return None
    tmax = _time_limit(requests)
    canvas = _Canvas()
    # A lone panel gets more height so the plot is not a thin strip.
    panel_h = 330 if len(specs) == 1 else 290
    canvas.height = _stack_height(len(specs), panel_h)
    for index, (title, ylabel, getter, sign, color) in enumerate(specs):
        points = [(r.end, v) for r in profiling if (v := getter(r)) is not None]
        if not points:
            continue
        q90, q75 = (0.9, 0.75) if sign > 0 else (0.1, 0.25)
        window = _rolling_window(len(points))
        roll90 = _rolling_percentile(points, q90, window)
        roll75 = _rolling_percentile(points, q75, window)
        cum90 = _cumulative_percentile(points, q90)
        top = max(v for _, v in roll90 + roll75 + cum90) or 1.0
        ymax = _ticks(0, top * 1.08, 4)[-1]
        panel = _Panel(
            canvas,
            0,
            4 + index * panel_h,
            _WIDTH,
            panel_h,
            title=title,
            xlim=(0, tmax),
            ylim=(0, ymax),
            xlabel="time since profiling start" if index == len(specs) - 1 else "",
            ylabel=ylabel,
            xticks=_time_ticks(0, tmax),
            # Title and legend each get a row so a long title cannot run
            # under the legend.
            title_h=44,
        )
        dots = [p for p in _subsample(points) if p[1] <= ymax]
        panel.scatter(
            dots,
            color,
            [f"t={_duration(x)}  {_fmt(v)}" for x, v in dots],
            radius=1.6,
            opacity=0.35,
        )
        # Cumulative underneath: where the lines meet, the rolling ones stay visible.
        panel.line(_subsample(cum90, _MAX_LINE_POINTS), "var(--bad)", 2.0)
        panel.line(_subsample(roll75, _MAX_LINE_POINTS), color, 1.6, dash="5 3")
        panel.line(_subsample(roll90, _MAX_LINE_POINTS), color, 2.0)
        panel.legend(
            [
                (f"P90 (rolling {window} req)", color, "line"),
                (f"P75 (rolling {window} req)", color, "dash"),
                ("Cumulative P90", "var(--bad)", "line"),
            ],
            y=4 + index * panel_h + 36,
        )
    return canvas.render(description)


def _chart_latency(requests, _slices) -> Optional[str]:
    return _percentile_stack(
        requests,
        [
            ("TTFT over time", "TTFT (s)", lambda r: r.ttft, 1.0, _SERIES[1]),
            (
                "End-to-end latency over time",
                "E2EL (s)",
                lambda r: r.e2el,
                1.0,
                _SERIES[0],
            ),
        ],
        "TTFT and end-to-end latency over the run: rolling and cumulative percentiles",
    )


def _chart_interactivity(requests, _slices) -> Optional[str]:
    return _percentile_stack(
        requests,
        [
            (
                "Interactivity over time",
                "Interactivity (tok/s/user)",
                lambda r: r.tps_user,
                -1.0,
                _SERIES[2],
            )
        ],
        "Per-request interactivity over the run: rolling and cumulative percentiles",
    )


def _line_stack(
    specs: Sequence[Tuple],
    tmax: float,
    description: str,
) -> Optional[str]:
    """Stacked line panels. A spec is ``(title, ylabel, points, color[, extra])``.

    ``extra`` is a dict: ``lines`` ``[(label, points, color, dash)]`` drawn over
    the main line (with a legend), ``main_label`` naming the main line, and
    ``ymax`` fixing the y range.
    """
    specs = [s for s in specs if s[2]]
    if not specs:
        return None
    canvas = _Canvas()
    panel_h = 190
    canvas.height = _stack_height(len(specs), panel_h)
    for index, spec in enumerate(specs):
        title, ylabel, points, color = spec[:4]
        extra = spec[4] if len(spec) > 4 else {}
        lines = extra.get("lines", [])
        top = (
            max(v for _, v in points + [p for _, pts, _, _ in lines for p in pts])
            or 1.0
        )
        ymax = extra.get("ymax") or _ticks(0, top, 4)[-1]
        y0 = 4 + index * panel_h
        panel = _Panel(
            canvas,
            0,
            y0,
            _WIDTH,
            panel_h,
            title=title,
            xlim=(0, tmax),
            ylim=(0, ymax),
            xlabel="time since profiling start" if index == len(specs) - 1 else "",
            ylabel=ylabel,
            xticks=_time_ticks(0, tmax),
            title_h=38 if lines else _PANEL_TITLE_H,
        )
        panel.line(
            _subsample(points, _MAX_LINE_POINTS),
            color,
            1.6,
            dash=extra.get("main_dash", ""),
        )
        for _label, pts, line_color, dash in lines:
            panel.line(_subsample(pts, _MAX_LINE_POINTS), line_color, 2.0, dash=dash)
        if lines:
            panel.legend(
                [
                    (
                        extra.get("main_label", ylabel),
                        color,
                        "dash" if extra.get("main_dash") else "line",
                    )
                ]
                + [(label, c, "dash" if d else "line") for label, _, c, d in lines],
                y=y0 + 32,
            )
    return canvas.render(description)


def _chart_throughput(requests, slices) -> Optional[str]:
    if not slices:
        return None
    return _line_stack(
        [
            (
                "Prefill throughput",
                "Tokens/s",
                slices.get("Effective Prefill Throughput", []),
                _SERIES[0],
            ),
            (
                "Decode throughput",
                "Tokens/s",
                slices.get("Effective Decode Throughput", []),
                _SERIES[1],
            ),
        ],
        _time_limit(requests),
        "Prefill and decode throughput over the run",
    )


def _chart_load(requests, slices) -> Optional[str]:
    completed = sorted(r.end for r in requests if r.phase == "profiling" and r.ok)
    cumulative = [(t, float(i + 1)) for i, t in enumerate(completed)]
    slices = slices or {}
    rolling_hit_rate = slices.get(HIT_RATE_ROLLING, [])
    if rolling_hit_rate:
        # What the cache is doing beats a raw footprint: it says whether the
        # engine kept the reuse the traces offered.
        cache_panel = (
            "Prefix cache hit rate",
            "Hit rate (%)",
            rolling_hit_rate,
            _SERIES[1],
            {
                "main_label": f"Rolling {_HIT_RATE_WINDOW_S} s",
                "lines": [
                    (
                        "Cumulative",
                        slices.get(HIT_RATE_CUMULATIVE, []),
                        "var(--bad)",
                        "",
                    )
                ],
                "ymax": 100,
            },
        )
        description = (
            "Concurrency, prefix cache hit rate and completed requests over the run"
        )
    else:
        # No engine metrics for this run: fall back to the client's own view of
        # how many tokens were in flight (the KV footprint).
        cache_panel = (
            "Tokens in flight",
            "Tokens (KV footprint)",
            slices.get("Tokens In Flight", []),
            _SERIES[1],
        )
        description = (
            "Concurrency, tokens in flight and completed requests over the run"
        )
    return _line_stack(
        [
            (
                "Effective concurrency",
                "Requests in flight",
                slices.get("Effective Concurrency", []),
                _SERIES[0],
            ),
            cache_panel,
            ("Completed requests (cumulative)", "Requests", cumulative, _SERIES[2]),
        ],
        _time_limit(requests),
        description,
    )


# --------------------------------------------------------------------------
# Mermaid twins
# --------------------------------------------------------------------------
#
# A GitHub job summary strips every image it cannot fetch (artifact paths,
# data: URIs, inline <svg>), but renders ```mermaid blocks. Each chart is
# therefore also written as a markdown fragment of ``xychart-beta`` blocks
# (``<name>.mmd`` beside ``<name>.svg``) that the CI summary swaps in for the
# image. xychart has no legend, dashes or log axis, so series are named in a
# caption by color and the histograms use log-spaced bins as categories.

_MERMAID_POINTS = 60
_MM_COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "light orange": "#f5b38f",
    "green": "#1baf7a",
    "light green": "#8fd9bd",
    "light blue": "#93bdec",
    "violet": "#7c6be0",
    "gray": "#8a8984",
}


def _mm_num(value: float) -> str:
    if abs(value) >= 100 or float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.2f}"


def _mm_grid(points: Sequence[Tuple[float, float]], tmax: float) -> List[float]:
    """Resample to ``_MERMAID_POINTS`` even steps (xychart spaces data evenly).

    Each step is the mean of the points inside it, so a spiky 1 s series is
    summarised rather than sampled; a step with none carries the last value.
    """
    ordered = sorted(points)
    out: List[float] = []
    j = 0
    last = ordered[0][1] if ordered else 0.0
    for i in range(_MERMAID_POINTS):
        t = tmax * (i + 1) / _MERMAID_POINTS
        bucket = []
        while j < len(ordered) and ordered[j][0] <= t:
            bucket.append(ordered[j][1])
            j += 1
        if bucket:
            last = sum(bucket) / len(bucket)
        out.append(last)
    return out


def _mm_block(
    title: str,
    x_axis: str,
    y_label: str,
    y_max: float,
    series: Sequence[Tuple[str, str, str, Sequence[float]]],
    note: str = "",
) -> str:
    """One xychart. ``series`` is ``(label, color name, "line"|"bar", values)``."""
    palette = ", ".join(_MM_COLORS[color] for _, color, _, _ in series)
    lines = [
        "```mermaid",
        '%%{init: {"xyChart": {"width": 900, "height": 300}, '
        f'"themeVariables": {{"xyChart": {{"plotColorPalette": "{palette}"}}}}}}}}%%',
        "xychart-beta",
        f'    title "{title}"',
        f"    x-axis {x_axis}",
        f'    y-axis "{y_label}" 0 --> {_mm_num(y_max)}',
    ]
    for _, _, kind, values in series:
        lines.append(f"    {kind} [{', '.join(_mm_num(v) for v in values)}]")
    lines.append("```")
    caption = " · ".join(f"{color}: {label}" for label, color, _, _ in series if label)
    tail = " — ".join(part for part in (caption, note) if part)
    return "\n".join(lines) + (f"\n\n_{tail}_" if tail else "")


def _mm_time_axis(tmax: float) -> str:
    return f'"minutes since profiling start" 0 --> {max(tmax / 60, 1):.0f}'


def _mm_isl_osl(requests, _slices) -> Optional[str]:
    profiling = [r for r in requests if r.phase == "profiling" and r.ok]
    blocks = []
    for name, values, color in (
        ("Input sequence length", [r.isl for r in profiling], "blue"),
        ("Output sequence length", [r.osl for r in profiling], "orange"),
    ):
        data = sorted(v for v in values if v is not None and v > 0)
        if not data:
            continue
        lo, hi = math.log10(data[0]), math.log10(data[-1])
        if hi - lo < 1e-9:
            hi = lo + 1.0
        bins = 20
        width = (hi - lo) / bins
        counts = [0] * bins
        for v in data:
            counts[min(int((math.log10(v) - lo) / width), bins - 1)] += 1
        labels = ", ".join(f'"{_fmt(10 ** (lo + i * width))}"' for i in range(bins))
        pcts = ", ".join(
            f"p{int(q * 100)} {_fmt(_percentile(data, q))}"
            for q in (0.5, 0.75, 0.9, 0.95)
        )
        blocks.append(
            _mm_block(
                f"{name} (tokens, log bins)",
                f"[{labels}]",
                "requests",
                _ticks(0, max(counts), 4)[-1],
                [("", color, "bar", counts)],
                note=f"{len(data):,} requests · {pcts}",
            )
        )
    return "\n\n".join(blocks) or None


def _mm_percentiles(requests, specs) -> Optional[str]:
    profiling = sorted(
        (r for r in requests if r.phase == "profiling" and r.ok), key=lambda r: r.end
    )
    if not profiling:
        return None
    tmax = _time_limit(requests)
    blocks = []
    for title, y_label, getter, sign, (strong, light) in specs:
        points = [(r.end, v) for r in profiling if (v := getter(r)) is not None]
        if not points:
            continue
        q90, q75 = (0.9, 0.75) if sign > 0 else (0.1, 0.25)
        window = _rolling_window(len(points))
        roll90 = _rolling_percentile(points, q90, window)
        roll75 = _rolling_percentile(points, q75, window)
        cum90 = _cumulative_percentile(points, q90)
        top = max(v for _, v in roll90 + roll75 + cum90) or 1.0
        blocks.append(
            _mm_block(
                title,
                _mm_time_axis(tmax),
                y_label,
                _ticks(0, top * 1.08, 4)[-1],
                # Drawn in order, so cumulative goes underneath.
                [
                    ("cumulative P90", "violet", "line", _mm_grid(cum90, tmax)),
                    (
                        f"P75 (rolling {window} req)",
                        light,
                        "line",
                        _mm_grid(roll75, tmax),
                    ),
                    (
                        f"P90 (rolling {window} req)",
                        strong,
                        "line",
                        _mm_grid(roll90, tmax),
                    ),
                ],
            )
        )
    return "\n\n".join(blocks) or None


def _mm_latency(requests, _slices) -> Optional[str]:
    return _mm_percentiles(
        requests,
        [
            (
                "TTFT over time",
                "TTFT (s)",
                lambda r: r.ttft,
                1.0,
                ("orange", "light orange"),
            ),
            (
                "End-to-end latency over time",
                "E2EL (s)",
                lambda r: r.e2el,
                1.0,
                ("blue", "light blue"),
            ),
        ],
    )


def _mm_interactivity(requests, _slices) -> Optional[str]:
    return _mm_percentiles(
        requests,
        [
            (
                "Interactivity over time",
                "tok/s/user",
                lambda r: r.tps_user,
                -1.0,
                ("green", "light green"),
            )
        ],
    )


def _mm_lines(
    tmax: float,
    specs: Sequence[
        Tuple[
            str,
            str,
            Sequence[Tuple[str, str, Sequence[Tuple[float, float]]]],
            Optional[float],
        ]
    ],
) -> Optional[str]:
    blocks = []
    for title, y_label, series, y_max in specs:
        series = [(label, color, pts) for label, color, pts in series if pts]
        if not series:
            continue
        top = (
            y_max
            or _ticks(0, max(v for _, _, pts in series for _, v in pts) or 1.0, 4)[-1]
        )
        blocks.append(
            _mm_block(
                title,
                _mm_time_axis(tmax),
                y_label,
                top,
                [
                    (label, color, "line", _mm_grid(pts, tmax))
                    for label, color, pts in series
                ],
            )
        )
    return "\n\n".join(blocks) or None


def _mm_throughput(requests, slices) -> Optional[str]:
    slices = slices or {}
    return _mm_lines(
        _time_limit(requests),
        [
            (
                "Prefill throughput",
                "tokens/s",
                [("", "blue", slices.get("Effective Prefill Throughput", []))],
                None,
            ),
            (
                "Decode throughput",
                "tokens/s",
                [("", "orange", slices.get("Effective Decode Throughput", []))],
                None,
            ),
        ],
    )


def _mm_load(requests, slices) -> Optional[str]:
    slices = slices or {}
    completed = sorted(r.end for r in requests if r.phase == "profiling" and r.ok)
    if slices.get(HIT_RATE_ROLLING):
        cache = (
            "Prefix cache hit rate",
            "hit rate (%)",
            [
                (f"rolling {_HIT_RATE_WINDOW_S} s", "orange", slices[HIT_RATE_ROLLING]),
                ("cumulative", "violet", slices.get(HIT_RATE_CUMULATIVE, [])),
            ],
            100.0,
        )
    else:
        cache = (
            "Tokens in flight",
            "tokens",
            [("", "orange", slices.get("Tokens In Flight", []))],
            None,
        )
    return _mm_lines(
        _time_limit(requests),
        [
            (
                "Effective concurrency",
                "requests in flight",
                [("", "blue", slices.get("Effective Concurrency", []))],
                None,
            ),
            cache,
            (
                "Completed requests (cumulative)",
                "requests",
                [("", "green", [(t, float(i + 1)) for i, t in enumerate(completed)])],
                None,
            ),
        ],
    )


_MERMAID: Dict[str, Callable] = {
    "isl_osl": _mm_isl_osl,
    "latency": _mm_latency,
    "interactivity": _mm_interactivity,
    "throughput": _mm_throughput,
    "load": _mm_load,
}


_CHARTS: Dict[str, Callable] = {
    "isl_osl": _chart_isl_osl,
    "latency": _chart_latency,
    "interactivity": _chart_interactivity,
    "throughput": _chart_throughput,
    "load": _chart_load,
}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def write_point_charts(artifact_dir: Path, out_dir: Path) -> Dict[str, str]:
    """Write the per-point charts and return ``{chart name: file name}``.

    Returns an empty mapping when the run left no per-request export. A chart
    that cannot be drawn is skipped, never fatal: the numbers in the report
    tables do not depend on these.
    """
    export = artifact_dir / PROFILE_EXPORT
    if not export.exists():
        logger.warning(
            "No %s under %s; skipping telemetry charts.", PROFILE_EXPORT, artifact_dir
        )
        return {}
    requests, origin_ns = load_requests(export)
    if not requests:
        return {}

    slices: Dict[str, List[Tuple[float, float]]] = {}
    slices_path = artifact_dir / TIMESLICES
    if slices_path.exists():
        try:
            slices = load_timeslices(slices_path, origin_ns, _SLICE_METRICS)
        except Exception:
            logger.exception("Could not read %s", slices_path)
    metrics_path = artifact_dir / SERVER_METRICS
    if metrics_path.exists():
        try:
            slices.update(load_prefix_cache_hit_rate(metrics_path, origin_ns))
        except Exception:
            logger.exception("Could not read %s", metrics_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for name, build in _CHARTS.items():
        try:
            svg = build(requests, slices)
        except Exception:
            logger.exception("Telemetry chart %r failed for %s", name, artifact_dir)
            continue
        if not svg:
            continue
        filename = f"{name}.svg"
        (out_dir / filename).write_text(svg, encoding="utf-8")
        written[name] = filename
        # The CI job summary's stand-in for this image; optional.
        try:
            fragment = _MERMAID[name](requests, slices)
        except Exception:
            logger.exception("Mermaid twin of %r failed for %s", name, artifact_dir)
            fragment = None
        if fragment:
            (out_dir / f"{name}{MERMAID_SUFFIX}").write_text(
                fragment + "\n", encoding="utf-8"
            )
    return written
