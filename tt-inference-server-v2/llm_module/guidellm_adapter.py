# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC

"""Convert a raw guidellm benchmark dump into one ``kind="guidellm"`` record.

Performs every aggregation, percentile lookup, and derived calculation
the report tables need so the report module just renders. No math
inside the renderer / generator / schema.

The output is a single record ready to drop into the unified
``[{kind, model, device, timestamp, ...}]`` schema alongside other kinds
like ``evals``, ``benchmarks`` or ``server_tests``. Section keys are
display-friendly strings (``"Run Configuration"``, ``"Request Totals"``,
``"TTFT vs. Context (Linear Regression)"``, ...); each value is a flat
dict, list of dicts, or dict-of-dicts that ``render_generic_table`` in
:mod:`report_module.renderers` emits as its own H4 sub-table beneath
the kind heading.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SUMMARY_METRICS: Tuple[str, ...] = (
    "time_to_first_token_ms",
    "inter_token_latency_ms",
    "time_per_output_token_ms",
    "request_latency",
    "output_tokens_per_second",
    "tokens_per_second",
    "requests_per_second",
    "request_concurrency",
    "prompt_token_count",
    "output_token_count",
    "total_token_count",
)

PERCENTILE_METRICS: Tuple[str, ...] = (
    "time_to_first_token_ms",
    "inter_token_latency_ms",
    "time_per_output_token_ms",
    "request_latency",
)

SLO = {
    "ttft_p95_ms": 250,
    "ttft_p99_ms": 400,
    "itl_p95_ms": 5,
    "tpot_p95_ms": 6,
    "error_rate": 0.01,
    "out_tps_mean": 200,
}


def to_report_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a single ``kind="guidellm"`` record from a raw guidellm dump.

    The returned record carries the universal record fields (``kind``,
    ``model``, ``device``, ``timestamp``) plus one nested entry per
    table the guidellm renderer emits.
    """
    md = raw.get("metadata", {})
    benchmark = raw["benchmarks"][0]
    config = benchmark["config"]
    metrics = benchmark["metrics"]
    scheduler = benchmark["scheduler_state"]
    successful = benchmark["requests"]["successful"]

    backend = config.get("backend", {}) or {}

    return {
        "kind": "guidellm",
        "model": backend.get("model", ""),
        "device": "",
        "timestamp": _epoch_to_timestamp(benchmark.get("end_time", 0)),
        "Run Configuration": _run_header(md, benchmark, config),
        "Request Totals": _request_totals(metrics, scheduler),
        "Stopping Conditions": _stop_conditions(scheduler),
        "Summary Statistics": _summary_stats(metrics),
        "Latency Percentiles (ms)": _latency_percentiles(metrics),
        "Tail Ratios": _tail_ratios(metrics),
        "Token Accounting Sanity Check": _token_sanity(metrics, successful),
        "Per-Turn Breakdown": _per_turn(successful),
        "Cold vs. Warm TTFT": _cold_vs_warm(successful),
        "TTFT vs. Context (Linear Regression)": _ttft_vs_context(successful),
        "ITL / TPOT Stability": _stability(successful),
        "Per-Request Latency Breakdown (ms)": _latency_breakdown(successful),
        "Server vs. Harness Time": _server_vs_harness(successful, scheduler),
        "Key Takeaways": _key_takeaways(metrics),
        "Top 3 Latency Outliers": _top_outliers(successful),
        "Errors / Incomplete": _errors_summary(benchmark),
        "Time Accounting": _time_accounting(benchmark, scheduler, metrics),
        "Workload Shape Verification": _shape_verification(config, successful),
        "SLO Checks": _slo_checks(metrics),
    }


def _epoch_to_timestamp(epoch: float) -> str:
    if not epoch:
        return ""
    return dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def _run_header(
    md: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    backend = config.get("backend", {}) or {}
    strategy = config.get("strategy", {}) or {}
    requests_cfg = config.get("requests", {}) or {}
    target = backend.get("target", "")
    http2 = str(backend.get("http2", "")).lower()
    verify = str(backend.get("verify", "")).lower()
    workers = strategy.get("worker_count", "")
    streams = strategy.get("streams", "")
    return {
        "model": backend.get("model", ""),
        "backend": f"{target}  (http2={http2}, verify={verify})",
        "strategy": (
            f"{strategy.get('type_', '')} @ "
            f"max_concurrency={strategy.get('max_concurrency', '')}   "
            f"({workers} worker, {streams} stream)"
        ),
        "shape": _strip_outer_brackets(str(requests_cfg.get("data", ""))),
        "guidellm": md.get("guidellm_version", ""),
        "python": md.get("python_version", ""),
        "platform": md.get("platform", ""),
        "wall_time_s": _r(
            benchmark.get("end_time", 0) - benchmark.get("start_time", 0), 3
        ),
        "warmup_s": benchmark.get("warmup_duration", 0),
        "cooldown_s": benchmark.get("cooldown_duration", 0),
    }


def _request_totals(
    metrics: Mapping[str, Any], scheduler: Mapping[str, Any]
) -> Dict[str, Any]:
    rt = metrics.get("request_totals", {}) or {}
    total = rt.get("total", 0) or 0
    successful = rt.get("successful", 0) or 0
    errored = rt.get("errored", 0) or 0
    incomplete = rt.get("incomplete", 0) or 0
    cancelled = scheduler.get("cancelled_requests", 0) or 0
    success_pct = (100.0 * successful / total) if total else 0.0
    error_rate = (errored / total) if total else 0.0
    return {
        "total": total,
        "successful": f"{successful}   ({success_pct:.1f}%)",
        "errored": errored,
        "incomplete": incomplete,
        "cancelled": cancelled,
        "error_rate": _r(error_rate, 4),
    }


def _stop_conditions(scheduler: Mapping[str, Any]) -> Dict[str, Any]:
    constraints = scheduler.get("end_processing_constraints", {}) or {}
    if not constraints:
        return {"stopping_constraint": "n/a"}
    name, c = next(iter(constraints.items()))
    meta = c.get("metadata", {}) or {}
    return {
        "stopping_constraint": name,
        "max_number": meta.get("max_number"),
        "created_requests": meta.get("created_requests"),
        "processed_requests": meta.get("processed_requests"),
        "remaining_requests": meta.get("remaining_requests"),
        "stop_time_epoch": meta.get("stop_time"),
        "queuing_action": c.get("request_queuing"),
        "processing_action": c.get("request_processing"),
        "conclusion": (
            "ran to completion (not a timeout/kill)"
            if meta.get("processed_exceeded")
            else "stopped early"
        ),
    }


def _summary_stats(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in SUMMARY_METRICS:
        s = (metrics.get(name) or {}).get("successful") or {}
        rows.append(
            {
                "metric": name,
                "mean": _r(s.get("mean")),
                "std": _r(s.get("std_dev"), 4),
                "min": _r(s.get("min")),
                "max": _r(s.get("max")),
                "n": s.get("count"),
            }
        )
    return rows


def _latency_percentiles(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in PERCENTILE_METRICS:
        p = ((metrics.get(name) or {}).get("successful") or {}).get("percentiles") or {}
        rows.append(
            {
                "metric": name,
                "p50": _r(p.get("p50"), 4),
                "p90": _r(p.get("p90"), 4),
                "p95": _r(p.get("p95"), 4),
                "p99": _r(p.get("p99"), 4),
                "p999": _r(p.get("p999"), 4),
            }
        )
    return rows


def _tail_ratios(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in PERCENTILE_METRICS:
        s = (metrics.get(name) or {}).get("successful") or {}
        p = s.get("percentiles") or {}
        p50 = p.get("p50") or 0
        if not p50:
            rows.append(
                {
                    "metric": name,
                    "p95/p50": "N/A",
                    "p99/p50": "N/A",
                    "p99/p95": "N/A",
                    "max/p50": "N/A",
                }
            )
            continue
        rows.append(
            {
                "metric": name,
                "p95/p50": _r(p["p95"] / p50, 4),
                "p99/p50": _r(p["p99"] / p50, 4),
                "p99/p95": _r(p["p99"] / p["p95"], 5) if p.get("p95") else "N/A",
                "max/p50": _r(s.get("max", 0) / p50, 4),
            }
        )
    return rows


def _token_sanity(
    metrics: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    sum_prompt = sum(int(r.get("prompt_tokens", 0)) for r in requests)
    sum_output = sum(int(r.get("output_tokens", 0)) for r in requests)
    agg_prompt = (
        (metrics.get("prompt_token_count") or {}).get("successful") or {}
    ).get("total_sum", 0)
    agg_output = (
        (metrics.get("output_token_count") or {}).get("successful") or {}
    ).get("total_sum", 0)
    all_match = all(
        r.get("prompt_tokens", 0) + r.get("output_tokens", 0)
        == r.get("total_tokens", 0)
        for r in requests
    )
    return {
        "sum_prompt_tokens_per_request": sum_prompt,
        "aggregate_prompt_token_count": (
            f"{agg_prompt}    " + ("MATCH" if sum_prompt == agg_prompt else "MISMATCH")
        ),
        "sum_output_tokens_per_request": sum_output,
        "aggregate_output_token_count": (
            f"{agg_output}    " + ("MATCH" if sum_output == agg_output else "MISMATCH")
        ),
        "all_prompt_plus_output_eq_total": all_match,
    }


def _per_turn(requests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    sorted_reqs = sorted(
        requests,
        key=lambda r: (
            r["info"].get("conversation_id", ""),
            r["info"].get("turn_index", 0),
        ),
    )
    rows: List[Dict[str, Any]] = []
    for r in sorted_reqs:
        rows.append(
            {
                "turn": r["info"].get("turn_index"),
                "ptok": r.get("prompt_tokens"),
                "otok": r.get("output_tokens"),
                "ttft_ms": _r(r.get("time_to_first_token_ms"), 3),
                "itl_ms": _r(r.get("inter_token_latency_ms"), 4),
                "tpot_ms": _r(r.get("time_per_output_token_ms"), 4),
                "lat_s": _r(r.get("request_latency"), 4),
                "out_tps": _r(r.get("output_tokens_per_second"), 2),
            }
        )
    return rows


def _cold_vs_warm(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cold = [
        r["time_to_first_token_ms"]
        for r in requests
        if r["info"].get("turn_index") == 0
    ]
    warm = [
        r["time_to_first_token_ms"]
        for r in requests
        if r["info"].get("turn_index", 0) > 0
    ]
    cold_mean = statistics.mean(cold) if cold else 0.0
    warm_mean = statistics.mean(warm) if warm else 0.0
    warm_stdev = statistics.stdev(warm) if len(warm) > 1 else 0.0
    delta = cold_mean - warm_mean
    speedup = (cold_mean / warm_mean) if warm_mean else 0.0
    cold_ptok = next(
        (r["prompt_tokens"] for r in requests if r["info"].get("turn_index") == 0), 0
    )
    warm_ptok = next(
        (r["prompt_tokens"] for r in requests if r["info"].get("turn_index", 0) > 0), 0
    )
    ratio = (warm_ptok / cold_ptok) if cold_ptok else 0.0
    n_warm = len(warm)
    return {
        "turn_0_cold": f"n={len(cold)}   mean_ttft = {cold_mean:.2f} ms",
        f"turns_1_to_{n_warm}_warm": (
            f"n={n_warm}   mean_ttft = {warm_mean:.2f} ms   stdev = {warm_stdev:.2f} ms"
        ),
        "delta_cold_minus_warm": f"{delta:+.2f} ms",
        "cold_warm_speedup": f"{speedup:.3f}x",
        "warm_cold_prompt_ratio": f"{warm_ptok} / {cold_ptok} = {ratio:.3f}",
    }


def _ttft_vs_context(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sorted_reqs = sorted(
        requests,
        key=lambda r: (
            r["info"].get("conversation_id", ""),
            r["info"].get("turn_index", 0),
        ),
    )
    xs: List[float] = []
    ys: List[float] = []
    ctx = 0.0
    for r in sorted_reqs:
        ctx += float(r.get("prompt_tokens", 0)) + float(r.get("output_tokens", 0))
        if r["info"].get("turn_index", 0) > 0:
            xs.append(ctx / 1000.0)
            ys.append(float(r["time_to_first_token_ms"]))
    if len(xs) < 2:
        return {"linreg": "insufficient data (need >=2 warm turns)", "r_squared": "N/A"}
    slope, intercept, r2 = _linear_regression(xs, ys)
    return {
        "linreg": f"TTFT_ms = {slope:.4f} * ctx_kTok + {intercept:.2f}",
        "r_squared": _r(r2, 4),
        "ctx_range_ktok": f"{min(xs):.1f} -> {max(xs):.1f} (ctx grows across warm turns)",
        "ttft_range_ms": f"{min(ys):.1f} -> {max(ys):.1f}",
    }


def _stability(requests: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    itl = [
        float(r["inter_token_latency_ms"])
        for r in requests
        if "inter_token_latency_ms" in r
    ]
    tpot = [
        float(r["time_per_output_token_ms"])
        for r in requests
        if "time_per_output_token_ms" in r
    ]
    return {
        "ITL_mean_ms": _r(statistics.mean(itl), 6) if itl else None,
        "ITL_stdev_ms": _r(statistics.stdev(itl), 6) if len(itl) > 1 else 0.0,
        "ITL_CV_pct": (
            _r(100 * statistics.stdev(itl) / statistics.mean(itl), 4)
            if len(itl) > 1 and statistics.mean(itl)
            else 0.0
        ),
        "TPOT_mean_ms": _r(statistics.mean(tpot), 4) if tpot else None,
        "TPOT_stdev_ms": _r(statistics.stdev(tpot), 4) if len(tpot) > 1 else 0.0,
        "TPOT_CV_pct": (
            _r(100 * statistics.stdev(tpot) / statistics.mean(tpot), 3)
            if len(tpot) > 1 and statistics.mean(tpot)
            else 0.0
        ),
    }


def _latency_breakdown(requests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    segments: Dict[str, List[float]] = {
        "queue_wait": [],
        "sched_overhead": [],
        "prefill (TTFT)": [],
        "decode": [],
        "finalize": [],
        "total (queued -> finalized)": [],
    }
    for r in requests:
        t = r["info"].get("timings", {}) or {}
        segments["queue_wait"].append((t["dequeued"] - t["queued"]) * 1000)
        segments["sched_overhead"].append((t["request_start"] - t["dequeued"]) * 1000)
        segments["prefill (TTFT)"].append(
            (t["first_token_iteration"] - t["request_start"]) * 1000
        )
        segments["decode"].append(
            (t["last_token_iteration"] - t["first_token_iteration"]) * 1000
        )
        segments["finalize"].append((t["finalized"] - t["request_end"]) * 1000)
        segments["total (queued -> finalized)"].append(
            (t["finalized"] - t["queued"]) * 1000
        )

    rows: List[Dict[str, Any]] = []
    for name, vals in segments.items():
        if not vals:
            continue
        rows.append(
            {
                "segment": name,
                "mean_ms": _r(statistics.mean(vals), 3),
                "p50_ms": _r(statistics.median(vals), 3),
                "p95_ms": _r(_quantile(vals, 0.95), 3),
                "min_ms": _r(min(vals), 3),
                "max_ms": _r(max(vals), 3),
            }
        )
    return rows


def _server_vs_harness(
    requests: Sequence[Mapping[str, Any]], scheduler: Mapping[str, Any]
) -> Dict[str, Any]:
    sum_lat = sum(float(r.get("request_latency", 0)) for r in requests)
    active = scheduler.get("end_requests_time", 0) - scheduler.get(
        "start_requests_time", 0
    )
    server_frac = (sum_lat / active) if active else 0.0
    return {
        "sum_request_latency_s": _r(sum_lat, 3),
        "active_window_s": _r(active, 3),
        "server_fraction_pct": _r(100 * server_frac, 1),
        "harness_fraction_pct": _r(100 * (1 - server_frac), 1),
    }


def _key_takeaways(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    def _med(name: str) -> float:
        return ((metrics.get(name) or {}).get("successful") or {}).get(
            "percentiles", {}
        ).get("p50", 0) or 0

    out_tps_mean = (
        (metrics.get("output_tokens_per_second") or {}).get("successful") or {}
    ).get("mean", 0) or 0
    concurrency = (
        (metrics.get("request_concurrency") or {}).get("successful") or {}
    ).get("mean", 0) or 0
    return {
        "TTFT_median_ms": _r(_med("time_to_first_token_ms"), 2),
        "ITL_median_ms": _r(_med("inter_token_latency_ms"), 2),
        "TPOT_median_ms": _r(_med("time_per_output_token_ms"), 2),
        "request_latency_median_ms": _r(_med("request_latency") * 1000, 1),
        "output_tokens_per_second_agg": _r(out_tps_mean, 2),
        "concurrency_observed": _r(concurrency, 3),
        "per_user_out_tps": _r(out_tps_mean / max(concurrency, 1e-9), 1),
    }


def _top_outliers(requests: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    top3 = sorted(requests, key=lambda r: -float(r.get("request_latency", 0)))[:3]
    rows: List[Dict[str, Any]] = []
    for rank, r in enumerate(top3, 1):
        t = r["info"].get("timings", {}) or {}
        prefill_ms = (t["first_token_iteration"] - t["request_start"]) * 1000
        decode_ms = (t["last_token_iteration"] - t["first_token_iteration"]) * 1000
        turn = r["info"].get("turn_index")
        if turn == 0:
            note = f"cold start, full {r.get('prompt_tokens')}-tok prefill"
        else:
            note = "warm-turn outlier (large cumulative ctx)"
        rows.append(
            {
                "rank": rank,
                "turn": turn,
                "ptok": r.get("prompt_tokens"),
                "lat_ms": _r(float(r.get("request_latency", 0)) * 1000, 2),
                "ttft_ms": _r(r.get("time_to_first_token_ms"), 2),
                "prefill_ms": _r(prefill_ms, 2),
                "decode_ms": _r(decode_ms, 2),
                "note": note,
            }
        )
    return rows


def _errors_summary(benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "errored": len((benchmark.get("requests") or {}).get("errored", []) or []),
        "incomplete": len(
            (benchmark.get("requests") or {}).get("incomplete", []) or []
        ),
    }


def _time_accounting(
    benchmark: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    wall = benchmark.get("end_time", 0) - benchmark.get("start_time", 0)
    active = scheduler.get("end_requests_time", 0) - scheduler.get(
        "start_requests_time", 0
    )
    warmup = benchmark.get("warmup_duration", 0) or 0
    cooldown = benchmark.get("cooldown_duration", 0) or 0
    idle = wall - active - warmup - cooldown
    active_frac = (100 * active / wall) if wall else 0.0
    avg_concurrency = (
        (metrics.get("request_concurrency") or {}).get("successful") or {}
    ).get("mean", 0)
    rps_mean = ((metrics.get("requests_per_second") or {}).get("successful") or {}).get(
        "mean", 0
    )
    out_tps_mean = (
        (metrics.get("output_tokens_per_second") or {}).get("successful") or {}
    ).get("mean", 0)
    return {
        "wall_s": _r(wall, 3),
        "active_s": _r(active, 3),
        "idle_s": _r(idle, 3),
        "active_fraction_pct": _r(active_frac, 1),
        "warmup_s": warmup,
        "cooldown_s": cooldown,
        "avg_concurrency": _r(avg_concurrency, 3),
        "observed_req_per_s_agg": _r(rps_mean, 3),
        "observed_out_tok_per_s": _r(out_tps_mean, 2),
    }


def _shape_verification(
    config: Mapping[str, Any], requests: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    advertised = _strip_outer_brackets(
        str((config.get("requests") or {}).get("data", ""))
    )
    ptoks = [int(r.get("prompt_tokens", 0)) for r in requests]
    otoks = [int(r.get("output_tokens", 0)) for r in requests]
    convo_turns = {
        (r["info"].get("conversation_id", ""), r["info"].get("turn_index", 0))
        for r in requests
    }
    convos = {r["info"].get("conversation_id", "") for r in requests}
    max_turn = max((t for _, t in convo_turns), default=0)
    return {
        "advertised": advertised,
        "observed_prompt_tokens": (
            f"min={min(ptoks)}   max={max(ptoks)}   mean={statistics.mean(ptoks):.2f}"
            if ptoks
            else "n/a"
        ),
        "observed_output_tokens": (
            f"min={min(otoks)}   max={max(otoks)}   mean={statistics.mean(otoks):.2f}"
            if otoks
            else "n/a"
        ),
        "observed_turns": (
            f"{len(convo_turns)} (turn_index 0..{max_turn} in {len(convos)} conversation)"
        ),
    }


def _slo_checks(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ttft = (metrics.get("time_to_first_token_ms") or {}).get("successful") or {}
    itl = (metrics.get("inter_token_latency_ms") or {}).get("successful") or {}
    tpot = (metrics.get("time_per_output_token_ms") or {}).get("successful") or {}
    out_tps = (metrics.get("output_tokens_per_second") or {}).get("successful") or {}
    rt = metrics.get("request_totals") or {}

    total = rt.get("total", 0) or 0
    error_rate = (rt.get("errored", 0) / total) if total else 0.0

    ttft_p95 = (ttft.get("percentiles") or {}).get("p95", math.inf)
    ttft_p99 = (ttft.get("percentiles") or {}).get("p99", math.inf)
    itl_p95 = (itl.get("percentiles") or {}).get("p95", math.inf)
    tpot_p95 = (tpot.get("percentiles") or {}).get("p95", math.inf)
    out_tps_mean = out_tps.get("mean", 0) or 0

    checks = [
        ("TTFT p95  <= 250 ms", f"{ttft_p95:.2f} ms", ttft_p95 <= SLO["ttft_p95_ms"]),
        ("TTFT p99  <= 400 ms", f"{ttft_p99:.2f} ms", ttft_p99 <= SLO["ttft_p99_ms"]),
        ("ITL  p95  <=   5 ms", f"{itl_p95:.3f} ms", itl_p95 <= SLO["itl_p95_ms"]),
        ("TPOT p95  <=   6 ms", f"{tpot_p95:.3f} ms", tpot_p95 <= SLO["tpot_p95_ms"]),
        (
            "error_rate <= 1%",
            f"{100 * error_rate:.2f}%",
            error_rate <= SLO["error_rate"],
        ),
        (
            "out_tok/s >= 200",
            f"{out_tps_mean:.2f}",
            out_tps_mean >= SLO["out_tps_mean"],
        ),
    ]

    rows = [
        {
            "slo_target": label,
            "observed": observed,
            "result": "PASS" if passed else "FAIL",
        }
        for label, observed, passed in checks
    ]
    rows.append(
        {
            "slo_target": "overall",
            "observed": "",
            "result": "PASS" if all(passed for *_, passed in checks) else "FAIL",
        }
    )
    return rows


def _linear_regression(
    xs: Sequence[float], ys: Sequence[float]
) -> Tuple[float, float, float]:
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0, mean_y, 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return slope, intercept, r2


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _r(value: Any, digits: int = 2) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


def _strip_outer_brackets(text: str) -> str:
    """Pull the inner string out of values like ``"['foo']"`` so the report
    shows ``foo`` instead of ``['foo']`` for guidellm config dumps."""
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    if (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    ):
        s = s[1:-1]
    return s
