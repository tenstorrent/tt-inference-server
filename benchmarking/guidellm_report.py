# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Encapsulated GuideLLM report generation.

This module owns ALL GuideLLM-specific report logic so that
`workflows/run_reports.py` only has to call a single entry point. When
`run_reports.py` is refactored, the only thing the new orchestrator needs to
do for GuideLLM is call `generate_guidellm_report(args, model_spec, report_id,
metadata)` and consume the returned `(release_str, release_data, disp_md_path,
data_file_path)` tuple — identical contract to the AIPerf / GenAI-Perf /
vLLM helpers it already calls.

Two data sources are used, chosen for what each is best at:

1. The normalized `guidellm_benchmark_<model_id>_*_sweep-K.json` files in
   `benchmarks_output/` — produced by
   `benchmarking/run_guidellm_benchmarks.py::emit_normalized_guidellm_result`.
   These are used to populate the AIPerf-compatible "Detailed Percentiles"
   summary table and to discover which scenarios were executed.
2. The native GuideLLM `benchmarks.json` files (referenced by the
   `source_benchmarks_json` field of every normalized file). These are used
   to render the five GuideLLM-native tables (Run Summary, Text Metrics,
   Request Token Stats, Request Latency Stats, Server Throughput Stats) so
   the report mirrors what GuideLLM prints to the console at the end of a
   run, with one row per benchmark/sweep point.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

ReportTuple = Tuple[str, List[Dict[str, Any]], Optional[Path], Optional[Path]]


def generate_guidellm_report(
    args: Any,
    model_spec: Any,
    report_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    benchmarks_output_dir: Optional[Path] = None,
) -> ReportTuple:
    """Generate the GuideLLM section of a benchmark report.

    Returns the same `(release_str, release_data, disp_md_path, data_file_path)`
    contract as the existing `aiperf_benchmark_generate_report` and
    `genai_perf_benchmark_generate_report` helpers in
    `workflows/run_reports.py`, so it is drop-in compatible with the current
    orchestrator and any future refactor of it.
    """
    metadata = metadata or {}

    if benchmarks_output_dir is None:
        # Default: same directory all benchmark tools write to. Resolved
        # lazily so this module does not import workflows.* eagerly.
        from workflows.utils import get_default_workflow_root_log_dir

        benchmarks_output_dir = Path(
            f"{get_default_workflow_root_log_dir()}/benchmarks_output"
        )

    output_dir = Path(args.output_path) / "benchmarks_guidellm"
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_files = _discover_normalized_files(
        benchmarks_output_dir, model_spec.model_id
    )
    logger.info(
        f"GuideLLM Benchmark Summary — found {len(normalized_files)} "
        f"normalized result file(s) for model_id={model_spec.model_id}"
    )
    if not normalized_files:
        logger.info("No GuideLLM benchmark files found. Skipping GuideLLM report.")
        return "", [], None, None

    # Deduplicate by (isl, osl, maxcon, n[, images, h, w], sweep_index) so
    # GuideLLM sweep points sharing a base config are preserved end-to-end.
    normalized_files = _deduplicate_by_config(normalized_files)
    text_files = [f for f in normalized_files if "_images-" not in Path(f).name]
    vlm_files = [f for f in normalized_files if "_images-" in Path(f).name]

    text_results = sorted(
        (_load_normalized_text(f) for f in text_files),
        key=lambda r: (r["isl"], r["osl"], r["concurrency"]),
    )
    vlm_results = sorted(
        filter(None, (_load_normalized_vlm(f) for f in vlm_files)),
        key=lambda r: (
            r["isl"],
            r["osl"],
            r["concurrency"],
            r["image_height"],
            r["image_width"],
        ),
    )

    if not text_results and not vlm_results:
        return "", [], None, None

    sections: List[str] = [
        f"### Benchmark Performance Results for {model_spec.model_name} on {args.device}\n"
    ]

    # ---- Detailed percentile summary tables (AIPerf-compatible style) ----
    if text_results:
        sections.append("#### GuideLLM Text Benchmarks - Detailed Percentiles\n")
        sections.append(
            "**Benchmarking Tool:** [GuideLLM](https://github.com/vllm-project/guidellm)\n"
        )
        sections.append(_render_percentile_table(text_results, vlm=False))
    if vlm_results:
        sections.append("#### GuideLLM VLM Benchmarks - Detailed Percentiles\n")
        sections.append(
            "**Benchmarking Tool:** [GuideLLM](https://github.com/vllm-project/guidellm)\n"
        )
        sections.append(_render_percentile_table(vlm_results, vlm=True))

    sections.append(_metric_definitions_block())

    # ---- Native GuideLLM-style tables (one block per source benchmarks.json) ----
    native_blocks = _render_native_tables(text_results + vlm_results)
    if native_blocks:
        sections.append("#### GuideLLM Run Details (native tables)\n")
        sections.append(
            "Tables below mirror the summary GuideLLM prints to the console at "
            "the end of each run, with one row per sweep point.\n"
        )
        sections.extend(native_blocks)

    release_str = "\n\n".join(s for s in sections if s).strip() + "\n"

    # ---- Persist markdown and CSV artifacts ----
    disp_md_path = output_dir / f"guidellm_benchmark_display_{report_id}.md"
    disp_md_path.write_text(release_str, encoding="utf-8")
    logger.info(f"GuideLLM report saved to: {disp_md_path}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    text_data_file_path = data_dir / f"guidellm_benchmark_text_stats_{report_id}.csv"
    vlm_data_file_path = data_dir / f"guidellm_benchmark_vlm_stats_{report_id}.csv"

    if text_results:
        _write_csv(text_data_file_path, text_results)
        logger.info(f"GuideLLM text CSV saved to: {text_data_file_path}")
    if vlm_results:
        _write_csv(vlm_data_file_path, vlm_results)
        logger.info(f"GuideLLM VLM CSV saved to: {vlm_data_file_path}")

    return (
        release_str,
        text_results + vlm_results,
        disp_md_path,
        text_data_file_path if text_results else None,
    )


# --------------------------------------------------------------------------- #
# Discovery + per-row loading from the normalized files
# --------------------------------------------------------------------------- #


def _discover_normalized_files(
    benchmarks_output_dir: Path, model_id: str
) -> List[str]:
    pattern = f"guidellm_benchmark_{model_id}_*.json"
    return glob(f"{benchmarks_output_dir}/{pattern}")


def _deduplicate_by_config(files: List[str]) -> List[str]:
    """Latest file wins for each unique config; sweep_index is part of the key
    so GuideLLM sweep points sharing (isl, osl, maxcon, n) are preserved."""
    config_to_file: Dict[Tuple[Any, ...], str] = {}
    for filepath in sorted(files, reverse=True):
        filename = Path(filepath).name
        m = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
        if not m:
            config_to_file[(filepath,)] = filepath
            continue
        isl, osl, con, n = map(int, m.groups())
        img_match = re.search(r"images-(\d+)_height-(\d+)_width-(\d+)", filename)
        sweep_match = re.search(r"_sweep-(\d+)", filename)
        sweep_idx = int(sweep_match.group(1)) if sweep_match else None
        if img_match:
            images, height, width = map(int, img_match.groups())
            key = (isl, osl, con, n, images, height, width, sweep_idx)
        else:
            key = (isl, osl, con, n, 0, 0, 0, sweep_idx)
        if key not in config_to_file:
            config_to_file[key] = filepath
    return list(config_to_file.values())


def _load_normalized_text(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    filename = Path(filepath).name
    m = re.search(r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)", filename)
    if m:
        isl, osl, concurrency, num_requests = map(int, m.groups())
    else:
        isl = data.get("total_input_tokens", 0) // max(data.get("num_prompts", 1), 1)
        osl = data.get("total_output_tokens", 0) // max(data.get("num_prompts", 1), 1)
        concurrency = data.get("max_concurrency", 1)
        num_requests = data.get("num_prompts", 0)

    sweep_match = re.search(r"_sweep-(\d+)", filename)
    return {
        "source": "guidellm",
        "scenario": data.get("scenario", ""),
        "sweep_index": int(sweep_match.group(1)) if sweep_match else None,
        "strategy_type": data.get("strategy_type"),
        "strategy_rate": data.get("strategy_rate"),
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "num_requests": num_requests,
        # Latency
        "mean_ttft_ms": data.get("mean_ttft_ms", 0),
        "median_ttft_ms": data.get("median_ttft_ms", 0),
        "p99_ttft_ms": data.get("p99_ttft_ms", 0),
        "std_ttft_ms": data.get("std_ttft_ms", 0),
        "mean_tpot_ms": data.get("mean_tpot_ms", 0),
        "median_tpot_ms": data.get("median_tpot_ms", 0),
        "p99_tpot_ms": data.get("p99_tpot_ms", 0),
        "std_tpot_ms": data.get("std_tpot_ms", 0),
        "mean_e2el_ms": data.get("mean_e2el_ms", 0),
        "median_e2el_ms": data.get("median_e2el_ms", 0),
        "p99_e2el_ms": data.get("p99_e2el_ms", 0),
        "std_e2el_ms": data.get("std_e2el_ms", 0),
        # Throughput
        "output_token_throughput": data.get("output_token_throughput", 0),
        "total_token_throughput": data.get("total_token_throughput", 0),
        "request_throughput": data.get("request_throughput", 0),
        # Volume
        "completed": data.get("completed", 0),
        "total_input_tokens": data.get("total_input_tokens", 0),
        "total_output_tokens": data.get("total_output_tokens", 0),
        "model_id": data.get("model_id", ""),
        "backend": "guidellm",
        # Source pointer used by the native-table renderer
        "source_benchmarks_json": data.get("source_benchmarks_json", ""),
    }


def _load_normalized_vlm(filepath: str) -> Optional[Dict[str, Any]]:
    filename = Path(filepath).name
    m = re.search(
        r"isl-(\d+)_osl-(\d+)_maxcon-(\d+)_n-(\d+)_images-(\d+)_height-(\d+)_width-(\d+)",
        filename,
    )
    if not m:
        logger.warning(f"Could not parse image params from GuideLLM file: {filename}")
        return None
    isl, osl, concurrency, num_requests, images, height, width = map(int, m.groups())
    row = _load_normalized_text(filepath)
    row.update(
        {
            "task_type": "vlm",
            "isl": isl,
            "osl": osl,
            "concurrency": concurrency,
            "max_con": concurrency,
            "num_requests": num_requests,
            "images": images,
            "image_height": height,
            "image_width": width,
            "images_per_prompt": images,
        }
    )
    return row


# --------------------------------------------------------------------------- #
# Detailed percentiles table (AIPerf-style; flat, easy to consume downstream)
# --------------------------------------------------------------------------- #


def _render_percentile_table(rows: List[Dict[str, Any]], vlm: bool) -> str:
    """Render the AIPerf-style detailed percentiles markdown table."""
    from benchmarking.summary_report import get_markdown_table

    cols: List[Tuple[str, str]] = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
    ]
    if vlm:
        cols.extend(
            [
                ("image_height", "Image Height"),
                ("image_width", "Image Width"),
                ("images_per_prompt", "Images per Prompt"),
            ]
        )
    cols.extend(
        [
            ("num_requests", "N"),
            ("mean_ttft_ms", "TTFT Avg (ms)"),
            ("median_ttft_ms", "TTFT P50 (ms)"),
            ("p99_ttft_ms", "TTFT P99 (ms)"),
            ("mean_tpot_ms", "TPOT Avg (ms)"),
            ("median_tpot_ms", "TPOT P50 (ms)"),
            ("p99_tpot_ms", "TPOT P99 (ms)"),
            ("mean_e2el_ms", "E2EL Avg (ms)"),
            ("median_e2el_ms", "E2EL P50 (ms)"),
            ("p99_e2el_ms", "E2EL P99 (ms)"),
            ("output_token_throughput", "Output Tok/s"),
            ("total_token_throughput", "Total Tok/s"),
            ("request_throughput", "Req/s"),
        ]
    )

    NA = "N/A"
    display: List[Dict[str, str]] = []
    for row in rows:
        d: Dict[str, str] = {}
        for key, header in cols:
            v = row.get(key, NA)
            if v is None or v == "":
                d[header] = NA
            elif isinstance(v, float):
                if key == "request_throughput":
                    d[header] = f"{v:.4f}"
                elif key in ("output_token_throughput", "total_token_throughput"):
                    d[header] = f"{v:.2f}"
                else:
                    d[header] = f"{v:.1f}"
            else:
                d[header] = str(v)
        display.append(d)
    return get_markdown_table(display) + "\n"


def _metric_definitions_block() -> str:
    return (
        "**Metric Definitions:**\n"
        "> - **ISL**: Input Sequence Length (tokens)\n"
        "> - **OSL**: Output Sequence Length (tokens)\n"
        "> - **Concur**: Concurrent requests (batch size)\n"
        "> - **N**: Total number of requests\n"
        "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median, 99th percentile (ms)\n"
        "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
        "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
        "> - **Output Tok/s**: Output token throughput\n"
        "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
        "> - **Req/s**: Request throughput\n"
    )


# --------------------------------------------------------------------------- #
# Native GuideLLM-style tables, sourced directly from benchmarks.json
# --------------------------------------------------------------------------- #


def _render_native_tables(rows: List[Dict[str, Any]]) -> List[str]:
    """For every unique source `benchmarks.json` referenced by the normalized
    rows, emit the same five tables GuideLLM prints to the console at the end
    of a run (Run Summary, Text Metrics, Request Tokens, Request Latency,
    Server Throughput) — each with one markdown row per benchmark/sweep point.
    """
    from benchmarking.summary_report import get_markdown_table

    # Preserve insertion order so scenarios appear in a stable sequence
    sources: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for row in rows:
        src = row.get("source_benchmarks_json")
        if not src:
            continue
        meta = sources.setdefault(
            src,
            {"scenario": row.get("scenario") or "", "sweep_indexes": []},
        )
        if row.get("sweep_index") is not None:
            meta["sweep_indexes"].append(row["sweep_index"])

    blocks: List[str] = []
    for source_path, meta in sources.items():
        if not Path(source_path).is_file():
            logger.warning(
                f"GuideLLM source benchmarks.json not found, skipping native "
                f"tables for: {source_path}"
            )
            continue
        try:
            with open(source_path, "r", encoding="utf-8") as fh:
                source_data = json.load(fh)
        except Exception as exc:
            logger.warning(
                f"Failed to read GuideLLM benchmarks.json '{source_path}': {exc}"
            )
            continue

        benches = source_data.get("benchmarks") or []
        if not benches:
            continue

        scenario = meta["scenario"] or Path(source_path).parent.name
        block = [
            f"##### GuideLLM Run Details — `{scenario}`",
            f"**Source:** `{source_path}`",
            "",
            "###### Run Summary",
            get_markdown_table(_table_run_summary(benches)) + "\n",
            "###### Text Metrics (Completed Requests) — Input",
            get_markdown_table(_table_text_metrics(benches, side="input")) + "\n",
            "###### Text Metrics (Completed Requests) — Output",
            get_markdown_table(_table_text_metrics(benches, side="output")) + "\n",
            "###### Request Token Statistics (Completed Requests)",
            get_markdown_table(_table_request_tokens(benches)) + "\n",
            "###### Request Latency Statistics (Completed Requests)",
            get_markdown_table(_table_request_latency(benches)) + "\n",
            "###### Server Throughput Statistics (All Requests)",
            get_markdown_table(_table_server_throughput(benches)) + "\n",
        ]
        blocks.append("\n".join(block))
    return blocks


# --- Per-table helpers -------------------------------------------------------

# Order of statuses we sum across ("all requests" buckets in GuideLLM).
_ALL_BUCKETS = ("successful", "incomplete", "errored")


def _stat(
    block: Optional[Dict[str, Any]],
    bucket: str,
    key: str,
    default: float = 0.0,
) -> float:
    """Read a value out of `<metric>.<bucket>.<key>`; falls back to
    `<metric>.<bucket>.percentiles.<key>` when the key looks like a percentile
    (e.g. `p50`, `p95`)."""
    if not isinstance(block, dict):
        return default
    bucket_block = block.get(bucket) or {}
    if key in bucket_block:
        v = bucket_block.get(key)
    else:
        v = (bucket_block.get("percentiles") or {}).get(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _strategy_label(bench: Dict[str, Any]) -> str:
    cfg = bench.get("config") or {}
    strat = cfg.get("strategy") or {}
    label = strat.get("type_") or "unknown"
    rate = strat.get("rate")
    if rate is not None:
        try:
            label = f"{label}@{float(rate):g}rps"
        except (TypeError, ValueError):
            label = f"{label}@{rate}"
    return label


def _fmt_time(epoch_seconds: Optional[float]) -> str:
    if not epoch_seconds:
        return ""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).strftime(
        "%H:%M:%S"
    )


def _table_run_summary(benches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, b in enumerate(benches):
        m = b.get("metrics") or {}
        totals = m.get("request_totals") or {}
        prompt = m.get("prompt_token_count")
        output = m.get("output_token_count")
        out.append(
            {
                "Sweep": str(i),
                "Strategy": _strategy_label(b),
                "Start": _fmt_time(b.get("start_time")),
                "End": _fmt_time(b.get("end_time")),
                "Dur (s)": f"{float(b.get('duration') or 0):.1f}",
                "Warm (s)": f"{float(b.get('warmup_duration') or 0):.1f}",
                "Cool (s)": f"{float(b.get('cooldown_duration') or 0):.1f}",
                "Comp": str(int(totals.get("successful") or 0)),
                "Inc": str(int(totals.get("incomplete") or 0)),
                "Err": str(int(totals.get("errored") or 0)),
                "In Tok (Comp)": f"{_stat(prompt, 'successful', 'total_sum'):.0f}",
                "In Tok (Inc)": f"{_stat(prompt, 'incomplete', 'total_sum'):.0f}",
                "In Tok (Err)": f"{_stat(prompt, 'errored', 'total_sum'):.0f}",
                "Out Tok (Comp)": f"{_stat(output, 'successful', 'total_sum'):.0f}",
                "Out Tok (Inc)": f"{_stat(output, 'incomplete', 'total_sum'):.0f}",
                "Out Tok (Err)": f"{_stat(output, 'errored', 'total_sum'):.0f}",
            }
        )
    return out


def _table_text_metrics(
    benches: List[Dict[str, Any]], side: str
) -> List[Dict[str, str]]:
    """One row per benchmark; columns mirror the GuideLLM "Text Metrics
    Statistics" table for either the input or the output side, covering
    tokens, words, and characters with median/p95 per-request and
    median/mean per-second statistics."""
    out: List[Dict[str, str]] = []
    for i, b in enumerate(benches):
        text_block = ((b.get("metrics") or {}).get("text") or {})
        row = {"Sweep": str(i), "Strategy": _strategy_label(b)}
        for unit in ("tokens", "words", "characters"):
            unit_block = text_block.get(unit) or {}
            per_req = unit_block.get(side)
            per_sec = unit_block.get(f"{side}_per_second")
            unit_short = {"tokens": "Tok", "words": "Wrd", "characters": "Chr"}[unit]
            row[f"{unit_short}/Req Mdn"] = f"{_stat(per_req, 'successful', 'median'):.1f}"
            row[f"{unit_short}/Req p95"] = f"{_stat(per_req, 'successful', 'p95'):.1f}"
            row[f"{unit_short}/Sec Mdn"] = f"{_stat(per_sec, 'successful', 'median'):.1f}"
            row[f"{unit_short}/Sec Mean"] = f"{_stat(per_sec, 'successful', 'mean'):.1f}"
        out.append(row)
    return out


def _table_request_tokens(benches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, b in enumerate(benches):
        m = b.get("metrics") or {}
        in_tok = m.get("prompt_token_count")
        out_tok = m.get("output_token_count")
        tot_tok = m.get("total_token_count")
        stream = m.get("request_streaming_iterations_count")
        out_per_iter = m.get("output_tokens_per_iteration")
        out.append(
            {
                "Sweep": str(i),
                "Strategy": _strategy_label(b),
                "In Tok/Req Mdn": f"{_stat(in_tok, 'successful', 'median'):.1f}",
                "In Tok/Req p95": f"{_stat(in_tok, 'successful', 'p95'):.1f}",
                "Out Tok/Req Mdn": f"{_stat(out_tok, 'successful', 'median'):.1f}",
                "Out Tok/Req p95": f"{_stat(out_tok, 'successful', 'p95'):.1f}",
                "Total Tok/Req Mdn": f"{_stat(tot_tok, 'successful', 'median'):.1f}",
                "Total Tok/Req p95": f"{_stat(tot_tok, 'successful', 'p95'):.1f}",
                "Stream Iter/Req Mdn": f"{_stat(stream, 'successful', 'median'):.1f}",
                "Stream Iter/Req p95": f"{_stat(stream, 'successful', 'p95'):.1f}",
                "Out Tok/Iter Mdn": f"{_stat(out_per_iter, 'successful', 'median'):.1f}",
                "Out Tok/Iter p95": f"{_stat(out_per_iter, 'successful', 'p95'):.1f}",
            }
        )
    return out


def _table_request_latency(benches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, b in enumerate(benches):
        m = b.get("metrics") or {}
        lat = m.get("request_latency")  # seconds
        ttft = m.get("time_to_first_token_ms")
        itl = m.get("inter_token_latency_ms")
        tpot = m.get("time_per_output_token_ms")
        out.append(
            {
                "Sweep": str(i),
                "Strategy": _strategy_label(b),
                "Req Latency Mdn (s)": f"{_stat(lat, 'successful', 'median'):.2f}",
                "Req Latency p95 (s)": f"{_stat(lat, 'successful', 'p95'):.2f}",
                "TTFT Mdn (ms)": f"{_stat(ttft, 'successful', 'median'):.1f}",
                "TTFT p95 (ms)": f"{_stat(ttft, 'successful', 'p95'):.1f}",
                "ITL Mdn (ms)": f"{_stat(itl, 'successful', 'median'):.1f}",
                "ITL p95 (ms)": f"{_stat(itl, 'successful', 'p95'):.1f}",
                "TPOT Mdn (ms)": f"{_stat(tpot, 'successful', 'median'):.1f}",
                "TPOT p95 (ms)": f"{_stat(tpot, 'successful', 'p95'):.1f}",
            }
        )
    return out


def _table_server_throughput(benches: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Mirrors GuideLLM's "Server Throughput Statistics (All Requests)" table.
    The "All Requests" qualifier means we sum the `successful + incomplete +
    errored` buckets for the per-second metrics (matching how GuideLLM itself
    presents these on the console)."""
    out: List[Dict[str, str]] = []
    for i, b in enumerate(benches):
        m = b.get("metrics") or {}
        conc = m.get("request_concurrency")
        req_per_sec = m.get("requests_per_second")
        in_per_sec = m.get("prompt_tokens_per_second")
        out_per_sec = m.get("output_tokens_per_second")
        tot_per_sec = m.get("tokens_per_second")
        out.append(
            {
                "Sweep": str(i),
                "Strategy": _strategy_label(b),
                "Concurrency Mdn": f"{_stat(conc, 'successful', 'median'):.1f}",
                "Concurrency Mean": f"{_stat(conc, 'successful', 'mean'):.1f}",
                "Req/s Mean (All)": f"{sum(_stat(req_per_sec, bk, 'mean') for bk in _ALL_BUCKETS):.2f}",
                "In Tok/s Mean (All)": f"{sum(_stat(in_per_sec, bk, 'mean') for bk in _ALL_BUCKETS):.1f}",
                "Out Tok/s Mean (All)": f"{sum(_stat(out_per_sec, bk, 'mean') for bk in _ALL_BUCKETS):.1f}",
                "Total Tok/s Mean (All)": f"{sum(_stat(tot_per_sec, bk, 'mean') for bk in _ALL_BUCKETS):.1f}",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# CSV writer
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([str(row.get(h, "")) for h in headers])
