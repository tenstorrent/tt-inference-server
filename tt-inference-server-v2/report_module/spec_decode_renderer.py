# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Markdown renderer for ``aiperf_spec_decode`` Blocks.

Renders the speculative-decoding per-run table (one row per SPEED-Bench
sweep point, with the acceptance metrics scraped from the vLLM
Prometheus counters).

The renderer is fed the collapsed records from every
``aiperf_spec_decode`` Block emitted in the sweep -- the report
generator's ``_collapse_same_heading_blocks`` merges per-run Blocks
(same model + device -> same Block id) before invoking the renderer
once.

Registered with :func:`report_module.renderers.register` at import time
(see the bottom of this module).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from report_module.markdown_table import build_markdown_table
from report_module.renderers import _extract_records, _resolve_model_device, register
from report_module.schema import Block

logger = logging.getLogger(__name__)

NA = "N/A"

# Columns surfaced for per-run rows.
DISPLAY_COLUMNS: List[Tuple[str, str]] = [
    ("public_dataset", "Dataset"),
    ("output_len", "OSL"),
    ("max_concurrency", "Concur"),
    ("completed", "N Req"),
    ("acceptance_rate", "Accept Rate"),
    ("mean_accepted_length", "Mean Acc Len"),
    ("mean_ttft_ms", "TTFT Avg (ms)"),
    ("p95_ttft_ms", "TTFT P95 (ms)"),
    ("mean_tpot_ms", "TPOT Avg (ms)"),
    ("p95_tpot_ms", "TPOT P95 (ms)"),
    ("mean_e2el_ms", "E2EL Avg (ms)"),
    ("p95_e2el_ms", "E2EL P95 (ms)"),
    ("p99_e2el_ms", "E2EL P99 (ms)"),
    ("output_throughput", "Output Tok/s"),
    ("total_token_throughput", "Total Tok/s"),
]


def _format_number(col: str, value: Any) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, (int, float)):
        return str(value)
    if col == "acceptance_rate":
        return f"{value:.3f}"
    if col == "mean_accepted_length":
        return f"{value:.2f}"
    if col in ("output_throughput", "total_token_throughput"):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _row_to_display(
    row: Mapping[str, Any], columns: Sequence[Tuple[str, str]]
) -> Dict[str, str]:
    return {
        display_header: _format_number(col_name, row.get(col_name))
        for col_name, display_header in columns
    }


def _sort_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            str(r.get("public_dataset") or ""),
            int(r.get("output_len") or 0),
            int(r.get("max_concurrency") or 0),
        ),
    )


def render_aiperf_spec_decode(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render every spec-decode run into the per-run table."""
    records = _extract_records(block)
    if not records:
        return ""
    rows = _sort_rows([dict(r) for r in records])
    model, device = _resolve_model_device(block, metadata, rows)

    parts: List[str] = []
    suffix = " on ".join([p for p in (model, device) if p])
    heading_suffix = f" for {suffix}" if suffix else ""
    parts.append(f"### Speculative Decoding Benchmark Results{heading_suffix}")
    parts.append(
        "**Benchmarking Tool:** "
        "[AIPerf](https://github.com/ai-dynamo/aiperf) driving "
        "[SPEED-Bench](https://huggingface.co/datasets/nvidia/Speed-Bench) "
        "via `--public-dataset`. Acceptance metrics are per-run deltas of "
        "the vLLM Prometheus counters `vllm:spec_decode_num_accepted_tokens_total` / "
        "`vllm:spec_decode_num_draft_tokens_total` / "
        "`vllm:spec_decode_num_drafts_total` scraped from `/metrics` before "
        "and after each run."
    )

    per_run_table = build_markdown_table(
        [_row_to_display(r, DISPLAY_COLUMNS) for r in rows]
    )
    if per_run_table:
        parts.append(f"#### Per-Run Benchmark Sweeps\n\n{per_run_table}")

    parts.append(
        "**Metric definitions:**\n"
        "> - **Accept Rate**: `accepted_tokens / draft_tokens` from the "
        "vLLM Prometheus spec-decode counters across the run window "
        "(0.000 means the server is not running with speculative decoding "
        "enabled).\n"
        "> - **Mean Acc Len**: `1 + accepted_tokens / num_drafts` — the "
        "+1 is the bonus token verified by the target model at the end "
        "of every draft round.\n"
        "> - **OSL**: forced output length (`ignore_eos`); `N/A` means the "
        "model decoded to its natural EOS.\n"
        "> - **TTFT / TPOT / E2EL**: AIPerf percentiles from "
        "`profile_export_aiperf.json`."
    )

    return "\n\n".join(parts)


# Register at import time so any code path that imports report_module
# picks the renderer up.
register("aiperf_spec_decode")(render_aiperf_spec_decode)
