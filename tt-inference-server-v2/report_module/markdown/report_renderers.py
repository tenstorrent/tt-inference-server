# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Markdown renderers for benchmark, stress-test and evaluation release sections.

These functions accept pre-computed data (flat dicts, row lists) and return
markdown strings.  They are consumed by the report strategies but live here
so that rendering is decoupled from data computation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from report_module.markdown.table_builder import get_markdown_table
from report_module.types import NOT_MEASURED_STR
from workflows.workflow_types import ModelType, ReportCheckTypes

ColumnSpec = Tuple[str, str, int]


PERCENTILE_METRIC_DEFINITIONS = (
    "**Metric Definitions:**\n"
    "> - **ISL**: Input Sequence Length (tokens)\n"
    "> - **OSL**: Output Sequence Length (tokens)\n"
    "> - **Concur**: Concurrent requests (batch size)\n"
    "> - **N**: Total number of requests\n"
    "> - **TTFT Avg/P50/P99**: Time To First Token - Average, Median (50th percentile), 99th percentile (ms)\n"
    "> - **TPOT Avg/P50/P99**: Time Per Output Token - Average, Median, 99th percentile (ms)\n"
    "> - **E2EL Avg/P50/P99**: End-to-End Latency - Average, Median, 99th percentile (ms)\n"
    "> - **Output Tok/s**: Output token throughput\n"
    "> - **Total Tok/s**: Total token throughput (input + output tokens)\n"
    "> - **Req/s**: Request throughput\n"
)


def percentile_table_markdown(rows: List[Dict[str, Any]], is_vlm: bool = False) -> str:
    """Render a mean/median/p99 percentile table from parsed benchmark rows."""
    display_cols: List[Tuple[str, str]] = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("concurrency", "Concur"),
    ]
    if is_vlm:
        display_cols.extend(
            [
                ("image_height", "Image Height"),
                ("image_width", "Image Width"),
                ("images_per_prompt", "Images per Prompt"),
            ]
        )
    display_cols.extend(
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

    display_dicts: List[Dict[str, str]] = []
    for row in rows:
        row_dict: Dict[str, str] = {}
        for col_name, header in display_cols:
            value = row.get(col_name, NOT_MEASURED_STR)
            if value is None or value == "":
                row_dict[header] = NOT_MEASURED_STR
            elif isinstance(value, float):
                if col_name == "request_throughput":
                    row_dict[header] = f"{value:.4f}"
                elif col_name in ("output_token_throughput", "total_token_throughput"):
                    row_dict[header] = f"{value:.2f}"
                else:
                    row_dict[header] = f"{value:.1f}"
            else:
                row_dict[header] = str(value)
        display_dicts.append(row_dict)

    return get_markdown_table(display_dicts)


def build_check_columns(target_checks: Optional[Dict]) -> List[Tuple[str, str]]:
    """Build display column definitions for target check metrics."""
    if not target_checks:
        return []

    check_cols = [
        (
            f"{k}_{metric}",
            " ".join(
                w.upper() if w.lower() == "ttft" else w.capitalize()
                for w in f"{k}_{metric}".split("_")
            )
            + (
                ""
                if metric.endswith("_check") or metric.endswith("_ratio")
                else " (ms)"
                if metric.startswith("ttft")
                else " (TPS)"
                if metric.startswith("tput")
                else ""
            ),
        )
        for k in target_checks
        for metric in ("ttft_check", "tput_user_check", "ttft", "tput_user")
    ]
    check_cols.sort(key=lambda col: not col[0].endswith("_check"))
    return check_cols


def render_target_table(
    rows: List[Dict[str, Any]],
    base_cols: List[Tuple[str, str]],
    check_cols: List[Tuple[str, str]],
) -> str:
    display_cols = base_cols + check_cols
    cols_to_round = [c[0] for c in check_cols]

    display_dicts = []
    for row in rows:
        row_dict = {}
        for col_name, header in display_cols:
            value = row.get(col_name, "N/A")
            if isinstance(value, ReportCheckTypes):
                row_dict[header] = ReportCheckTypes.to_display_string(value)
            elif col_name in cols_to_round and isinstance(value, float):
                row_dict[header] = f"{value:.2f}"
            else:
                row_dict[header] = str(value)
        display_dicts.append(row_dict)

    return get_markdown_table(display_dicts)


def benchmark_release_markdown(
    release_raw: List[Dict[str, Any]], target_checks: Optional[Dict] = None
) -> str:
    base_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Concurrency"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    check_cols = build_check_columns(target_checks)
    return render_target_table(release_raw, base_cols, check_cols)


def benchmark_vlm_release_markdown(
    release_raw: List[Dict[str, Any]], target_checks: Optional[Dict] = None
) -> str:
    base_cols = [
        ("isl", "ISL"),
        ("osl", "OSL"),
        ("max_concurrency", "Max Concurrency"),
        ("image_height", "Image Height"),
        ("image_width", "Image Width"),
        ("images_per_prompt", "Images per Prompt"),
        ("num_requests", "Num Requests"),
        ("ttft", "TTFT (ms)"),
        ("tput_user", "Tput User (TPS)"),
        ("tput", "Tput Decode (TPS)"),
    ]
    check_cols = build_check_columns(target_checks)
    return render_target_table(release_raw, base_cols, check_cols)


# -- Stress-test release rendering -------------------------------------------

_STRESS_CONFIG_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("input_sequence_length", "ISL", 0),
    ("output_sequence_length", "OSL", 0),
    ("max_con", "Concurrency", 0),
    ("num_prompts", "Num Prompts", 0),
)

_STRESS_SIMPLE_METRIC_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("mean_ttft_ms", "TTFT (ms)", 1),
    ("mean_tpot_ms", "TPOT (ms)", 1),
    ("mean_itl_ms", "ITL (ms)", 1),
    ("mean_e2el_ms", "E2EL (ms)", 1),
)

_STRESS_PERCENTILE_METRIC_GROUPS: Tuple[Tuple[str, str], ...] = (
    ("ttft", "TTFT"),
    ("tpot", "TPOT"),
    ("itl", "ITL"),
    ("e2el", "E2EL"),
)

_STRESS_PERCENTILE_SUFFIXES: Tuple[str, ...] = (
    "mean",
    "p5",
    "p25",
    "p50",
    "p95",
    "p99",
)

_STRESS_THROUGHPUT_COLUMNS: Tuple[ColumnSpec, ...] = (
    ("mean_tps", "Tput User (TPS)", 2),
    ("tps_decode_throughput", "Tput Decode (TPS)", 1),
)


def _build_stress_detailed_columns() -> Tuple[ColumnSpec, ...]:
    """Expand the percentile metric grid into per-column specs.

    Produces, in order, mean / p5 / p25 / p50 / p95 / p99 for each of
    TTFT, TPOT, ITL, E2EL.
    """
    columns: List[ColumnSpec] = []
    for metric_key, metric_label in _STRESS_PERCENTILE_METRIC_GROUPS:
        for suffix in _STRESS_PERCENTILE_SUFFIXES:
            if suffix == "mean":
                data_key = f"mean_{metric_key}_ms"
                header = f"{metric_label} (ms)"
            else:
                data_key = f"{suffix}_{metric_key}_ms"
                header = f"{suffix.upper()} {metric_label} (ms)"
            columns.append((data_key, header, 1))
    return tuple(columns)


_STRESS_DETAILED_METRIC_COLUMNS: Tuple[ColumnSpec, ...] = (
    _build_stress_detailed_columns()
)


def _format_numeric_cell(value: Any, decimals: int) -> str:
    """Format a numeric cell with fixed decimals; ``NOT_MEASURED_STR`` for missing/NaN."""
    if value is None or value == "" or value == NOT_MEASURED_STR:
        return NOT_MEASURED_STR
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:  # NaN guard
        return NOT_MEASURED_STR
    if decimals == 0:
        return str(int(numeric))
    return f"{numeric:.{decimals}f}"


def stress_tests_release_markdown(
    release_raw: List[Dict[str, Any]], percentile: bool = False
) -> str:
    """Render the stress-test release table.

    ``percentile=True`` expands each metric into mean + p5/p25/p50/p95/p99
    columns; otherwise a simple mean-only table is produced.
    """
    metric_cols = (
        _STRESS_DETAILED_METRIC_COLUMNS if percentile else _STRESS_SIMPLE_METRIC_COLUMNS
    )
    columns = _STRESS_CONFIG_COLUMNS + metric_cols + _STRESS_THROUGHPUT_COLUMNS

    display_dicts: List[Dict[str, str]] = [
        {
            header: _format_numeric_cell(row.get(data_key), decimals)
            for data_key, header, decimals in columns
        }
        for row in release_raw
    ]
    return get_markdown_table(display_dicts)


_TIERED_TARGETS_CNN_IMAGE_VIDEO = [
    ("tier", "Tier", lambda _row, tier, _key: tier.capitalize()),
    (
        "ttft_target",
        "TTFT Target (s)",
        lambda _row, _tier, tc: _fmt_float(tc.get("ttft")),
    ),
    (
        "ttft_measured",
        "TTFT Measured (s)",
        lambda row, _tier, _tc: _fmt_float(row.get("ttft")),
    ),
    (
        "ttft_ratio",
        "TTFT Ratio",
        lambda _row, _tier, tc: _fmt_float(tc.get("ttft_ratio")),
    ),
    (
        "ttft_check",
        "TTFT Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("ttft_check")),
    ),
    (
        "tput_check",
        "Tput User Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("tput_check")),
    ),
]

_TIERED_TARGETS_AUDIO = [
    ("tier", "Tier", lambda _row, tier, _tc: tier.capitalize()),
    (
        "ttft_target",
        "TTFT Target (ms)",
        lambda _row, _tier, tc: _fmt_float(tc.get("ttft")),
    ),
    (
        "ttft_measured",
        "TTFT Measured (ms)",
        lambda row, _tier, _tc: _fmt_float(
            row.get("mean_ttft_ms")
            or (
                row.get("ttft", 0) * 1000
                if isinstance(row.get("ttft"), (int, float))
                else None
            )
        ),
    ),
    (
        "ttft_ratio",
        "TTFT Ratio",
        lambda _row, _tier, tc: _fmt_float(tc.get("ttft_ratio")),
    ),
    (
        "ttft_check",
        "TTFT Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("ttft_check")),
    ),
]

_TIERED_TARGETS_TTS = _TIERED_TARGETS_AUDIO + [
    ("rtr_target", "RTR Target", lambda _row, _tier, tc: _fmt_float(tc.get("rtr"))),
    (
        "rtr_measured",
        "RTR Measured",
        lambda row, _tier, _tc: _fmt_float(row.get("rtr")),
    ),
    ("rtr_ratio", "RTR Ratio", lambda _row, _tier, tc: _fmt_float(tc.get("rtr_ratio"))),
    ("rtr_check", "RTR Check", lambda _row, _tier, tc: _fmt_check(tc.get("rtr_check"))),
]

_TIERED_TARGETS_EMBEDDING = [
    ("tier", "Tier", lambda _row, tier, _tc: tier.capitalize()),
    (
        "tput_user_target",
        "Tput User Target (TPS)",
        lambda _row, _tier, tc: _fmt_float(tc.get("tput_user")),
    ),
    (
        "tput_user_measured",
        "Tput User Measured (TPS)",
        lambda row, _tier, _tc: _fmt_float(row.get("tput_user")),
    ),
    (
        "tput_user_ratio",
        "Tput User Ratio",
        lambda _row, _tier, tc: _fmt_float(tc.get("tput_user_ratio")),
    ),
    (
        "tput_user_check",
        "Tput User Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("tput_user_check")),
    ),
    (
        "tput_prefill_target",
        "Tput Prefill Target (TPS)",
        lambda _row, _tier, tc: _fmt_float(tc.get("tput_prefill")),
    ),
    (
        "tput_prefill_measured",
        "Tput Prefill Measured (TPS)",
        lambda row, _tier, _tc: _fmt_float(row.get("tput_prefill")),
    ),
    (
        "tput_prefill_ratio",
        "Tput Prefill Ratio",
        lambda _row, _tier, tc: _fmt_float(tc.get("tput_prefill_ratio")),
    ),
    (
        "tput_prefill_check",
        "Tput Prefill Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("tput_prefill_check")),
    ),
    (
        "e2el_target",
        "E2EL Target (ms)",
        lambda _row, _tier, tc: _fmt_float(tc.get("e2el_ms")),
    ),
    (
        "e2el_measured",
        "E2EL Measured (ms)",
        lambda row, _tier, _tc: _fmt_float(row.get("e2el_ms")),
    ),
    (
        "e2el_ratio",
        "E2EL Ratio",
        lambda _row, _tier, tc: _fmt_float(tc.get("e2el_ms_ratio")),
    ),
    (
        "e2el_check",
        "E2EL Check",
        lambda _row, _tier, tc: _fmt_check(tc.get("e2el_ms_check")),
    ),
]

_TIERED_TARGETS_COLUMN_SETS = {
    ModelType.CNN.name: _TIERED_TARGETS_CNN_IMAGE_VIDEO,
    ModelType.IMAGE.name: _TIERED_TARGETS_CNN_IMAGE_VIDEO,
    ModelType.VIDEO.name: _TIERED_TARGETS_CNN_IMAGE_VIDEO,
    ModelType.AUDIO.name: _TIERED_TARGETS_AUDIO,
    ModelType.TEXT_TO_SPEECH.name: _TIERED_TARGETS_TTS,
    ModelType.EMBEDDING.name: _TIERED_TARGETS_EMBEDDING,
}


def _fmt_float(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, ReportCheckTypes):
        return ReportCheckTypes.to_display_string(value)
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 0.01 else f"{value:.2f}"
    return str(value)


def _fmt_check(value: Any) -> str:
    if isinstance(value, ReportCheckTypes):
        return ReportCheckTypes.to_display_string(value)
    if value is None:
        return "N/A"
    return str(value)


def _tier_is_missing(target_checks_tier: Dict[str, Any]) -> bool:
    """A tier is missing when every check field is NA."""
    check_fields = [k for k in target_checks_tier if k.endswith("_check")]
    if not check_fields:
        return True
    return all(target_checks_tier.get(k) == ReportCheckTypes.NA for k in check_fields)


def tiered_targets_markdown(summary_row: Dict[str, Any], model_type_name: str) -> str:
    """Render per-tier target-check rows as a single markdown table.

    Tiers whose checks are all ``N/A`` are skipped.  Returns an empty
    string if no tier has data or the model type has no column set.
    """
    column_set = _TIERED_TARGETS_COLUMN_SETS.get(model_type_name)
    if column_set is None:
        return ""

    target_checks = summary_row.get("target_checks", {}) or {}
    if not target_checks:
        return ""

    display_dicts: List[Dict[str, str]] = []
    for tier, tier_checks in target_checks.items():
        if _tier_is_missing(tier_checks):
            continue
        display_dicts.append(
            {
                header: getter(summary_row, tier, tier_checks)
                for _, header, getter in column_set
            }
        )

    if not display_dicts:
        return ""

    return get_markdown_table(display_dicts, include_notes=False)


def generate_evals_release_markdown(report_rows: List[Dict[str, Any]]) -> str:
    def format_value(key: str, value: Any, row: Dict) -> str:
        if key == "published_score":
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref = row.get("published_score_ref", "")
            return f"[{score_val}]({ref})" if ref else score_val
        if key == "gpu_reference_score":
            score_val = f"{value:.2f}" if isinstance(value, float) else str(value)
            ref = row.get("gpu_reference_score_ref", "")
            return f"[{score_val}]({ref})" if ref else score_val
        if key == "accuracy_check":
            return ReportCheckTypes.to_display_string(value)
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    formatted_rows = [
        {k: format_value(k, v, row) for k, v in row.items()} for row in report_rows
    ]

    remove_keys = {"published_score_ref", "metadata", "gpu_reference_score_ref"}
    headers = [h for h in formatted_rows[0].keys() if h not in remove_keys]

    column_widths = {
        h: max(len(h), max(len(row[h]) for row in formatted_rows)) for h in headers
    }

    header_row = "| " + " | ".join(f"{h:<{column_widths[h]}}" for h in headers) + " |"
    divider = "|-" + "-|-".join("-" * column_widths[h] for h in headers) + "-|"
    data_rows = [
        "| " + " | ".join(f"{row[h]:<{column_widths[h]}}" for h in headers) + " |"
        for row in formatted_rows
    ]

    explain = (
        "\n\nNote: The ratio to published scores defines if eval ran roughly correctly, "
        "as the exact methodology of the model publisher cannot always be reproduced. "
        "For this reason the accuracy check is based first on being equivalent to the "
        "GPU reference within a +/- tolerance. If a value GPU reference is not available, "
        "the accuracy check is based on the direct ratio to the published score."
    )

    return header_row + "\n" + divider + "\n" + "\n".join(data_rows) + explain


def generate_evals_summary_table(results: Dict, meta_data: Dict) -> str:
    rows = []
    for task_name, metrics in results.items():
        for metric_name, metric_value in metrics.items():
            if metric_name and metric_name != " " and isinstance(metric_value, float):
                rows.append((task_name, metric_name, f"{metric_value:.4f}"))

    if not rows:
        return "No evaluation results to display."

    col_widths = [max(len(row[i]) for row in rows) for i in range(3)]
    header = f"| {'Task Name'.ljust(col_widths[0])} | {'Metric'.ljust(col_widths[1])} | {'Value'.rjust(col_widths[2])} |"
    separator = f"|{'-' * (col_widths[0] + 2)}|{'-' * (col_widths[1] + 2)}|{'-' * (col_widths[2] + 2)}|"
    md = header + "\n" + separator + "\n"
    for task_name, metric_name, metric_value in rows:
        md += f"| {task_name.ljust(col_widths[0])} | {metric_name.ljust(col_widths[1])} | {metric_value.rjust(col_widths[2])} |\n"
    return md


# ── Simple-eval release + summary rendering ──────────────────────────────
#
# Simple-eval JSONs are already flat (one dict per task) and carry all the
# fields needed for display.  Rendering is a purely declarative mapping:
# ``_SIMPLE_EVAL_COLUMN_SETS[task_type] -> list of (key, header, getter)``.
# Adding a new task_type is two edits in this file: a new column list plus
# a new dispatch-dict entry.

SimpleEvalColumn = Tuple[str, str, Callable[[Dict[str, Any]], str]]


def _fmt_published_score(row: Dict[str, Any]) -> str:
    value = row.get("published_score")
    rendered = _fmt_float(value)
    ref = row.get("published_score_ref")
    return f"[{rendered}]({ref})" if ref else rendered


def _fmt_check_field(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return "N/A"
    if isinstance(value, ReportCheckTypes):
        return ReportCheckTypes.to_display_string(value)
    if isinstance(value, int):
        try:
            return ReportCheckTypes.to_display_string(ReportCheckTypes(value))
        except ValueError:
            return "N/A"
    return "N/A"


def _fmt_accuracy_check(row: Dict[str, Any]) -> str:
    return _fmt_check_field(row, "accuracy_check")


def _fmt_performance_check(row: Dict[str, Any]) -> str:
    return _fmt_check_field(row, "performance_check")


def _fmt_summary_score(row: Dict[str, Any]) -> str:
    # Embedding rows expose ``main_score`` instead of ``score``.
    value = row.get("score")
    if value is None:
        value = row.get("main_score")
    return _fmt_float(value)


_SIMPLE_EVAL_IMAGE_COLUMNS: List[SimpleEvalColumn] = [
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("score", "Score", lambda r: _fmt_float(r.get("score"))),
    ("fid_score", "FID", lambda r: _fmt_float(r.get("fid_score"))),
    ("average_clip", "CLIP (avg)", lambda r: _fmt_float(r.get("average_clip"))),
    (
        "deviation_clip_score",
        "CLIP (stdev)",
        lambda r: _fmt_float(r.get("deviation_clip_score")),
    ),
    ("tput_user", "Tput User (TPS)", lambda r: _fmt_float(r.get("tput_user"))),
    ("published_score", "Published Score", _fmt_published_score),
    ("accuracy_check", "Accuracy Check", _fmt_accuracy_check),
]

_SIMPLE_EVAL_VIDEO_COLUMNS: List[SimpleEvalColumn] = [
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("average_clip", "CLIP (avg)", lambda r: _fmt_float(r.get("average_clip"))),
    ("min_clip", "CLIP (min)", lambda r: _fmt_float(r.get("min_clip"))),
    ("max_clip", "CLIP (max)", lambda r: _fmt_float(r.get("max_clip"))),
    (
        "clip_standard_deviation",
        "CLIP (stdev)",
        lambda r: _fmt_float(r.get("clip_standard_deviation")),
    ),
    ("fvd", "FVD", lambda r: _fmt_float(r.get("fvd"))),
    ("fvmd", "FVMD", lambda r: _fmt_float(r.get("fvmd"))),
    ("num_prompts", "Prompts", lambda r: str(r.get("num_prompts", "N/A"))),
    (
        "num_inference_steps",
        "Steps",
        lambda r: str(r.get("num_inference_steps", "N/A")),
    ),
    ("published_score", "Published Score", _fmt_published_score),
    ("accuracy_check", "Accuracy Check", _fmt_accuracy_check),
]

_SIMPLE_EVAL_CNN_COLUMNS: List[SimpleEvalColumn] = [
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("score", "Score", lambda r: _fmt_float(r.get("score"))),
    ("published_score", "Published Score", _fmt_published_score),
    ("accuracy_check", "Accuracy Check", _fmt_accuracy_check),
]

_SIMPLE_EVAL_EMBEDDING_COLUMNS: List[SimpleEvalColumn] = [
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("main_score", "Main Score", lambda r: _fmt_float(r.get("main_score"))),
    ("pearson", "Pearson", lambda r: _fmt_float(r.get("pearson"))),
    ("spearman", "Spearman", lambda r: _fmt_float(r.get("spearman"))),
    (
        "cosine_pearson",
        "Cosine Pearson",
        lambda r: _fmt_float(r.get("cosine_pearson")),
    ),
    (
        "cosine_spearman",
        "Cosine Spearman",
        lambda r: _fmt_float(r.get("cosine_spearman")),
    ),
    (
        "manhattan_pearson",
        "Manhattan Pearson",
        lambda r: _fmt_float(r.get("manhattan_pearson")),
    ),
    (
        "manhattan_spearman",
        "Manhattan Spearman",
        lambda r: _fmt_float(r.get("manhattan_spearman")),
    ),
    (
        "euclidean_pearson",
        "Euclidean Pearson",
        lambda r: _fmt_float(r.get("euclidean_pearson")),
    ),
    (
        "euclidean_spearman",
        "Euclidean Spearman",
        lambda r: _fmt_float(r.get("euclidean_spearman")),
    ),
]

_SIMPLE_EVAL_TTS_COLUMNS: List[SimpleEvalColumn] = [
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("score", "Score", lambda r: _fmt_float(r.get("score"))),
    ("rtr", "RTR", lambda r: _fmt_float(r.get("rtr"))),
    ("p90_ttft", "P90 TTFT (ms)", lambda r: _fmt_float(r.get("p90_ttft"))),
    ("p95_ttft", "P95 TTFT (ms)", lambda r: _fmt_float(r.get("p95_ttft"))),
    ("published_score", "Published Score", _fmt_published_score),
    ("performance_check", "Performance Check", _fmt_performance_check),
    ("accuracy_check", "Accuracy Check", _fmt_accuracy_check),
]

# Dispatch is keyed on the literal ``task_type`` value found in each eval JSON.
# TTS emits ``"text_to_speech"`` (not ``"tts"``); if that ever changes the
# key must be updated in lock-step with the producer.
_SIMPLE_EVAL_COLUMN_SETS: Dict[str, List[SimpleEvalColumn]] = {
    "image": _SIMPLE_EVAL_IMAGE_COLUMNS,
    "video": _SIMPLE_EVAL_VIDEO_COLUMNS,
    "cnn": _SIMPLE_EVAL_CNN_COLUMNS,
    "embedding": _SIMPLE_EVAL_EMBEDDING_COLUMNS,
    "text_to_speech": _SIMPLE_EVAL_TTS_COLUMNS,
}

_SIMPLE_EVAL_SUMMARY_COLUMNS: List[SimpleEvalColumn] = [
    ("task_type", "Task Type", lambda r: str(r.get("task_type", "N/A"))),
    ("task_name", "Task", lambda r: str(r.get("task_name", "N/A"))),
    ("score", "Score", _fmt_summary_score),
    ("published_score", "Published Score", _fmt_published_score),
    ("accuracy_check", "Accuracy Check", _fmt_accuracy_check),
]


def _section_heading(task_type: str) -> str:
    return task_type.replace("_", " ").title()


def generate_simple_evals_release_markdown(rows: List[Dict[str, Any]]) -> str:
    """Render one markdown sub-section per ``task_type`` in *rows*.

    Rows whose ``task_type`` has no registered column set are skipped; this
    keeps a single unsupported row from breaking the whole report.
    """
    if not rows:
        return ""

    rows_by_task_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_task_type.setdefault(row.get("task_type", "unknown"), []).append(row)

    sections: List[str] = []
    for task_type, typed_rows in rows_by_task_type.items():
        columns = _SIMPLE_EVAL_COLUMN_SETS.get(task_type)
        if columns is None:
            continue
        display = [
            {header: getter(row) for _, header, getter in columns} for row in typed_rows
        ]
        sections.append(
            f"#### {_section_heading(task_type)} Accuracy Results\n\n"
            + get_markdown_table(display, include_notes=False)
        )
    return "\n\n".join(sections)


def generate_simple_evals_summary_table(rows: List[Dict[str, Any]]) -> str:
    """Compact per-row summary table spanning all task_types."""
    if not rows:
        return "No evaluation results to display."
    display = [
        {header: getter(row) for _, header, getter in _SIMPLE_EVAL_SUMMARY_COLUMNS}
        for row in rows
    ]
    return get_markdown_table(display, include_notes=False)
