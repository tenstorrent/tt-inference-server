# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence

from report_module.display import (
    DISPLAY_NAMES,
    FOOTNOTES,
    decimal_places,
    display_name,
    target_checks_header,
)
from report_module.formatting import format_value
from report_module.markdown_table import build_markdown_table
from report_module.schema import Block
from report_module.status import TestStatus, glyph_for_label

logger = logging.getLogger(__name__)

RendererFn = Callable[[Block, Mapping[str, Any]], str]

HIDDEN_COLUMNS = frozenset(
    {
        "kind",
        "model",
        "device",
        "timestamp",
        "test_name",
        "description",
        "elapsed_seconds",
        "logs",
    }
)

GENERIC_KINDS: tuple[str, ...] = (
    "spec_tests",
    "stress_tests",
)

# Per-kind columns hidden on top of HIDDEN_COLUMNS. For test blocks the
# pass/fail wrapper fields duplicate the 🧪 Test Results summary table, so we
# drop them here and let the block table carry only the metrics.
_KIND_EXTRA_HIDDEN: Dict[str, frozenset] = {
    kind: frozenset({"status", "success", "attempts"}) for kind in GENERIC_KINDS
}


def _hidden_columns(kind: str) -> frozenset:
    return HIDDEN_COLUMNS | _KIND_EXTRA_HIDDEN.get(kind, frozenset())


EVALS_KIND = "evals"

BENCHMARKS_KIND = "benchmarks"

EVALS_METHODOLOGY_NOTE = (
    "Note: The ratio to published scores defines if eval ran roughly correctly, "
    "as the exact methodology of the model publisher cannot always be reproduced. "
    "For this reason the accuracy check is based first on being equivalent to the "
    "GPU reference within a +/- tolerance. If a value GPU reference is not "
    "available, the accuracy check is based on the direct ratio to the published "
    "score."
)

BENCHMARK_TIER_NOTE = (
    "Note: The Target Check column reflects only the strictest `target` tier. "
    "The Target Checks table grades three tiers — functional, complete, and "
    "target — from most to least lenient. Acceptance criteria pass a benchmark "
    "when any single tier meets all of its checks."
)

UNGRADED_SWEEP_NOTE = (
    "Note: No perf targets are configured for these sweep points, so these "
    "rows are reported for information only and are not graded."
)

_REGISTRY: Dict[str, RendererFn] = {}


def register(kind: str) -> Callable[[RendererFn], RendererFn]:
    """Decorator used by renderer functions to self-register at import time."""

    def decorator(fn: RendererFn) -> RendererFn:
        if kind in _REGISTRY:
            raise ValueError(f"Renderer for kind '{kind}' is already registered")
        _REGISTRY[kind] = fn
        return fn

    return decorator


def get_renderer(kind: str) -> RendererFn:
    """Return the renderer for ``kind``, falling back to a generic table."""
    return _REGISTRY.get(kind, render_generic_table)


def registered_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _extract_records(block: Block) -> List[Dict[str, Any]]:
    data = block.data
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        candidate: Sequence[Any] = data
    elif isinstance(data, Mapping):
        raw = data.get("records")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            candidate = raw
        elif data and all(isinstance(v, Mapping) for v in data.values()):
            # dict-of-dicts: surface the outer key as a "name" column so
            # callers can pass shapes like {"metric_a": {...}, "metric_b": {...}}
            # without flattening.
            return [{"name": str(k), **dict(v)} for k, v in data.items()]
        else:
            # Flat dict: render as a single pivoted record.
            return [dict(data)]
    else:
        return []

    rows: List[Dict[str, Any]] = []
    for item in candidate:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _resolve_model_device(
    block: Block, metadata: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    data = block.data if isinstance(block.data, Mapping) else {}
    targets = block.targets if isinstance(block.targets, Mapping) else {}
    model = str(
        data.get("model")
        or targets.get("model")
        or metadata.get("model_repo")
        or metadata.get("model_name")
        or ""
    )
    device = str(
        data.get("device") or targets.get("device") or metadata.get("device") or ""
    )
    if (not model or not device) and records:
        first = records[0]
        if not model:
            model = str(first.get("model") or "")
        if not device:
            device = str(first.get("device") or "")
    return model, device


def _heading(
    kind: str,
    model: str,
    device: str,
    title_override: str = "",
    task_type: str = "",
) -> str:
    if title_override:
        label = title_override
    elif task_type and kind:
        label = (
            f"{task_type.replace('_', ' ').title()} {kind.replace('_', ' ').title()}"
        )
    else:
        label = kind.replace("_", " ").title()
    suffix_bits = [bit for bit in (model, device) if bit]
    if not suffix_bits:
        return f"### {label}"
    return f"### {label} for {' on '.join(suffix_bits)}"


def _ordered_display_columns(
    records: Sequence[Mapping[str, Any]], hidden: frozenset = HIDDEN_COLUMNS
) -> List[str]:
    seen: Dict[str, None] = {}
    for record in records:
        for key in record.keys():
            if key in hidden or key in seen:
                continue
            seen[key] = None
    columns = list(seen.keys())
    return [col for col in columns if not _column_is_empty(col, records)]


def _column_is_empty(column: str, records: Sequence[Mapping[str, Any]]) -> bool:
    for record in records:
        if column not in record:
            continue
        value = record[column]
        if value is None:
            continue
        if isinstance(value, (list, tuple, dict, set)) and not value:
            continue
        if isinstance(value, str) and not value:
            continue
        return False
    return True


PIVOT_FIELD_HEADER = "**field**"
PIVOT_VALUE_HEADER = "**value**"

_CHECK_COLUMN_SUFFIX = "_check"
_CHECK_INT_TO_LABEL: Dict[int, str] = {1: "NA", 2: "PASS", 3: "FAIL"}

# Columns that carry a grading verdict rather than a measurement.
_GRADING_COLUMNS: tuple[str, ...] = ("status", "target_check")


def _is_na_cell(column: str, value: Any) -> bool:
    if column.endswith(_CHECK_COLUMN_SUFFIX):
        return _CHECK_INT_TO_LABEL.get(value) == "NA"
    return TestStatus.from_value(value) is TestStatus.NA


def _ungraded_columns(records: Sequence[Mapping[str, Any]]) -> frozenset:
    """Grading columns to drop because not one row in the table is graded."""
    present = frozenset(
        column for column in _GRADING_COLUMNS if not _column_is_empty(column, records)
    )
    if not present:
        return frozenset()
    for record in records:
        for column in present:
            value = record.get(column)
            if value is not None and not _is_na_cell(column, value):
                return frozenset()
    return present


def _format_cell(column: str, value: Any) -> str:
    if (
        column.endswith(_CHECK_COLUMN_SUFFIX)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value in _CHECK_INT_TO_LABEL
    ):
        label = _CHECK_INT_TO_LABEL[value]
        emoji = glyph_for_label(label)
        return f"{emoji} {label}" if emoji else label
    if column == "status" and isinstance(value, str) and value:
        label = value.upper()
        emoji = glyph_for_label(value)
        return f"{emoji} {label}" if emoji else label
    places = decimal_places(column)
    if places is not None and isinstance(value, float) and value == value:
        return f"{value:.{places}f}"
    return format_value(value)


def _pivot_single_record(
    record: Mapping[str, Any], columns: Sequence[str]
) -> List[Dict[str, str]]:
    """Transpose a single record into a two-column field/value table.

    Dict keys double as column headers, so they're bolded markdown here
    to make the header row stand out.
    """
    return [
        {
            PIVOT_FIELD_HEADER: display_name(col),
            PIVOT_VALUE_HEADER: _format_cell(col, record.get(col)),
        }
        for col in columns
    ]


def _build_table(
    records: Sequence[Mapping[str, Any]],
    header_fn: Callable[[str], str] = display_name,
    hidden: frozenset = HIDDEN_COLUMNS,
) -> str:
    columns = _ordered_display_columns(records, hidden)
    if not columns:
        return ""
    display_rows = [
        {header_fn(col): _format_cell(col, record.get(col)) for col in columns}
        for record in records
    ]
    return build_markdown_table(display_rows)


_TARGET_CHECKS_KEY = "target_checks"

_SNAKE_CASE = re.compile(r"^[a-z0-9_]+$")


def _humanize_key(key: str) -> str:
    """Turn ``"summary_stats"`` into ``"Summary Stats"`` while leaving
    already-formatted keys (``"TTFT vs. Context"``) untouched.

    Prefers an explicit :mod:`display` mapping when one exists"""
    if key in DISPLAY_NAMES:
        return display_name(key)
    if _SNAKE_CASE.match(key):
        return key.replace("_", " ").title()
    return key


def _is_subtable_value(value: Any) -> bool:
    """Whether a field value should render as its own sub-table.

    Mappings (flat or dict-of-dicts) and lists of dicts qualify;
    scalars and lists of primitives stay inline in the parent row.
    """
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple)) and value:
        return any(isinstance(item, Mapping) for item in value)
    return False


def _footnote_for(block: Block) -> str | None:
    targets = block.targets if isinstance(block.targets, Mapping) else {}
    tool = targets.get("tool")
    if tool:
        footnote = FOOTNOTES.get(str(tool))
        if footnote:
            return footnote
    return FOOTNOTES.get(block.kind)


def render_generic_table(
    block: Block, metadata: Mapping[str, Any], extra_hidden: frozenset = frozenset()
) -> str:
    """Render a block as a heading plus one or more markdown tables.

    - Multi-record blocks render as a single multi-row table (the
      existing behavior for grouped records like several ``evals``).
    - Single-record blocks split the record's fields: scalar fields
      collapse into the main pivot table; fields whose value is a dict
      or list-of-dicts become their own H4 sub-table, in insertion order.

    ``extra_hidden`` drops columns on top of the per-kind hidden set, for
    callers that decide per-block what is worth showing.
    """
    records = _extract_records(block)
    if not records:
        return ""

    model, device = _resolve_model_device(block, metadata, records)
    heading = _heading(
        block.kind, model, device, block.title or "", block.task_type or ""
    )
    footnote = _footnote_for(block)
    hidden = _hidden_columns(block.kind) | extra_hidden

    if len(records) > 1:
        table = _build_table(records, hidden=hidden)
        if not table:
            return ""
        parts = [heading, table]
        if footnote:
            parts.append(footnote)
        return "\n\n".join(parts)

    record = records[0]
    scalar_fields: Dict[str, Any] = {}
    nested_fields: List[tuple[str, Any]] = []
    for key, value in record.items():
        if key in hidden:
            continue
        if _is_subtable_value(value):
            nested_fields.append((key, value))
        else:
            scalar_fields[key] = value

    parts: List[str] = [heading]
    if scalar_fields:
        scalar_table = _build_table([scalar_fields], hidden=hidden)
        if scalar_table:
            parts.append(scalar_table)

    for key, value in nested_fields:
        sub_records = _extract_records(Block(kind=key, data=value))
        if not sub_records:
            continue
        header_fn = target_checks_header if key == _TARGET_CHECKS_KEY else display_name
        sub_table = _build_table(sub_records, header_fn=header_fn)
        if not sub_table:
            continue
        parts.append(f"#### {_humanize_key(key)}\n\n{sub_table}")

    if len(parts) <= 1:
        return ""
    if footnote:
        parts.append(footnote)
    return "\n\n".join(parts)


@register(EVALS_KIND)
def render_evals(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render eval results as one table followed by the methodology note."""
    records = _extract_records(block)
    if not records:
        return ""
    model, device = _resolve_model_device(block, metadata, records)
    heading = _heading(
        block.kind, model, device, block.title or "", block.task_type or ""
    )
    table = _build_table(records)
    if not table:
        return ""
    return f"{heading}\n\n{table}\n\n{EVALS_METHODOLOGY_NOTE}"


def _has_target_checks(block: Block) -> bool:
    """Whether the block carries a tiered ``target_checks`` mapping.

    Handles both shapes: top-level (audio sweep, run summaries) and nested
    one level under a section dict (image/video/cnn/etc. "Benchmarks").
    """
    data = block.data
    if not isinstance(data, Mapping):
        return False
    if isinstance(data.get(_TARGET_CHECKS_KEY), Mapping):
        return True
    return any(
        isinstance(value, Mapping)
        and isinstance(value.get(_TARGET_CHECKS_KEY), Mapping)
        for value in data.values()
    )


VBENCH_KIND = "vbench"


VBENCH_OUTPUT_NOTE = (
    "Means over each frame against the GPU clip from the same noise: PSNR and "
    "SSIM higher is closer, LPIPS lower."
)


def _vbench_mark(check: Mapping[str, Any] | None) -> str:
    """A failed gate check's marker for its table cell."""
    return " ❌" if check and check.get("status") == "FAIL" else ""


def _vbench_number(value: Any, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


@register(VBENCH_KIND)
def render_vbench(block: Block, metadata: Mapping[str, Any]) -> str:
    """The I2V eval's VBench scores against each GPU reference, and the gate.

    Built from the scorer's summary (vbench_i2v.report_summary): one row per
    dimension with each reference's score and TT's change from it, then each
    reference's frame metrics per resolution. A failed gate check is marked in
    its cell and listed below. Per-clip detail stays in the run's vbench.json.
    """
    data = block.data if isinstance(block.data, Mapping) else {}
    gate = data.get("gate") or {}
    checks = {
        (c.get("reference"), c.get("check"), c.get("resolution")): c
        for c in gate.get("checks") or []
    }
    references = [r for r in data.get("references") or [] if "error" not in r]
    scores = data.get("scores") or {}
    problems = data.get("problems") or {}

    intro = (
        f"{data.get('num_clips') or 0} clips, scored on {data.get('device') or '?'}."
    )
    if gate:
        verdict = gate.get("verdict", "NA")
        intro += f" Gate: {glyph_for_label(verdict)} {verdict}".rstrip()
        intro += f" ({sum(c.get('status') == 'FAIL' for c in checks.values())} of"
        intro += f" {len(checks)} checks failed)."
    lines = [f"### VBench: {data.get('heading') or block.title}", "", intro]
    if data.get("error"):
        lines += ["", f"{data['error']}; the scores are what finished."]

    rows = []
    for name in [*scores, *(p for p in problems if p not in scores)]:
        run_checks = [checks.get((r.get("run"), name, None)) for r in references]
        reported = any(c and c.get("status") == "REPORTED" for c in run_checks)
        row = {
            "Dimension": f"{name} (reported)" if reported else name,
            "TT": _vbench_number(scores.get(name), 4) or str(problems[name])[:80],
        }
        for reference, check in zip(references, run_checks):
            run = reference.get("run") or reference["reference"]
            label = f"{run} ({reference['role']})" if reference.get("role") else run
            limit = reference.get("tolerance_pct")
            gpu = (reference.get("dimensions") or {}).get(name) or {}
            change = ""
            if gpu.get("gpu"):
                change = f"{100 * (gpu['tt'] - gpu['gpu']) / gpu['gpu']:+.2f}"
            row[f"GPU {label}"] = _vbench_number(gpu.get("gpu"), 4)
            delta = f"Δ % vs {run}" + (f" (≥ −{limit:g})" if limit is not None else "")
            row[delta] = change + _vbench_mark(check)
        rows.append(row)
    if rows:
        lines += ["", build_markdown_table(rows)]

    output = []
    for reference in references:
        run = reference.get("run") or reference["reference"]
        for res, m in (reference.get("output") or {}).items():
            mark = {
                key: _vbench_mark(checks.get((run, f"mean_{key}", res)))
                for key in ("psnr", "ssim", "lpips")
            }
            worst = ""
            if m.get("worst_lpips") is not None:
                worst = f"{m['worst_lpips']:.3f} ({m['worst_clip']})"
                worst += _vbench_mark(checks.get((run, "clip_lpips", res)))
            output.append(
                {
                    "GPU reference": run,
                    "Resolution": res,
                    "Clips": m.get("clips"),
                    "PSNR (dB)": _vbench_number(m.get("psnr"), 2) + mark["psnr"],
                    "SSIM": _vbench_number(m.get("ssim"), 3) + mark["ssim"],
                    "LPIPS": _vbench_number(m.get("lpips"), 3) + mark["lpips"],
                    "Worst clip LPIPS": worst,
                }
            )
    if output:
        lines += ["", build_markdown_table(output), "", VBENCH_OUTPUT_NOTE]

    notes = []
    unchecked: Dict[str, List[str]] = {}
    for key, c in checks.items():
        where = " ".join(str(x) for x in key if x)
        if c.get("status") == "FAIL":
            notes.append(
                f"- ❌ {where}: {c.get('value')} (limit {c.get('threshold')})"
                + (f", {c['detail']}" if c.get("detail") else "")
            )
        elif c.get("status") == "NA":
            unchecked.setdefault(c.get("detail") or "no result", []).append(where)
    for reason, where in unchecked.items():
        notes.append(f"- Not checked ({reason}): {', '.join(where)}")
    for reference in data.get("references") or []:
        run = reference.get("run") or reference["reference"]
        if reference.get("noise_differs"):
            notes.append(
                f"- {run}: {reference['noise_differs']} of {reference['noise_checked']}"
                " seeds do not redraw the GPU run's saved noise on this host"
            )
        if reference.get("failed_clips"):
            notes.append(
                f"- {run}: frame metrics failed for {', '.join(reference['failed_clips'])}"
            )
        if reference.get("lpips_error"):
            notes.append(
                f"- {run}: LPIPS unavailable, {reference['lpips_error'][:200]}"
            )
        if "error" in reference and not gate:
            notes.append(f"- {run}: comparison failed, {reference['error'][:200]}")
    if notes:
        lines += ["", *notes]
    lines += ["", "Per-clip scores and frame metrics: vbench.json with the eval clips."]
    return "\n".join(lines)


@register(BENCHMARKS_KIND)
def render_benchmarks(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render a benchmark block plus whichever notes apply.

    The tier note is appended when tiered ``target_checks`` are present (LLM
    benchmarks without tiers get none). A sweep whose every row is ungraded
    drops its all-NA grading columns and explains why instead: the block data
    keeps ``status="na"`` either way, since that is what stops acceptance from
    reading "no targets configured" as "benchmarks passed".
    """
    ungraded = _ungraded_columns(_extract_records(block))
    table = render_generic_table(block, metadata, extra_hidden=ungraded)
    if not table:
        return ""
    notes = []
    if _has_target_checks(block):
        notes.append(BENCHMARK_TIER_NOTE)
    if ungraded:
        notes.append(UNGRADED_SWEEP_NOTE)
    return "\n\n".join([table, *notes])


for _kind in GENERIC_KINDS:
    register(_kind)(render_generic_table)


# Side-effect import: register the AIPerf prefix-cache renderer so any
# report_module consumer picks it up without importing the submodule
# explicitly. Placed after GENERIC_KINDS so all decorators from this file
# run first. The spec-decode renderer now lives in ``llm_module.parsers``
# and self-registers when that package is imported.
from report_module import prefix_cache_renderer as _prefix_cache_renderer  # noqa: E402,F401
from report_module import prefill_decode_renderer as _prefill_decode_renderer  # noqa: E402,F401
from report_module import agentic_traces_renderer as _agentic_traces_renderer  # noqa: E402,F401
