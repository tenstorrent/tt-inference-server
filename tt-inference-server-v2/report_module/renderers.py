# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from report_module.formatting import format_value
from report_module.markdown_table import build_markdown_table
from report_module.schema import Block

logger = logging.getLogger(__name__)

RendererFn = Callable[[Block, Mapping[str, Any]], str]

HIDDEN_COLUMNS = frozenset({"kind", "model", "device", "timestamp"})

GENERIC_KINDS: tuple[str, ...] = (
    "benchmarks",
    "benchmarks_summary",
    "evals",
    "percentile_benchmarks",
    "parameter_support_tests",
    "stress_tests",
    "server_test_results",
    "results_details",
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
    model = str(data.get("model") or metadata.get("model_name") or "")
    device = str(data.get("device") or metadata.get("device") or "")
    if (not model or not device) and records:
        first = records[0]
        if not model:
            model = str(first.get("model") or "")
        if not device:
            device = str(first.get("device") or "")
    return model, device


def _heading(kind: str, model: str, device: str, title_override: str = "") -> str:
    label = title_override or kind.replace("_", " ").title()
    suffix_bits = [bit for bit in (model, device) if bit]
    if not suffix_bits:
        return f"### {label}"
    return f"### {label} for {' on '.join(suffix_bits)}"


def _ordered_display_columns(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """First-seen column order, with hidden columns excluded."""
    seen: Dict[str, None] = {}
    for record in records:
        for key in record.keys():
            if key in HIDDEN_COLUMNS or key in seen:
                continue
            seen[key] = None
    return list(seen.keys())


PIVOT_FIELD_HEADER = "**field**"
PIVOT_VALUE_HEADER = "**value**"


def _pivot_single_record(
    record: Mapping[str, Any], columns: Sequence[str]
) -> List[Dict[str, str]]:
    """Transpose a single record into a two-column field/value table.

    Dict keys double as column headers, so they're bolded markdown here
    to make the header row stand out.
    """
    return [
        {
            PIVOT_FIELD_HEADER: col,
            PIVOT_VALUE_HEADER: format_value(record.get(col)),
        }
        for col in columns
    ]


def _build_table(records: Sequence[Mapping[str, Any]]) -> str:
    """Render a list of records to a markdown table (no heading).

    Single-record inputs pivot into a two-column field/value table;
    multi-record inputs render as one row per record.
    """
    columns = _ordered_display_columns(records)
    if not columns:
        return ""
    if len(records) == 1:
        display_rows = _pivot_single_record(records[0], columns)
    else:
        display_rows = [
            {col: format_value(record.get(col)) for col in columns}
            for record in records
        ]
    return build_markdown_table(display_rows)


_SNAKE_CASE = re.compile(r"^[a-z0-9_]+$")


def _humanize_key(key: str) -> str:
    """Turn ``"summary_stats"`` into ``"Summary Stats"`` while leaving
    already-formatted keys (``"TTFT vs. Context"``) untouched."""
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


def render_generic_table(block: Block, metadata: Mapping[str, Any]) -> str:
    """Render a block as a heading plus one or more markdown tables.

    - Multi-record blocks render as a single multi-row table (the
      existing behavior for grouped records like several ``evals``).
    - Single-record blocks split the record's fields: scalar fields
      collapse into the main pivot table; fields whose value is a dict
      or list-of-dicts become their own H4 sub-table, in insertion order.
    """
    records = _extract_records(block)
    if not records:
        return ""

    model, device = _resolve_model_device(block, metadata, records)
    heading = _heading(block.kind, model, device, block.title or "")

    if len(records) > 1:
        table = _build_table(records)
        return f"{heading}\n\n{table}" if table else ""

    record = records[0]
    scalar_fields: Dict[str, Any] = {}
    nested_fields: List[tuple[str, Any]] = []
    for key, value in record.items():
        if key in HIDDEN_COLUMNS:
            continue
        if _is_subtable_value(value):
            nested_fields.append((key, value))
        else:
            scalar_fields[key] = value

    parts: List[str] = [heading]
    if scalar_fields:
        scalar_table = _build_table([scalar_fields])
        if scalar_table:
            parts.append(scalar_table)

    for key, value in nested_fields:
        sub_records = _extract_records(Block(kind=key, data=value))
        if not sub_records:
            continue
        sub_table = _build_table(sub_records)
        if not sub_table:
            continue
        parts.append(f"#### {_humanize_key(key)}\n\n{sub_table}")

    return "\n\n".join(parts) if len(parts) > 1 else ""


for _kind in GENERIC_KINDS:
    register(_kind)(render_generic_table)
