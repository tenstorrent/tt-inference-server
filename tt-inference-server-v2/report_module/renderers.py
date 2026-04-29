# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from __future__ import annotations

import logging
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


def get_renderer(kind: str) -> Optional[RendererFn]:
    return _REGISTRY.get(kind)


def registered_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _extract_records(block: Block) -> List[Dict[str, Any]]:
    data = block.data
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        candidate: Sequence[Any] = data
    elif isinstance(data, Mapping):
        raw = data.get("records")
        candidate = raw if isinstance(raw, Sequence) else ()
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


def render_generic_table(block: Block, metadata: Mapping[str, Any]) -> str:
    records = _extract_records(block)
    if not records:
        return ""

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

    model, device = _resolve_model_device(block, metadata, records)
    heading = _heading(block.kind, model, device, block.title or "")
    table = build_markdown_table(display_rows)
    return f"{heading}\n\n{table}"


for _kind in GENERIC_KINDS:
    register(_kind)(render_generic_table)
