# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


@dataclass(frozen=True)
class Block:
    """One section of a report.

    ``kind`` selects the renderer. ``data`` and ``targets`` are free-form
    and only the registered renderer for ``kind`` is expected to
    understand their shape. ``id`` disambiguates blocks that share a
    ``kind`` (e.g. two ``benchmarks`` blocks for different tools).
    """

    kind: str
    data: Any
    title: Optional[str] = None
    id: Optional[str] = None
    targets: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """Stable identifier used in filenames and JSON keys."""
        return f"{self.kind}__{self.id}" if self.id else self.kind

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Block":
        if "kind" not in payload:
            raise ValueError("Block payload is missing required 'kind' field")
        return Block(
            kind=str(payload["kind"]),
            data=payload.get("data"),
            title=payload.get("title"),
            id=payload.get("id"),
            targets=dict(payload.get("targets") or {}),
            checks=dict(payload.get("checks") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "data": self.data}
        if self.title is not None:
            out["title"] = self.title
        if self.id is not None:
            out["id"] = self.id
        if self.targets:
            out["targets"] = dict(self.targets)
        if self.checks:
            out["checks"] = dict(self.checks)
        return out


@dataclass(frozen=True)
class ReportSchema:
    """Top-level input to the generator."""

    metadata: Dict[str, Any]
    sections: List[Block]

    @property
    def report_id(self) -> str:
        value = self.metadata.get("report_id")
        if not value:
            raise ValueError("metadata.report_id is required")
        return str(value)

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("model_name", ""))

    @property
    def device(self) -> str:
        return str(self.metadata.get("device", ""))

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ReportSchema":
        metadata = dict(payload.get("metadata") or {})
        raw_sections = payload.get("sections") or []
        if not isinstance(raw_sections, list):
            raise TypeError("'sections' must be a list of block dicts")
        sections = [Block.from_dict(section) for section in raw_sections]
        return ReportSchema(metadata=metadata, sections=sections)

    @staticmethod
    def from_records(
        records: Sequence[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ReportSchema":
        """Build a schema from a flat list of records.

        Records are grouped by ``(kind, model, device)`` so each Block's
        rows come from the same test kind on the same (model, device)
        pair. Block titles, report IDs and top-level metadata are
        synthesised when absent.
        """
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of dicts")
        for idx, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"records[{idx}] is not a dict")
            if "kind" not in record:
                raise ValueError(f"records[{idx}] missing required 'kind' field")

        meta = dict(metadata or {})
        if records:
            first = records[0]
            meta.setdefault("model_name", str(first.get("model", "")))
            meta.setdefault("device", str(first.get("device", "")))
            meta.setdefault(
                "report_id", _synthesize_report_id(meta["model_name"], first)
            )
            meta.setdefault("generated_at", _first_valid_timestamp(records))
        else:
            meta.setdefault("model_name", "")
            meta.setdefault("device", "")
            meta.setdefault("report_id", _synthesize_report_id("", {}))

        sections = _group_records_to_blocks(records)
        return ReportSchema(metadata=meta, sections=sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "sections": [block.to_dict() for block in self.sections],
        }


SchemaLike = Union[ReportSchema, Mapping[str, Any], Sequence[Mapping[str, Any]]]


def _group_records_to_blocks(records: Sequence[Mapping[str, Any]]) -> List[Block]:
    """Group records by ``(kind, model, device)`` preserving first-seen order.

    The resulting Block's ``data`` is ``{"model", "device", "records"}``;
    the renderer registered for the Block's ``kind`` is responsible for
    rendering the records list.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    order: List[tuple] = []
    for record in records:
        kind = str(record.get("kind"))
        model = str(record.get("model", ""))
        device = str(record.get("device", ""))
        key = (kind, model, device)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(dict(record))

    blocks: List[Block] = []
    for kind, model, device in order:
        rows = groups[(kind, model, device)]
        block_id = _slugify_block_id(model, device)
        blocks.append(
            Block(
                kind=kind,
                id=block_id or None,
                data={"model": model, "device": device, "records": rows},
            )
        )
    return blocks


def _slugify_block_id(model: str, device: str) -> str:
    parts = [p for p in (model, device) if p]
    if not parts:
        return ""
    return "_".join(parts).replace("/", "__").replace(" ", "_")


def _synthesize_report_id(model_name: str, first_record: Mapping[str, Any]) -> str:
    ts_text = _record_timestamp_text(first_record) or datetime.utcnow().isoformat()
    ts_compact = _compact_timestamp(ts_text)
    base = model_name or "report"
    return f"{base}_{ts_compact}"


def _first_valid_timestamp(records: Sequence[Mapping[str, Any]]) -> str:
    for record in records:
        text = _record_timestamp_text(record)
        if text:
            return text
    return datetime.utcnow().isoformat()


def _record_timestamp_text(record: Mapping[str, Any]) -> str:
    raw = record.get("timestamp")
    if raw is None:
        return ""
    return str(raw).strip()


def _compact_timestamp(text: str) -> str:
    """Turn ``'2026-04-11 01:50:50'`` into ``'20260411_015050'``.

    Falls back to stripping non-alphanumerics for inputs that don't parse
    as an ISO-ish timestamp.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return "".join(ch for ch in text if ch.isalnum()) or "unknown"
