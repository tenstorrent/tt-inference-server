# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Data-driven report generator.

Consumes a :class:`report_module.schema.ReportSchema`, renders each
``Block`` via the renderer registered for its ``kind``, writes the
combined markdown and the schema JSON to disk, and returns paths to
both artefacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from report_module import renderers
from report_module.report_file_saver import ReportFileSaver
from report_module.schema import Block, ReportSchema, SchemaLike

logger = logging.getLogger(__name__)


_METADATA_RENDERING_KEYS = frozenset(
    {"acceptance_summary_markdown", "acceptance_criteria"}
)


@dataclass(frozen=True)
class GenerateResult:
    markdown_path: Path
    json_path: Path
    markdown: str


class ReportGenerator:
    """Transform a :class:`ReportSchema` into a markdown + JSON report."""

    def __init__(self, file_saver: Optional[ReportFileSaver] = None) -> None:
        self._file_saver = file_saver or ReportFileSaver()

    def generate(
        self,
        schema: SchemaLike,
        output_dir: Union[str, Path],
    ) -> GenerateResult:
        normalized = _coerce_schema(schema)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        render_sections = _collapse_same_heading_blocks(normalized.sections)
        section_markdowns = [
            self._render_block(block, normalized.metadata) for block in render_sections
        ]
        rendered = [md for md in section_markdowns if md]

        release_md = _assemble_release_markdown(normalized, rendered)

        md_path = out_dir / f"report_{normalized.report_id}.md"
        json_path = out_dir / f"report_{normalized.report_id}.json"

        self._file_saver.write_markdown(release_md, md_path, strict=True)
        json_payload = normalized.to_dict()
        json_payload["metadata"] = {
            k: v
            for k, v in json_payload.get("metadata", {}).items()
            if k not in _METADATA_RENDERING_KEYS
        }
        self._file_saver.write_json(json_payload, json_path, strict=True)
        logger.info("Generated report: md=%s, json=%s", md_path, json_path)

        return GenerateResult(
            markdown_path=md_path,
            json_path=json_path,
            markdown=release_md,
        )

    @staticmethod
    def _render_block(block: Block, metadata: Mapping[str, Any]) -> str:
        """Invoke the renderer registered for ``block.kind``.

        Renderers own their own headings; the generator only routes.
        Unregistered kinds fall through to the generic table renderer.
        """
        renderer = renderers.get_renderer(block.kind)
        try:
            return renderer(block, metadata) or ""
        except Exception:
            logger.exception(
                "Renderer for kind '%s' failed on block '%s'", block.kind, block.slug
            )
            return ""


def _coerce_schema(schema: SchemaLike) -> ReportSchema:
    if isinstance(schema, ReportSchema):
        return schema
    if isinstance(schema, Mapping):
        return ReportSchema.from_dict(schema)
    if isinstance(schema, (list, tuple)):
        return ReportSchema.from_records(schema)
    raise TypeError(
        f"schema must be ReportSchema, Mapping or Sequence, got {type(schema).__name__}"
    )


def _assemble_release_markdown(schema: ReportSchema, sections: list[str]) -> str:
    header = (
        f"## Tenstorrent Model Release Summary: {schema.model_name} on {schema.device}"
    )
    visible_metadata = {
        k: v for k, v in schema.metadata.items() if k not in _METADATA_RENDERING_KEYS
    }
    metadata_json = json.dumps(visible_metadata, indent=4, default=str)
    metadata_block = (
        f"### Metadata: {schema.model_name} on {schema.device}\n"
        f"```json\n{metadata_json}\n```"
    )
    preamble = [header, metadata_block]
    acceptance_md = str(
        schema.metadata.get("acceptance_summary_markdown") or ""
    ).strip()
    if acceptance_md:
        preamble.append(acceptance_md)
    return "\n\n".join(preamble + sections)


def generate_report(
    schema: SchemaLike,
    output_dir: Union[str, Path],
    file_saver: Optional[ReportFileSaver] = None,
) -> GenerateResult:
    """Convenience functional wrapper around :class:`ReportGenerator`."""
    return ReportGenerator(file_saver=file_saver).generate(schema, output_dir)


def _collapse_same_heading_blocks(sections: List[Block]) -> List[Block]:
    """Merge consecutive blocks that would render under the same heading."""
    merged: List[Block] = []
    for block in sections:
        if merged and _has_same_heading(merged[-1], block):
            merged[-1] = _merge_blocks(merged[-1], block)
        else:
            merged.append(block)
    return merged


def _has_same_heading(a: Block, b: Block) -> bool:
    return (
        a.kind == b.kind
        and (a.title or "") == (b.title or "")
        and (a.id or "") == (b.id or "")
    )


def _merge_blocks(a: Block, b: Block) -> Block:
    records = _records_for_merge(a) + _records_for_merge(b)
    return Block(
        kind=a.kind,
        title=a.title,
        id=a.id,
        targets=dict(a.targets),
        checks=dict(a.checks),
        data={"records": records},
    )


def _records_for_merge(block: Block) -> List[Dict[str, Any]]:
    return [_flatten_one_level(record) for record in renderers._extract_records(block)]


def _flatten_one_level(record: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, Mapping):
            for sub_key, sub_value in value.items():
                out.setdefault(sub_key, sub_value)
        else:
            out[key] = value
    return out
