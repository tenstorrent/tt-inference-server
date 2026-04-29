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
from typing import Any, Mapping, Optional, Union

from report_module import renderers
from report_module.report_file_saver import ReportFileSaver
from report_module.schema import Block, ReportSchema, SchemaLike

logger = logging.getLogger(__name__)


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

        section_markdowns = [
            self._render_block(block, normalized.metadata)
            for block in normalized.sections
        ]
        rendered = [md for md in section_markdowns if md]

        release_md = _assemble_release_markdown(normalized, rendered)

        md_path = out_dir / f"report_{normalized.report_id}.md"
        json_path = out_dir / f"report_{normalized.report_id}.json"

        self._file_saver.write_markdown(release_md, md_path, strict=True)
        self._file_saver.write_json(normalized.to_dict(), json_path, strict=True)
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
        """
        renderer = renderers.get_renderer(block.kind)
        if renderer is None:
            logger.warning(
                "No renderer registered for kind '%s'; skipping block '%s'",
                block.kind,
                block.slug,
            )
            return ""
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
    metadata_json = json.dumps(schema.metadata, indent=4, default=str)
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
