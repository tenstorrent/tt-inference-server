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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from report_module import renderers
from report_module.acceptance_criteria import ACCEPTANCE_EXPORT_KEYS
from report_module.markdown_table import build_markdown_table
from report_module.report_file_saver import ReportFileSaver
from report_module.schema import Block, ReportSchema, SchemaLike
from report_module.status import STATUS_GLYPHS as _STATUS_GLYPHS
from report_module.status import TestStatus

logger = logging.getLogger(__name__)


_METADATA_RENDERING_KEYS = frozenset(ACCEPTANCE_EXPORT_KEYS)

_EVALS_KIND = "evals"
_CONSOLIDATED_EVALS_TITLE = "Accuracy Evaluations"

# Divider placed between the preamble and each top-level result section so
# adjacent sections (e.g. a benchmark table and the "📋 Summary" block) are
# visually separated in both raw CI logs and rendered markdown.
_SECTION_SEPARATOR = "\n\n---\n\n"


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
        render_sections = _consolidate_eval_blocks(render_sections)
        section_markdowns = [
            self._render_block(block, normalized.metadata) for block in render_sections
        ]
        rendered_pairs = list(zip(render_sections, section_markdowns))
        rendered_pairs = [(block, md) for block, md in rendered_pairs if md]

        release_md = _assemble_release_markdown(normalized, rendered_pairs)

        md_path = out_dir / f"report_{normalized.report_id}.md"
        json_path = out_dir / "data" / f"report_data_{normalized.report_id}.json"

        self._file_saver.write_markdown(release_md, md_path, strict=True)
        json_payload = normalized.to_dict()
        metadata = json_payload.get("metadata", {})
        for key in ACCEPTANCE_EXPORT_KEYS:
            if key in metadata:
                json_payload[key] = metadata.pop(key)
        self._file_saver.write_json(json_payload, json_path, strict=True)
        self._write_report_data(out_dir, json_payload)
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

    def _write_report_data(
        self,
        out_dir: Path,
        json_payload: Dict[str, Any],
    ) -> None:
        """Write report data JSON consumed by CI aggregation steps."""
        report_id = str(json_payload.get("metadata", {}).get("report_id") or "")
        if not report_id:
            logger.warning("Skipping report data output: missing report_id")
            return

        data_path = out_dir / "data" / f"report_data_{report_id}.json"
        self._file_saver.write_json(json_payload, data_path, strict=True)


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


def _assemble_release_markdown(
    schema: ReportSchema, rendered_pairs: list[tuple[Block, str]]
) -> str:
    header = (
        f"## Tenstorrent Model Release Summary: {schema.model_name} on {schema.device}"
    )
    visible_metadata = {
        k: v for k, v in schema.metadata.items() if k not in _METADATA_RENDERING_KEYS
    }
    metadata_json = json.dumps(visible_metadata, indent=4, default=str)
    metadata_block = (
        f"### Metadata: {schema.model_name} on {schema.device}\n\n"
        f"```json\n{metadata_json}\n```"
    )
    preamble = [header, metadata_block]
    acceptance_md = str(
        schema.metadata.get("acceptance_summary_markdown") or ""
    ).strip()
    if acceptance_md:
        preamble.append(acceptance_md)

    sections = _inject_spec_test_summary(rendered_pairs, schema.metadata)
    preamble_md = "\n\n".join(preamble)
    return _SECTION_SEPARATOR.join([preamble_md] + sections)


def _inject_spec_test_summary(
    rendered_pairs: list[tuple[Block, str]], metadata: Mapping[str, Any]
) -> list[str]:
    runs: List[Mapping[str, Any]] = []
    first_spec_idx: Optional[int] = None
    for idx, (block, _md) in enumerate(rendered_pairs):
        block_runs = _spec_test_runs(block)
        if block_runs and first_spec_idx is None:
            first_spec_idx = idx
        runs.extend(block_runs)

    sections = [md for _block, md in rendered_pairs]
    if first_spec_idx is None or not runs:
        return sections

    summary_md = _build_spec_test_summary_markdown(
        runs, str(metadata.get("generated_at") or "")
    )
    if not summary_md:
        return sections
    return sections[:first_spec_idx] + [summary_md] + sections[first_spec_idx:]


def _spec_test_runs(block: Block) -> List[Mapping[str, Any]]:
    data = block.data
    if not isinstance(data, Mapping):
        return []
    if "test_name" in data and "attempts" in data:
        return [data]
    records = data.get("records") if isinstance(data, Mapping) else None
    if isinstance(records, list):
        return [
            record
            for record in records
            if isinstance(record, Mapping)
            and "test_name" in record
            and "attempts" in record
        ]
    return []


def _run_status(run: Mapping[str, Any]) -> TestStatus:
    """Resolve a run's :class:`TestStatus`, falling back to legacy fields."""
    resolved = TestStatus.from_value(run.get("status"))
    if resolved is not None:
        return resolved
    return TestStatus.from_legacy(run.get("success"), skipped=bool(run.get("skipped")))


def _status_cell(status: TestStatus) -> str:
    return f"{_STATUS_GLYPHS.get(status, '❌')} {status.value.upper()}"  # noqa: E501


def _run_description(run: Mapping[str, Any], status: TestStatus) -> str:
    """Description column: append the reason/error for non-pass outcomes."""
    description = str(run.get("description") or "")
    if status is TestStatus.SKIP or status is TestStatus.NA:
        reason = str(run.get("reason") or "")
        if reason:
            return f"{description} — {status.value.upper()}: {reason}".lstrip(" —")
    if status is TestStatus.ERROR:
        error = run.get("error")
        message = error.get("message") if isinstance(error, Mapping) else None
        if message:
            return f"{description} — ERROR: {message}".lstrip(" —")
    return description


def _build_spec_test_summary_markdown(
    runs: Sequence[Mapping[str, Any]], generated_at: str
) -> str:
    """Render the 📋 Summary + 🧪 Test Results section for spec tests."""
    if not runs:
        return ""

    total = len(runs)
    statuses = [_run_status(run) for run in runs]
    passed = sum(1 for s in statuses if s is TestStatus.PASS)
    failed = sum(1 for s in statuses if s.is_blocking)
    skipped = sum(1 for s in statuses if s is TestStatus.SKIP)
    na = sum(1 for s in statuses if s is TestStatus.NA)
    attempted = passed + failed
    total_duration = sum(_coerce_float(r.get("elapsed_seconds")) for r in runs)
    total_attempts = sum(_coerce_int(r.get("attempts")) for r in runs)
    # Success rate excludes non-blocking skips/NA — they weren't graded.
    gradable = passed + failed
    success_rate = (passed / gradable * 100.0) if gradable else 0.0

    summary_rows = [
        ("Total Tests", str(total)),
        ("Passed", str(passed)),
        ("Failed", str(failed)),
        ("Skipped", str(skipped)),
        ("NA", str(na)),
        ("Attempted", str(attempted)),
        ("Success Rate", f"{success_rate:.1f}%"),
        ("Total Duration", f"{total_duration:.2f}s"),
        ("Total Attempts", str(total_attempts)),
        ("Generated", generated_at or "-"),
    ]

    summary_table = build_markdown_table(
        [{"Metric": metric, "Value": value} for metric, value in summary_rows]
    )

    results_table = build_markdown_table(
        [
            {
                "Status": _status_cell(status),
                "Test Name": str(run.get("test_name") or ""),
                "Duration": f"{_coerce_float(run.get('elapsed_seconds')):.2f}s",
                "Attempts": str(_coerce_int(run.get("attempts"))),
                "Description": _run_description(run, status),
            }
            for run, status in zip(runs, statuses)
        ]
    )

    return f"## 📋 Summary\n\n{summary_table}\n\n## 🧪 Test Results\n\n{results_table}"


def _coerce_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def generate_report(
    schema: SchemaLike,
    output_dir: Union[str, Path],
    file_saver: Optional[ReportFileSaver] = None,
) -> GenerateResult:
    """Convenience functional wrapper around :class:`ReportGenerator`."""
    return ReportGenerator(file_saver=file_saver).generate(schema, output_dir)


def _consolidate_eval_blocks(sections: List[Block]) -> List[Block]:
    """Merge every ``evals`` block into one consolidated table block."""
    eval_blocks = [block for block in sections if block.kind == _EVALS_KIND]
    if len(eval_blocks) <= 1:
        return sections

    records: List[Dict[str, Any]] = []
    for block in eval_blocks:
        records.extend(renderers._extract_records(block))
    merged = Block(
        kind=_EVALS_KIND,
        title=_CONSOLIDATED_EVALS_TITLE,
        task_type=None,
        id=eval_blocks[0].id,
        targets={},
        data={"records": records},
    )

    out: List[Block] = []
    inserted = False
    for block in sections:
        if block.kind != _EVALS_KIND:
            out.append(block)
        elif not inserted:
            out.append(merged)
            inserted = True
    return out


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
        and (a.task_type or "") == (b.task_type or "")
        and (a.title or "") == (b.title or "")
        and (a.id or "") == (b.id or "")
    )


def _merge_blocks(a: Block, b: Block) -> Block:
    records = _records_for_merge(a) + _records_for_merge(b)
    return Block(
        kind=a.kind,
        title=a.title,
        task_type=a.task_type,
        id=a.id,
        targets=dict(a.targets),
        data={"records": records},
    )


def _records_for_merge(block: Block) -> List[Dict[str, Any]]:
    return [_flatten_one_level(record) for record in renderers._extract_records(block)]


def _flatten_one_level(record: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, Mapping) and all(
            not isinstance(v, Mapping) for v in value.values()
        ):
            for sub_key, sub_value in value.items():
                out.setdefault(sub_key, sub_value)
        else:
            out[key] = value
    return out
