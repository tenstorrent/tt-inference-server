# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.summary_report`` discovery + aggregation."""

from __future__ import annotations

import json
from pathlib import Path

from report_module.schema import Block, ReportSchema
from workflow_module.summary_report import (
    _compact_timestamp,
    build_summary_schema,
    discover_run_reports,
    generate_summary_report,
    load_run_reports,
    summarize_container,
)


def _run_schema(model="m", device="n300", ttft=10.0, *, generated_at="2026-05-05 12:00:00"):
    return ReportSchema(
        metadata={
            "report_id": "r",
            "model_name": model,
            "device": device,
            "generated_at": generated_at,
        },
        sections=[
            Block(kind="benchmarks", title="Bench", task_type="image",
                  data={"ttft": ttft, "inference_steps_per_second": 16.0})
        ],
    )


def _write_run_report(container: Path, run_idx: int, schema: ReportSchema) -> Path:
    run_dir = container / f"run_{run_idx:02d}"
    run_dir.mkdir(parents=True)
    path = run_dir / f"report_{run_idx}.json"
    path.write_text(json.dumps(schema.to_dict()), encoding="utf-8")
    return path


class TestCompactTimestamp:
    def test_parses_space_separated(self):
        assert _compact_timestamp("2026-05-05 12:00:00") == "20260505_120000"

    def test_parses_iso_t(self):
        assert _compact_timestamp("2026-05-05T12:00:00") == "20260505_120000"

    def test_unparseable_strips_to_alnum(self):
        assert _compact_timestamp("garbage!!") == "garbage"

    def test_empty_is_unknown(self):
        assert _compact_timestamp("") == "unknown"


class TestDiscoverAndLoad:
    def test_discover_finds_sorted_run_reports(self, tmp_path: Path):
        _write_run_report(tmp_path, 2, _run_schema())
        _write_run_report(tmp_path, 1, _run_schema())
        found = discover_run_reports(tmp_path)
        assert [p.parent.name for p in found] == ["run_01", "run_02"]

    def test_load_skips_unreadable_json(self, tmp_path: Path):
        good = _write_run_report(tmp_path, 1, _run_schema())
        bad_dir = tmp_path / "run_02"
        bad_dir.mkdir()
        bad = bad_dir / "report_2.json"
        bad.write_text("{not valid json", encoding="utf-8")
        schemas = load_run_reports([good, bad])
        assert len(schemas) == 1


class TestBuildSummarySchema:
    def test_none_when_no_benchmarks(self):
        schema = ReportSchema(
            metadata={"report_id": "r", "model_name": "m", "device": "n300"},
            sections=[Block(kind="evals", data={"score": 1})],
        )
        assert build_summary_schema([schema]) is None

    def test_aggregates_runs_into_summary_block(self):
        schemas = [_run_schema(ttft=10.0), _run_schema(ttft=20.0)]
        summary = build_summary_schema(schemas)
        assert summary is not None
        assert summary.metadata["summary"] is True
        assert summary.metadata["num_runs_total"] == 2
        assert summary.metadata["report_id"].startswith("summary_m_n300_")
        block = summary.sections[0]
        assert block.kind == "benchmarks"
        assert block.data["num_runs"] == 2
        assert "summary of 2 runs" in block.title

    def test_latest_generated_at_wins(self):
        schemas = [
            _run_schema(generated_at="2026-05-05 12:00:00"),
            _run_schema(generated_at="2026-05-06 09:00:00"),
        ]
        summary = build_summary_schema(schemas)
        assert summary.metadata["generated_at"] == "2026-05-06 09:00:00"


class TestSummarizeContainer:
    def test_end_to_end_writes_report(self, tmp_path: Path):
        container = tmp_path / "m_n300_benchmarks"
        _write_run_report(container, 1, _run_schema(ttft=10.0))
        _write_run_report(container, 2, _run_schema(ttft=12.0))
        result = summarize_container(container, container / "summary")
        assert result is not None
        assert result.markdown_path.exists()
        assert "summary of 2 runs" in result.markdown

    def test_no_reports_returns_none(self, tmp_path: Path):
        assert summarize_container(tmp_path / "empty") is None

    def test_generate_summary_report_adds_acceptance_metadata(self, tmp_path: Path):
        result = generate_summary_report([_run_schema(), _run_schema()], tmp_path)
        assert result is not None
        # Acceptance section is injected into the rendered markdown.
        assert "Acceptance Criteria" in result.markdown
