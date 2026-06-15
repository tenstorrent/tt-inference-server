# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.generator`` assembly + schema coercion."""

from __future__ import annotations

from pathlib import Path

import pytest

from report_module.generator import (
    ReportGenerator,
    _coerce_schema,
    _collapse_same_heading_blocks,
    generate_report,
)
from report_module.schema import Block, ReportSchema


def _schema(*blocks: Block) -> ReportSchema:
    return ReportSchema(
        metadata={"report_id": "r1", "model_name": "m", "device": "n300"},
        sections=list(blocks),
    )


class TestCoerceSchema:
    def test_passes_through_report_schema(self):
        s = _schema()
        assert _coerce_schema(s) is s

    def test_accepts_mapping(self):
        s = _coerce_schema({"metadata": {"report_id": "r"}, "sections": []})
        assert s.report_id == "r"

    def test_accepts_record_sequence(self):
        s = _coerce_schema([{"kind": "benchmarks", "model": "m", "device": "n300"}])
        assert s.sections[0].kind == "benchmarks"

    def test_rejects_other_types(self):
        with pytest.raises(TypeError):
            _coerce_schema(42)


class TestCollapseSameHeadingBlocks:
    def test_consecutive_same_heading_blocks_merge(self):
        a = Block(kind="benchmarks", title="B", data={"records": [{"x": 1}]})
        b = Block(kind="benchmarks", title="B", data={"records": [{"x": 2}]})
        merged = _collapse_same_heading_blocks([a, b])
        assert len(merged) == 1
        assert len(merged[0].data["records"]) == 2

    def test_different_headings_stay_separate(self):
        a = Block(kind="benchmarks", title="B", data={"records": [{"x": 1}]})
        b = Block(kind="benchmarks", title="Other", data={"records": [{"x": 2}]})
        assert len(_collapse_same_heading_blocks([a, b])) == 2


class TestGenerate:
    def test_writes_markdown_and_json(self, tmp_path: Path):
        result = generate_report(
            _schema(Block(kind="benchmarks", title="B", data={"records": [{"ttft": 1.0}]})),
            tmp_path,
        )
        assert result.markdown_path.exists()
        assert result.json_path.exists()
        assert result.json_path.parent == tmp_path / "data"

    def test_release_header_carries_model_and_device(self, tmp_path: Path):
        result = ReportGenerator().generate(_schema(), tmp_path)
        assert "Tenstorrent Model Release Summary: m on n300" in result.markdown

    def test_acceptance_markdown_keys_dropped_from_json(self, tmp_path: Path):
        schema = ReportSchema(
            metadata={
                "report_id": "r1",
                "model_name": "m",
                "device": "n300",
                "acceptance_summary_markdown": "### Acceptance Criteria",
                "acceptance_criteria": {"accepted": True},
            },
            sections=[],
        )
        result = ReportGenerator().generate(schema, tmp_path)
        # The acceptance markdown appears in the rendered report...
        assert "### Acceptance Criteria" in result.markdown
        # ...but the rendering-only metadata keys are stripped from JSON.
        payload = result.json_path.read_text()
        assert "acceptance_summary_markdown" not in payload
        assert "acceptance_criteria" not in payload

    def test_renderer_exception_does_not_abort_report(self, tmp_path: Path, monkeypatch):
        from report_module import renderers

        def _boom(block, metadata):
            raise RuntimeError("renderer blew up")

        monkeypatch.setattr(renderers, "get_renderer", lambda kind: _boom)
        # A failing renderer is logged and skipped; generate still succeeds.
        result = ReportGenerator().generate(
            _schema(Block(kind="benchmarks", data={"x": 1})), tmp_path
        )
        assert result.markdown_path.exists()

    def test_spec_test_summary_injected(self, tmp_path: Path):
        block = Block(
            kind="spec_tests",
            title="Spec",
            data={
                "test_name": "smoke",
                "attempts": 1,
                "success": True,
                "elapsed_seconds": 1.5,
                "description": "a smoke test",
            },
        )
        result = ReportGenerator().generate(_schema(block), tmp_path)
        md = result.markdown
        assert "## 📋 Summary" in md
        assert "## 🧪 Test Results" in md
        assert "smoke" in md
        assert "Success Rate" in md
