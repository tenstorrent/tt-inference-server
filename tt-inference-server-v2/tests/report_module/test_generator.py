# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.generator`` assembly + schema coercion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report_module.generator import (
    ReportGenerator,
    _build_spec_test_summary_markdown,
    _coerce_schema,
    _collapse_same_heading_blocks,
    _consolidate_eval_blocks,
    generate_report,
)
from report_module.schema import Block, ReportSchema


def test_spec_summary_distinguishes_skip_error_from_pass_fail():
    runs = [
        {"test_name": "A", "status": "pass", "success": True, "attempts": 1},
        {"test_name": "B", "status": "fail", "success": False, "attempts": 1},
        {
            "test_name": "C",
            "status": "error",
            "success": False,
            "attempts": 0,
            "error": {"message": "boom"},
        },
        {
            "test_name": "D",
            "status": "skip",
            "success": False,
            "reason": "no board",
            "attempts": 0,
        },
        {"test_name": "E", "status": "na", "reason": "no dataset", "attempts": 0},
    ]
    md = _build_spec_test_summary_markdown(runs, "2026-07-05")

    assert "| Passed | 1 |" in md
    assert "| Failed | 2 |" in md  # fail + error both blocking
    assert "| Skipped | 1 |" in md
    assert "| NA | 1 |" in md
    # Success rate excludes non-blocking skip/NA: 1 pass / (1 pass + 2 blocking).
    assert "| Success Rate | 33.3% |" in md
    # Distinct glyphs + reason surfaced in the description column.
    assert "⏭️" in md and "🟨" in md
    assert "no board" in md and "boom" in md


def test_spec_summary_legacy_rows_without_status():
    runs = [
        {"test_name": "A", "success": True, "attempts": 1},
        {"test_name": "B", "success": False, "attempts": 1},
    ]
    md = _build_spec_test_summary_markdown(runs, "2026-07-05")
    assert "| Passed | 1 |" in md
    assert "| Failed | 1 |" in md


def _eval_block(task: str, **data) -> Block:
    payload = {"task_name": task, "tolerance": 0.05, **data}
    return Block(
        kind="evals", task_type="llm", title=f"LLM Eval — {task}", data=payload
    )


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


class TestConsolidateEvalBlocks:
    def test_multiple_evals_merge_into_one_block(self):
        sections = [_eval_block("a", score=1.0), _eval_block("b", score=2.0)]
        merged = _consolidate_eval_blocks(sections)
        assert len(merged) == 1
        assert merged[0].title == "Accuracy Evaluations"
        assert merged[0].task_type is None
        records = merged[0].data["records"]
        assert [r["task_name"] for r in records] == ["a", "b"]

    def test_single_eval_block_is_left_unchanged(self):
        sections = [_eval_block("a", score=1.0)]
        assert _consolidate_eval_blocks(sections) is sections

    def test_non_eval_blocks_are_preserved_in_order(self):
        head = Block(kind="benchmarks", title="B", data={"records": [{"x": 1}]})
        tail = Block(kind="spec_tests", title="S", data={"records": [{"y": 2}]})
        merged = _consolidate_eval_blocks(
            [head, _eval_block("a"), _eval_block("b"), tail]
        )
        assert [b.kind for b in merged] == ["benchmarks", "evals", "spec_tests"]
        assert merged[1].title == "Accuracy Evaluations"


class TestGenerate:
    def test_writes_markdown_and_json(self, tmp_path: Path):
        result = generate_report(
            _schema(
                Block(kind="benchmarks", title="B", data={"records": [{"ttft": 1.0}]})
            ),
            tmp_path,
        )
        assert result.markdown_path.exists()
        assert result.json_path.exists()
        assert result.json_path.parent == tmp_path / "data"

    def test_release_header_carries_model_and_device(self, tmp_path: Path):
        result = ReportGenerator().generate(_schema(), tmp_path)
        assert "Tenstorrent Model Release Summary: m on n300" in result.markdown

    def test_evals_render_as_single_table_with_note(self, tmp_path: Path):
        result = ReportGenerator().generate(
            _schema(
                _eval_block("meta_ifeval", gpu_reference_score=81.38, accuracy_check=2),
                _eval_block("longbench_code_e", score=0.0, accuracy_check=3),
            ),
            tmp_path,
        )
        md = result.markdown
        assert md.count("### Accuracy Evaluations for m on n300") == 1
        assert "LLM Eval — meta_ifeval" not in md
        assert "Note: The ratio to published scores" in md
        # Both tasks live in the one consolidated table.
        assert "meta_ifeval" in md and "longbench_code_e" in md

    def test_acceptance_criteria_promoted_to_top_level_json(self, tmp_path: Path):
        schema = ReportSchema(
            metadata={
                "report_id": "r1",
                "model_name": "m",
                "device": "n300",
                "acceptance_summary_markdown": "### Acceptance Criteria",
                "acceptance_criteria": True,
                "acceptance_blockers": {},
                "acceptance_criteria_metadata": {
                    "enforcement_result": "PASS",
                    "model_status": "COMPLETE",
                },
            },
            sections=[],
        )
        result = ReportGenerator().generate(schema, tmp_path)
        # The acceptance section is rendered, but kept out of the inline
        # metadata code-block...
        assert "### Acceptance Criteria" in result.markdown
        metadata_block = result.markdown.split("### Acceptance Criteria")[0]
        assert "acceptance_summary_markdown" not in metadata_block
        assert "acceptance_criteria" not in metadata_block
        # ...while the JSON promotes the acceptance fields to top-level (v1 shape)
        # and removes them from metadata.
        payload = json.loads(result.json_path.read_text())
        assert payload["acceptance_criteria"] is True
        assert payload["acceptance_blockers"] == {}
        assert payload["acceptance_criteria_metadata"]["model_status"] == "COMPLETE"
        assert payload["acceptance_summary_markdown"] == "### Acceptance Criteria"
        assert "acceptance_criteria" not in payload["metadata"]

    def test_renderer_exception_does_not_abort_report(
        self, tmp_path: Path, monkeypatch
    ):
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
