# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit tests for the training driver's acceptance-export path.

Focus is ``launchers/run_training_test.py``'s ``_write_report`` (the 2a wiring):
feeding the ``spec_tests`` loss records through ``acceptance_criteria_check`` so
the ``.md`` gains an ``### Acceptance Criteria`` section and the JSON gains a
top-level ``acceptance_criteria`` key — plus ``_read_acceptance_inputs``, which
pulls ``model_status`` / ``known_issues`` out of the runtime-model-spec JSON.

No server is touched: these exercise the pure report/grading path only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from launchers.run_training_test import _read_acceptance_inputs, _write_report

MODEL = "Llama-3.1-8B"
DEVICE = "p150"


def _rec(test_name: str, status: str) -> Dict[str, Any]:
    """One ``spec_tests``-shaped record like the loss grader emits."""
    return {
        "kind": "spec_tests",
        "model": MODEL,
        "device": DEVICE,
        "test_name": test_name,
        "status": status,
        "attempts": 1,
        "elapsed_seconds": 1.5,
        "description": f"{test_name} {status}",
    }


def _read_report(output_dir: Path) -> tuple[str, Dict[str, Any]]:
    md_files = list(output_dir.glob("report_*.md"))
    json_files = list((output_dir / "data").glob("report_data_*.json"))
    assert len(md_files) == 1, f"expected one .md, got {md_files}"
    assert len(json_files) == 1, f"expected one JSON, got {json_files}"
    return md_files[0].read_text(), json.loads(json_files[0].read_text())


# --- _write_report: acceptance export -------------------------------------


def test_write_report_all_pass_accepts(tmp_path):
    records: List[Dict[str, Any]] = [
        _rec("train_loss@step5", "pass"),
        _rec("final_train_loss_threshold", "pass"),
    ]
    accepted = _write_report(
        tmp_path,
        MODEL,
        DEVICE,
        records,
        {"verdict": "PASS", "summary": "2/2"},
        model_status="EXPERIMENTAL",
        known_issues=[],
    )
    assert accepted is True

    md, payload = _read_report(tmp_path)
    # .md gains the acceptance section (generator only renders it when the
    # export is present in metadata).
    assert "### Acceptance Criteria" in md
    # JSON gains the hoisted top-level acceptance keys (generator pops them out
    # of metadata into the payload root).
    assert payload["acceptance_criteria"] is True
    assert payload["acceptance_blockers"] == {}
    assert "acceptance_criteria_metadata" in payload
    # The keys are hoisted, not left duplicated inside metadata.
    assert "acceptance_criteria" not in payload.get("metadata", {})


def test_write_report_failure_blocks_with_per_test_key(tmp_path):
    records = [
        _rec("train_loss@step5", "pass"),
        _rec("final_train_loss_threshold", "fail"),
    ]
    accepted = _write_report(
        tmp_path,
        MODEL,
        DEVICE,
        records,
        {"verdict": "FAIL", "summary": "1/2"},
        model_status="EXPERIMENTAL",
        known_issues=[],
    )
    assert accepted is False

    md, payload = _read_report(tmp_path)
    assert payload["acceptance_criteria"] is False
    # One block per record -> per-test blocker key, not a single opaque one.
    assert "spec.spec_tests:final_train_loss_threshold" in payload["acceptance_blockers"]
    assert "spec.spec_tests:train_loss@step5" not in payload["acceptance_blockers"]
    # The failing test name surfaces in the acceptance section's blocker list.
    assert "final_train_loss_threshold" in md


def test_write_report_spec_tests_enforced_even_when_experimental(tmp_path):
    # Spec tests have no status gate: a failing loss check blocks acceptance
    # regardless of an EXPERIMENTAL model status (the launcher relies on this).
    accepted = _write_report(
        tmp_path,
        MODEL,
        DEVICE,
        [_rec("train_loss@step5", "fail")],
        {"verdict": "FAIL"},
        model_status="EXPERIMENTAL",
        known_issues=[],
    )
    assert accepted is False


def test_write_report_layout_has_single_summary_table(tmp_path):
    # Each record is its own (empty-rendering) block; the generator injects one
    # 🧪 Test Results summary. Guard against per-record tables creeping back.
    records = [_rec("train_loss@step5", "pass"), _rec("val_loss@step5", "pass")]
    _write_report(
        tmp_path,
        MODEL,
        DEVICE,
        records,
        {"verdict": "PASS"},
        model_status="EXPERIMENTAL",
        known_issues=[],
    )
    md, _ = _read_report(tmp_path)
    assert md.count("## 🧪 Test Results") == 1
    assert md.count("## 📋 Summary") == 1
    # Both tests appear as rows in that single table.
    assert "train_loss@step5" in md and "val_loss@step5" in md


# --- _read_acceptance_inputs ----------------------------------------------


def test_read_acceptance_inputs_from_spec(tmp_path):
    spec_path = tmp_path / "spec.json"
    known = [{"workflow_type": "TRAINING_TESTS", "reason": "flaky", "task_name": None}]
    spec_path.write_text(
        json.dumps(
            {
                "runtime_model_spec": {
                    "status": "FUNCTIONAL",
                    "device_model_spec": {"known_issues": known},
                },
                "runtime_config": {},
            }
        )
    )
    status, issues = _read_acceptance_inputs(str(spec_path))
    assert status == "FUNCTIONAL"
    assert issues == known


def test_read_acceptance_inputs_missing_path_falls_back():
    status, issues = _read_acceptance_inputs(None)
    assert status == "EXPERIMENTAL"
    assert issues == []


def test_read_acceptance_inputs_unreadable_falls_back(tmp_path):
    status, issues = _read_acceptance_inputs(str(tmp_path / "does_not_exist.json"))
    assert status == "EXPERIMENTAL"
    assert issues == []


def test_read_acceptance_inputs_missing_status_defaults(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"runtime_model_spec": {}, "runtime_config": {}}))
    status, issues = _read_acceptance_inputs(str(spec_path))
    assert status == "EXPERIMENTAL"
    assert issues == []
