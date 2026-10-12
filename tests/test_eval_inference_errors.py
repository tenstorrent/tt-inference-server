# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""An eval whose requests errored is not a measurement, and a qualification
pass never rests on a waived must-eval.

Shield run 38216483981 (Mistral-Small-3.1, humaneval_instruct): every request
was refused with HTTP 400, lm-eval scored the error text as 164 wrong answers
(pass@1 0.00 at 0.089 s/task), and the EXPERIMENTAL tier waived the failed
check as informational, so acceptance read PASS.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from report_module.acceptance_criteria import (
    CATEGORY_EVALS,
    acceptance_criteria_check,
    build_acceptance_export,
)
from report_module.schema import Block, ReportSchema
from test_module._test_common import ReportCheckTypes, TestStatus
from test_module.llm_tests import llm_eval_tests as mod

ERROR = "__INFERENCE_ERROR__: ClientResponseError(status=400, message='Bad Request')"


def _attempt(tmp_path: Path, answers):
    """An lm-eval attempt dir: results JSON plus one per-sample log."""
    model_spec = SimpleNamespace(model_id="id_quetzal_m_p300x2", hf_model_repo="org/m")
    out = tmp_path / f"eval_{model_spec.model_id}" / "org__m"
    out.mkdir(parents=True)
    (out / "results_2026-10-12T00-00-00.json").write_text(
        json.dumps({"results": {"t": {"pass@1,create_test": 0.0}}})
    )
    rows = [
        {
            "doc_id": i,
            # The prompt may mention anything; only responses are inspected.
            "arguments": {"gen_args_0": {"arg_0": ERROR}},
            "resps": [[answer]],
            "filtered_resps": [[answer]],
        }
        for i, answer in enumerate(answers)
    ]
    (out / "samples_t_2026-10-12T00-00-00.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    return tmp_path, model_spec


def test_inference_errors_are_counted_from_responses_only(tmp_path):
    root, model_spec = _attempt(tmp_path, [ERROR, "def f(): pass", ERROR])
    files = mod.discover_eval_results(root, model_spec)
    assert mod.count_inference_errors(files) == (2, 3)


def test_no_errors_is_zero(tmp_path):
    root, model_spec = _attempt(tmp_path, ["a", "b"])
    files = mod.discover_eval_results(root, model_spec)
    assert mod.count_inference_errors(files) == (0, 2)


def _graded_block(score=0.0):
    return Block(
        kind="evals",
        title="LLM Eval — t",
        data={
            "task_name": "t",
            "score": score,
            "accuracy_check": ReportCheckTypes.FAIL,
            "priority": "must",
        },
    )


@pytest.mark.parametrize("answers", [[ERROR] * 4, [ERROR, "def f(): pass"]])
def test_an_errored_eval_is_graded_incomplete_not_scored(
    tmp_path, monkeypatch, answers
):
    root, model_spec = _attempt(tmp_path, answers)
    ctx = SimpleNamespace(model_spec=model_spec)
    monkeypatch.setattr(mod, "load_eval_results", lambda files: ({}, {}))
    monkeypatch.setattr(mod, "blocks_for_task", lambda *a, **k: [_graded_block()])
    (block,) = mod._score_task(
        ctx, SimpleNamespace(task_name="t"), output_path=root, rc=0, elapsed_seconds=1
    )
    assert block.data["incomplete"] is True
    assert block.data["status"] == TestStatus.FAIL.value
    assert block.data["accuracy_check"] == ReportCheckTypes.FAIL
    assert block.data["inference_error_samples"] == answers.count(ERROR)
    assert block.data["samples_logged"] == len(answers)


def test_a_clean_eval_is_left_to_the_grader(tmp_path, monkeypatch):
    root, model_spec = _attempt(tmp_path, ["def f(): pass"])
    ctx = SimpleNamespace(model_spec=model_spec)
    monkeypatch.setattr(mod, "load_eval_results", lambda files: ({}, {}))
    graded = _graded_block(score=88.0)
    graded.data["accuracy_check"] = ReportCheckTypes.PASS
    monkeypatch.setattr(mod, "blocks_for_task", lambda *a, **k: [graded])
    (block,) = mod._score_task(
        ctx, SimpleNamespace(task_name="t"), output_path=root, rc=0, elapsed_seconds=1
    )
    assert "incomplete" not in block.data
    assert block.data["accuracy_check"] == ReportCheckTypes.PASS


# --- acceptance ---------------------------------------------------------------


def _check(*blocks, model_status="EXPERIMENTAL", known_issues=None):
    accepted, blockers, categories = acceptance_criteria_check(
        ReportSchema(metadata={"report_id": "r"}, sections=list(blocks)),
        known_issues=known_issues,
        model_status=model_status,
    )
    export = build_acceptance_export(accepted, blockers, categories, model_status)
    return accepted, blockers, export


def _eval(task, check, priority="must", **extra):
    return Block(
        kind="evals",
        title=f"LLM Eval — {task}",
        data={
            "task_name": task,
            "score": 50.0,
            "accuracy_check": check,
            "priority": priority,
            **extra,
        },
    )


@pytest.mark.parametrize("model_status", ["EXPERIMENTAL", "FUNCTIONAL", None])
def test_an_incomplete_eval_blocks_under_every_tier(model_status):
    block = _eval(
        "t",
        ReportCheckTypes.FAIL,
        status=TestStatus.FAIL.value,
        incomplete=True,
        reason="164/164 samples are inference errors",
    )
    accepted, blockers, export = _check(block, model_status=model_status)
    assert accepted is False
    assert any("incomplete" in message for message in blockers.values())
    assert (
        export["acceptance_criteria_metadata"]["qualification"]["strict_pass"] is False
    )


def test_an_incomplete_eval_is_not_waived_by_a_known_issue_or_should_priority():
    block = _eval("t", ReportCheckTypes.FAIL, priority="should", incomplete=True)
    issues = [{"workflow_type": "EVALS", "task_name": "t", "reason": "reviewed"}]
    accepted, _, _ = _check(block, known_issues=issues)
    assert accepted is False


def test_experimental_waiver_passes_acceptance_but_never_qualification():
    # The audit's case: a must-eval FAIL that EXPERIMENTAL makes informational.
    accepted, blockers, export = _check(
        _eval("ifeval", ReportCheckTypes.PASS), _eval("mbpp", ReportCheckTypes.FAIL)
    )
    assert accepted is True and not blockers
    qualification = export["acceptance_criteria_metadata"]["qualification"]
    assert qualification["strict_pass"] is False
    assert list(qualification["waived_must_evals"]) and all(
        "informational" in m for m in qualification["waived_must_evals"].values()
    )
    assert (
        "Qualification (every must-eval passes unwaived): "
        in (export["acceptance_summary_markdown"])
    )


def test_a_known_issue_waived_must_eval_is_not_a_qualification_pass():
    issues = [{"workflow_type": "EVALS", "task_name": "mbpp", "reason": "reviewed"}]
    accepted, _, export = _check(
        _eval("mbpp", ReportCheckTypes.FAIL),
        model_status="FUNCTIONAL",
        known_issues=issues,
    )
    assert accepted is True
    assert (
        export["acceptance_criteria_metadata"]["qualification"]["strict_pass"] is False
    )


def test_a_should_priority_waiver_does_not_block_qualification():
    accepted, _, export = _check(
        _eval("ifeval", ReportCheckTypes.PASS),
        _eval("extra", ReportCheckTypes.FAIL, priority="should"),
        model_status="FUNCTIONAL",
    )
    assert accepted is True
    assert (
        export["acceptance_criteria_metadata"]["qualification"]["strict_pass"] is True
    )


def test_every_must_eval_passing_is_a_qualification_pass():
    accepted, _, export = _check(
        _eval("ifeval", ReportCheckTypes.PASS), _eval("gpqa", ReportCheckTypes.PASS)
    )
    assert accepted is True
    qualification = export["acceptance_criteria_metadata"]["qualification"]
    assert qualification == {"strict_pass": True, "waived_must_evals": {}}
    _, _, categories = acceptance_criteria_check(
        ReportSchema(
            metadata={"report_id": "r"}, sections=[_eval("x", ReportCheckTypes.PASS)]
        )
    )
    assert next(c for c in categories if c.name == CATEGORY_EVALS).waived_must == {}
