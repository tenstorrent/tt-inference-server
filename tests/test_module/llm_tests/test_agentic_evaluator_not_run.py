# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Offline Harbor coverage for optional tau3 evaluator metadata."""

import json
import logging
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_module.parsers.agentic import AgenticEvalParser
from report_module.acceptance_criteria import (
    acceptance_criteria_check,
    strict_qualification,
)
from report_module.generator import generate_report
from report_module.schema import ReportSchema
from test_module.llm_tests import agentic_eval_tests as bridge
from workflow_module.engine_types import ReportCheckTypes, WorkflowVenvType


@pytest.fixture
def job(tmp_path):
    def write(*flags):
        names = [f"trial-{i}" for i in range(len(flags))]
        for name, flag in zip(names, flags):
            verifier = tmp_path / name / "verifier"
            verifier.mkdir(parents=True)
            (verifier / "result.json").write_text(
                json.dumps({"status": "mismatch", "used_tau2_evaluator": flag})
            )
        summary = dict(
            n_trials=len(flags),
            n_errors=0 if flags else 5,
            metrics=[{"mean": 0.0}] if flags else [],
            reward_stats={"reward": {"0.0": names}},
        )
        raw = dict(n_total_trials=len(flags), stats={"evals": {"tau3_bench": summary}})
        path = tmp_path / "result.json"
        path.write_text(json.dumps(raw))
        return raw, path

    return write


def _parse(raw, path, task_name="tau3_bench", score=None):
    parser = AgenticEvalParser(task_name=task_name, score=score, result_path=path)
    return parser.parse(raw)


def test_all_evaluator_not_run_reference_verdict(job):
    raw, path = job(False, False)
    score = SimpleNamespace(
        published_score=26.8,
        gpu_reference_score=35.1,
        published_score_ref="https://example.com",
        tolerance=0.05,
    )
    block = _parse(raw, path, "tau3_bench_banking", score)
    assert block.data["n_evaluator_not_run"] == block.targets["n_trials"] == 2
    schema = ReportSchema(metadata={"report_id": "offline"}, sections=[block])
    accepted, blockers, categories = acceptance_criteria_check(schema)
    # These exact verdicts equal the pre-change behavior.
    assert accepted is False
    assert blockers == {"evals:LLM Eval — tau3_bench_banking": "Accuracy check failed."}
    assert strict_qualification(accepted, categories) == {
        "strict_pass": False,
        "waived_must_evals": {},
        "ungraded_must_evals": {},
    }


def test_non_tau3_data_and_targets_are_unchanged_with_result_path(job):
    raw, path = job(None)
    block = _parse(raw, path, "terminal_bench_2")
    assert block.data == {
        "task_name": "terminal_bench_2",
        "tolerance": None,
        "published_score": None,
        "published_score_ref": None,
        "gpu_reference_score": None,
        "score": 0.0,
        "ratio_to_published": "N/A",
        "ratio_to_reference": "N/A",
        "accuracy_check": ReportCheckTypes.NA,
        "mean_seconds_per_task": None,
        "priority": "must",
    }
    assert block.targets == {
        "task_name": "terminal_bench_2",
        "job_result_path": str(path),
        "n_trials": 1,
        "n_errors": 0,
        "n_total_trials": 1,
        "pass_at_1": 0.0,
    }


@pytest.mark.parametrize("used", [False, True])
def test_list_status_does_not_affect_boolean_flag(job, used):
    raw, path = job(used)
    result = path.parent / "trial-0/verifier/result.json"
    result.write_text(json.dumps({"status": ["mismatch"], "used_tau2_evaluator": used}))
    block = _parse(raw, path)
    assert block.data["n_evaluator_not_run"] == int(not used)
    assert block.targets["n_evaluator_not_run"] == int(not used)


def test_deduplication_and_first_eval_scope(job):
    raw, path = job(True, False)
    summary = raw["stats"]["evals"]["tau3_bench"]
    raw["stats"]["evals"]["other"] = deepcopy(summary)
    summary["reward_stats"]["reward"]["0.0"] = ["trial-0", "trial-0"]
    assert _parse(raw, path).data["n_evaluator_not_run"] == 0
    summary["reward_stats"]["reward"]["0.0"] = ["trial-1", "trial-1"]
    assert _parse(raw, path).data["n_evaluator_not_run"] == 1


@pytest.mark.parametrize("name", ["nested/trial", "nested\\trial", "../outside_trial"])
def test_existing_traversal_paths_are_ignored(job, name):
    raw, path = job(True)
    result = path.parent / name / "verifier/result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"used_tau2_evaluator": False}))
    raw["stats"]["evals"]["tau3_bench"]["reward_stats"]["reward"]["0.0"].append(name)
    assert _parse(raw, path).data["n_evaluator_not_run"] == 0


@pytest.mark.parametrize(
    "kind", ["deep_nesting", "oversize", "directory", "fifo", "outside_symlink"]
)
def test_unsafe_metadata_is_skipped_per_file(job, kind):
    raw, path = job(False, True)
    result = path.parent / "trial-0/verifier/result.json"
    result.unlink()
    if kind == "deep_nesting":
        result.write_text("[" * 10000 + "0" + "]" * 10000)
    elif kind == "oversize":
        result.write_text(
            json.dumps({"used_tau2_evaluator": False, "padding": "x" * (1024 * 1024)})
        )
    elif kind == "directory":
        result.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFOs are unavailable")
        os.mkfifo(result)
    else:
        outside = path.parent.parent / (path.parent.name + "-outside.json")
        outside.write_text(json.dumps({"used_tau2_evaluator": False}))
        result.symlink_to(outside)
    assert _parse(raw, path).data["n_evaluator_not_run"] == 0


def test_partial_coverage_and_report_export(job, tmp_path):
    raw, path = job(False, True, True)
    (path.parent / "trial-1/verifier/result.json").unlink()
    (path.parent / "trial-2/verifier/result.json").write_text("{")
    block = _parse(raw, path)
    schema = ReportSchema(metadata={"report_id": "offline"}, sections=[block])
    report = generate_report(schema, tmp_path / "report")
    exported = json.loads(report.json_path.read_text())["sections"][0]
    assert exported["data"]["n_evaluator_not_run"] == 1
    assert exported["targets"]["n_evaluator_not_run"] == 1
    assert exported["targets"]["n_trials"] == 3
    assert not {"n_trials", "n_errors", "n_total_trials"} & block.data.keys()
    assert "Evaluator Did Not Run" in report.markdown


@pytest.mark.parametrize("flags", [(False,), (True,), (False, True), ()])
def test_runner_exact_warnings(job, caplog, monkeypatch, flags):
    raw, path = job(*flags)
    task = MagicMock(
        task_name="tau3_bench", workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC
    )
    driver = MagicMock()
    driver.run.return_value = SimpleNamespace(return_code=0, raw=raw)
    driver.parse.return_value = _parse(raw, path)
    ctx = MagicMock()
    ctx.all_params.tasks = [task]
    ctx.output_path = str(path.parent)
    for name in ("_configure_openai_env", "_require_openai_server", "accept_blocks"):
        monkeypatch.setattr(bridge, name, MagicMock())
    monkeypatch.setattr(bridge, "make_agentic_driver", MagicMock(return_value=driver))
    with caplog.at_level(logging.WARNING, logger=bridge.__name__):
        assert len(bridge.run_llm_agentic_eval(ctx)) == 1
    warnings = [r.getMessage() for r in caplog.records if r.name == bridge.__name__]
    expected = []
    if flags == (False,):
        expected = [
            "Task tau3_bench: the evaluator did not run for any of the 1 scored trials (used_tau2_evaluator is false); the score is not a measurement of the model."
        ]
    elif not flags:
        expected = ["Task tau3_bench: no scored trials; 5 trial errors."]
    assert warnings == expected
