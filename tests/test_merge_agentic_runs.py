# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for ``python -m report_module.merge_agentic_runs``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

from report_module import merge_agentic_runs as mar
from workflows.agentic_task_names import parse_task_names

EVAL_KEY = "terminus-2__openai/org/model__terminal-bench/terminal-bench-2-1"


@dataclass
class FakeScore:
    published_score: Optional[float] = 60.0
    published_score_ref: str = "https://example.com/pub"
    gpu_reference_score: Optional[float] = 50.0
    gpu_reference_score_ref: str = "https://example.com/ref"
    tolerance: float = 0.05
    mode_reference_scores: dict = field(default_factory=dict)


@dataclass
class FakeAgenticConfig:
    task_names: List[str] = field(default_factory=list)
    n_tasks: Optional[int] = None


def _task(**cfg):
    return SimpleNamespace(
        task_name="terminal_bench_2_1",
        score=FakeScore(),
        agentic_eval_config=FakeAgenticConfig(**cfg),
        priority="must",
    )


@pytest.fixture
def fake_task(monkeypatch):
    holder = {"task": _task()}
    monkeypatch.setattr(mar, "eval_task", lambda model, name: holder["task"])
    return holder


def _job(
    root: Path,
    name: str,
    outcomes: dict,
    *,
    finished_at: Optional[str] = "2026-11-20T10:00:00Z",
    n_total: Optional[int] = None,
    trial_dirs: bool = True,
    eval_key: str = EVAL_KEY,
    task_prefix: str = "terminal-bench/",
) -> Path:
    """A Harbor job dir: outcomes maps task -> 1.0 / 0.0 / exception type."""
    job_dir = root / name
    job_dir.mkdir(parents=True)
    rewards: dict = {}
    exceptions: dict = {}
    for i, (task, outcome) in enumerate(outcomes.items()):
        trial = f"{task[:32].rstrip('_-')}__abc{i:04d}"
        if isinstance(outcome, str):
            exceptions.setdefault(outcome, []).append(trial)
        else:
            rewards.setdefault(str(outcome), []).append(trial)
        if trial_dirs:
            (job_dir / trial).mkdir()
            (job_dir / trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": f"{task_prefix}{task}",
                        "trial_name": trial,
                        "finished_at": finished_at,
                    }
                )
            )
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "00000000-0000-0000-0000-000000000000",
                "started_at": "2026-11-20T09:00:00Z",
                "finished_at": finished_at,
                "n_total_trials": n_total if n_total is not None else len(outcomes),
                "stats": {
                    "evals": {
                        eval_key: {
                            "n_trials": sum(len(v) for v in rewards.values()),
                            "n_errors": sum(len(v) for v in exceptions.values()),
                            "metrics": [{"mean": 0.0}],
                            "reward_stats": {"reward": rewards},
                            "exception_stats": exceptions,
                        }
                    }
                },
            }
        )
    )
    return job_dir


def _run(tmp_path, *args):
    out = tmp_path / "merged"
    rc = mar.main(
        ["--model", "org/model", "--task", "terminal_bench_2_1", "--out", str(out)]
        + [str(a) for a in args]
    )
    result = json.loads((out / mar.MERGED_RESULT_NAME).read_text())
    return rc, result, (out / mar.MERGE_REPORT_NAME).read_text()


def test_two_slices_merge_to_a_pass(tmp_path, fake_task):
    a = _job(tmp_path, "tb_a", {"t1": 1.0, "t2": 1.0})
    b = _job(tmp_path, "tb_b", {"t3": 1.0, "t4": 0.0})
    rc, result, report = _run(tmp_path, "--expected-tasks", "t1,t2,t3,t4", a, b)
    assert rc == 0
    assert result["status"] == "PASS" and result["accuracy_check"] == "PASS"
    assert (result["n"], result["n_resolved"], result["score"]) == (4, 3, 75.0)
    assert result["stderr"] == pytest.approx(100 * (0.75 * 0.25 / 4) ** 0.5)
    assert (result["reference"], result["published"]) == (50.0, 60.0)
    assert result["tolerance"] == 0.05
    assert result["per_task"]["t4"]["outcome"] == "not_resolved"
    assert result["missing"] == [] and result["duplicates"] == {}
    assert "PASS" in report


def test_errored_trials_score_zero_and_fail(tmp_path, fake_task):
    job = _job(
        tmp_path,
        "tb",
        {"t1": 1.0, "t2": "AgentTimeoutError", "t3": 0.0, "t4": "RuntimeError"},
    )
    rc, result, _ = _run(tmp_path, job)
    assert rc == 1
    assert result["status"] == "FAIL"
    assert (result["n"], result["n_errored"], result["n_errors"]) == (4, 2, 2)
    assert result["score"] == 25.0
    assert result["per_task"]["t2"]["outcome"] == "errored"


def test_missing_task_is_incomplete(tmp_path, fake_task):
    job = _job(tmp_path, "tb", {"t1": 1.0})
    rc, result, report = _run(tmp_path, "--expected-tasks", "t1,t2", job)
    assert rc == 2
    assert result["status"] == "INCOMPLETE"
    assert result["accuracy_check"] == "NA"
    assert result["missing"] == ["t2"]
    assert "t2" in report


def test_duplicate_keeps_the_latest_job(tmp_path, fake_task):
    old = _job(tmp_path, "old", {"t1": 0.0}, finished_at="2026-11-20T10:00:00Z")
    new = _job(
        tmp_path, "new", {"t1": 1.0, "t2": 1.0}, finished_at="2026-11-21T10:00:00Z"
    )
    rc, result, _ = _run(tmp_path, "--expected-tasks", "t1,t2", old, new)
    assert rc == 0
    assert result["per_task"]["t1"]["outcome"] == "resolved"
    assert result["duplicates"]["t1"]["kept_job"] == str(new.resolve())
    assert result["duplicates"]["t1"]["dropped_jobs"] == [str(old.resolve())]
    assert result["n"] == 2


def test_unexpected_task_refuses(tmp_path, fake_task):
    job = _job(tmp_path, "tb", {"t1": 1.0, "zz": 1.0})
    rc, result, _ = _run(tmp_path, "--expected-tasks", "t1", job)
    assert rc == 2
    assert result["unexpected"] and "zz" in result["unexpected"][0]


def test_unfinished_job_refuses(tmp_path, fake_task):
    job = _job(tmp_path, "tb", {"t1": 1.0}, n_total=3)
    rc, result, _ = _run(tmp_path, job)
    assert rc == 2
    assert any("1 of 3 trials" in p for p in result["problems"])


def test_different_agent_or_model_refuses(tmp_path, fake_task):
    a = _job(tmp_path, "a", {"t1": 1.0})
    b = _job(tmp_path, "b", {"t2": 1.0}, eval_key="other-agent__m__d")
    rc, result, _ = _run(tmp_path, a, b)
    assert rc == 2
    assert any("different agent" in p for p in result["problems"])


def test_union_must_match_configured_n_tasks(tmp_path, fake_task):
    fake_task["task"] = _task(n_tasks=3)
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 1.0})
    rc, result, _ = _run(tmp_path, a)
    assert rc == 2
    assert result["expected_source"] == "union of the runs"
    b = _job(tmp_path, "b", {"t3": 1.0})
    rc, result, _ = _run(tmp_path, a, b)
    assert rc == 0 and result["n_tasks"] == 3


def test_configured_plain_task_names_are_the_expected_set(tmp_path, fake_task):
    fake_task["task"] = _task(task_names=["terminal-bench/t1", "terminal-bench/t2"])
    a = _job(tmp_path, "a", {"t1": 1.0})
    rc, result, _ = _run(tmp_path, a)
    assert rc == 2
    assert result["expected_source"] == "config task_names"
    assert result["missing"] == ["terminal-bench/t2"]


def test_trial_names_alone_resolve_truncated_task_names(tmp_path, fake_task):
    long_name = "a-very-long-terminal-bench-task-name-beyond-32"
    job = _job(tmp_path, "tb", {long_name: 1.0, "t2": 0.0}, trial_dirs=False)
    rc, result, _ = _run(
        tmp_path, "--expected-tasks", f"terminal-bench/{long_name},t2", job
    )
    assert rc == 0  # 50% against the 50% reference
    assert result["per_task"][f"terminal-bench/{long_name}"]["outcome"] == "resolved"


def test_registry_and_local_tau3_names_match(tmp_path, fake_task):
    job = _job(
        tmp_path,
        "tau",
        {"tau3-banking_knowledge-task-031": 1.0},
        task_prefix="",
    )
    expected = "sierra-research/tau3-bench__tau3-banking_knowledge-task-031"
    rc, result, _ = _run(tmp_path, "--expected-tasks", expected, job)
    assert result["missing"] == []
    assert result["per_task"][expected]["outcome"] == "resolved"


def test_parent_dir_is_searched_and_file_input_accepted(tmp_path, fake_task):
    _job(tmp_path / "runs", "a", {"t1": 1.0})
    b = _job(tmp_path / "other", "b", {"t2": 1.0})
    rc, result, _ = _run(
        tmp_path, "--expected-tasks", "t1,t2", tmp_path / "runs", b / "result.json"
    )
    assert rc == 0 and len(result["jobs"]) == 2


def test_expected_tasks_from_file(tmp_path, fake_task):
    names = tmp_path / "names.txt"
    names.write_text("# slice list\nt1\n\nt2\n")
    a = _job(tmp_path, "a", {"t1": 1.0, "t2": 1.0})
    rc, _, _ = _run(tmp_path, "--expected-tasks", f"@{names}", a)
    assert rc == 0


def test_no_job_found_is_refused(tmp_path, fake_task, capsys):
    (tmp_path / "empty").mkdir()
    out = tmp_path / "merged"
    rc = mar.main(
        ["--model", "m", "--task", "t", "--out", str(out), str(tmp_path / "empty")]
    )
    assert rc == 2
    assert "no Harbor job result.json" in capsys.readouterr().err


def test_real_config_expectation_uses_n_tasks():
    task = mar.eval_task("zai-org/GLM-5.2", "terminal_bench_2_1")
    assert mar.configured_expectation(task) == (None, 89)
    tau3 = mar.eval_task("zai-org/GLM-5.2", "tau3_bench_banking")
    # A glob is not a task list; the count still binds the union.
    assert mar.configured_expectation(tau3) == (None, 97)


@pytest.mark.parametrize(
    "value,expected",
    [("a,b", ["a", "b"]), (" a , b ", ["a", "b"]), ("x", ["x"])],
)
def test_parse_task_names(value, expected):
    assert parse_task_names(value) == expected


@pytest.mark.parametrize("value", ["", "a,,b", "a,a", "@/nonexistent/file"])
def test_parse_task_names_rejects(value):
    with pytest.raises(ValueError):
        parse_task_names(value)
