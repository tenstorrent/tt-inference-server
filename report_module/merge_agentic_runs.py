# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merge partial ``run.py --workflow agentic --agentic-task-names`` runs and grade them.

This is the only way a benchmark run in slices can PASS. Each slice's own
report is PARTIAL by construction; this tool collects every trial of the given
Harbor jobs, checks that together they cover the benchmark's task set exactly
once, scores the union the way a full run is scored, and grades it against the
eval config's reference with the full run's accept rule.

    python -m report_module.merge_agentic_runs --model <hf_repo> --task <task> \\
        --out <dir> [--expected-tasks a,b,...|@file] <job_dir_or_result_json> ...

Each input is a Harbor job dir (the ``<task>_<stamp>/`` dir holding the job's
``result.json`` and one dir per trial), that job's ``result.json``, or any
directory above job dirs (searched recursively for job ``result.json`` files).

Trial outcomes come from the fields the agentic parser grades a full run from:
the job's ``stats.evals.*.reward_stats.reward`` (reward value -> trial names;
a reward > 0 is resolved) and ``exception_stats`` (exception type -> trial
names; a trial without a reward is errored and scores 0, as Harbor's mean
does). Each trial's own ``<job_dir>/<trial_name>/result.json`` supplies its
full task name and finish time; without it the task is read from the trial
name (``<task name, cut at 32 chars>__<suffix>``).

The expected task set is ``--expected-tasks``, else the task's configured
``task_names`` when they are plain names (no glob), else the union of the
merged tasks, which must then number the configured ``n_tasks`` when one is
set. Every expected task must be present and nothing else may be: a missing
task, an unexpected one, an unfinished job, or jobs of different
agent/model/dataset refuse the merge. A task found in more than one job keeps
the trials of the job that finished it last; the others are reported as
duplicates.

Writes ``merged_result.json`` and ``merge_report.md`` under ``--out``. Exit 0
PASS, 1 graded but not PASS, 2 the runs do not form one complete set (nothing
is graded; ``merged_result.json`` still records what is missing).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from workflows.agentic_task_names import parse_task_names

MERGED_RESULT_NAME = "merged_result.json"
MERGE_REPORT_NAME = "merge_report.md"
JOB_RESULT_NAME = "result.json"
# Harbor's trial name: f"{task_name[:32].rstrip('_-')}__{7-char suffix}".
TRIAL_NAME_TASK_CHARS = 32
_GLOB_CHARS = set("*?[")

RESOLVED = "resolved"
NOT_RESOLVED = "not_resolved"
ERRORED = "errored"


class MergeError(Exception):
    """The given runs do not form one complete, gradable set."""


@dataclass
class Trial:
    trial_name: str
    job_dir: str
    outcome: str
    reward: Optional[float]
    exception_type: Optional[str] = None
    task_name: Optional[str] = None  # full name from the trial's result.json
    finished_at: Optional[str] = None


@dataclass
class Job:
    job_dir: Path
    eval_keys: List[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    n_total_trials: Optional[int]
    trials: List[Trial] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


# --- reading Harbor jobs -----------------------------------------------------


def _is_job_result(data: Any) -> bool:
    return isinstance(data, dict) and "stats" in data and "n_total_trials" in data


def find_job_dirs(paths: Sequence[Path]) -> List[Path]:
    """Every Harbor job dir named by ``paths`` (see the module docstring)."""
    found: List[Path] = []
    for path in map(Path, paths):
        if path.is_file():
            candidates = [path]
        elif (path / JOB_RESULT_NAME).is_file():
            candidates = [path / JOB_RESULT_NAME]
        elif path.is_dir():
            candidates = sorted(path.rglob(JOB_RESULT_NAME))
        else:
            raise MergeError(f"{path}: no such file or directory")
        jobs = []
        for candidate in candidates:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                if candidate == path or candidate.parent == path:
                    raise MergeError(f"{candidate}: unreadable ({exc})") from exc
                continue
            if _is_job_result(data):
                jobs.append(candidate.parent.resolve())
        if not jobs:
            raise MergeError(f"{path}: no Harbor job result.json found")
        found.extend(jobs)
    unique = sorted(set(found))
    return unique


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _reward_map(eval_stats: Mapping[str, Any]) -> Mapping[str, Any]:
    """The pass/fail reward's value -> trial names map (the parser's field).

    ``reward`` when the verifier writes several rewards (DeepSWE), else the
    single reward key Harbor recorded.
    """
    reward_stats = eval_stats.get("reward_stats") or {}
    if not isinstance(reward_stats, Mapping):
        return {}
    if "reward" in reward_stats:
        rewards = reward_stats["reward"]
    elif len(reward_stats) == 1:
        rewards = next(iter(reward_stats.values()))
    else:
        rewards = {}
    return rewards if isinstance(rewards, Mapping) else {}


def _read_trial_result(job_dir: Path, trial_name: str) -> Optional[dict]:
    path = job_dir / trial_name / JOB_RESULT_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "trial_name" in data else None


def load_job(job_dir: Path) -> Job:
    raw = json.loads((job_dir / JOB_RESULT_NAME).read_text(encoding="utf-8"))
    stats = raw.get("stats") or {}
    evals = stats.get("evals") or {}
    job = Job(
        job_dir=job_dir,
        eval_keys=sorted(evals),
        started_at=raw.get("started_at"),
        finished_at=raw.get("finished_at"),
        n_total_trials=raw.get("n_total_trials")
        if isinstance(raw.get("n_total_trials"), int)
        else None,
    )
    rewards: Dict[str, float] = {}
    exceptions: Dict[str, str] = {}
    for eval_stats in evals.values():
        if not isinstance(eval_stats, Mapping):
            continue
        for value, names in _reward_map(eval_stats).items():
            try:
                reward = float(value)
            except (TypeError, ValueError):
                continue
            for name in names if isinstance(names, list) else []:
                rewards[name] = reward
        exception_stats = eval_stats.get("exception_stats") or {}
        if isinstance(exception_stats, Mapping):
            for exc_type, names in exception_stats.items():
                for name in names if isinstance(names, list) else []:
                    exceptions[name] = exc_type
    names = set(rewards) | set(exceptions)
    # A trial that finished with neither a reward nor an exception (no
    # verifier result) is in no stats list but counts 0 in Harbor's mean.
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        if trial_dir.name in names:
            continue
        data = _read_trial_result(job_dir, trial_dir.name)
        if data is not None and data.get("finished_at"):
            names.add(trial_dir.name)
    for name in sorted(names):
        reward = rewards.get(name)
        if reward is not None:
            outcome = RESOLVED if reward > 0 else NOT_RESOLVED
        elif name in exceptions:
            outcome = ERRORED
        else:
            outcome = NOT_RESOLVED
        trial_result = _read_trial_result(job_dir, name) or {}
        job.trials.append(
            Trial(
                trial_name=name,
                job_dir=str(job_dir),
                outcome=outcome,
                reward=reward,
                exception_type=exceptions.get(name),
                task_name=trial_result.get("task_name"),
                finished_at=trial_result.get("finished_at"),
            )
        )
    cancelled = [
        t.trial_name for t in job.trials if t.exception_type == "CancelledError"
    ]
    if cancelled:
        job.problems.append(f"{job_dir}: {len(cancelled)} trial(s) were cancelled")
    if job.finished_at is None:
        job.problems.append(f"{job_dir}: job has no finished_at (still running?)")
    if job.n_total_trials is not None and len(job.trials) < job.n_total_trials:
        job.problems.append(
            f"{job_dir}: {len(job.trials)} of {job.n_total_trials} trials finished"
        )
    return job


# --- task names --------------------------------------------------------------


def canonical_task_name(name: str) -> str:
    """Dataset-independent task key.

    ``terminal-bench/compile-compcert`` -> ``compile-compcert``;
    ``sierra-research/tau3-bench__tau3-banking_knowledge-task-031`` and its
    local-copy name ``tau3-banking_knowledge-task-031`` -> the latter.
    """
    return name.split("/")[-1].split("__")[-1]


def _trial_name_prefix(name: str) -> str:
    """What Harbor keeps of a task name in its trial names."""
    return name[:TRIAL_NAME_TASK_CHARS].rstrip("_-")


def _trial_task_prefix(trial_name: str) -> str:
    return trial_name.rsplit("__", 1)[0] if "__" in trial_name else trial_name


def _matches(trial: Trial, expected: str) -> bool:
    if trial.task_name:
        return canonical_task_name(trial.task_name) == canonical_task_name(expected)
    prefix = _trial_task_prefix(trial.trial_name)
    base = expected.split("/")[-1]
    return prefix in {
        _trial_name_prefix(base),
        _trial_name_prefix(canonical_task_name(expected)),
    }


def assign_tasks(
    trials: Sequence[Trial], expected: Optional[Sequence[str]]
) -> Tuple[Dict[str, List[Trial]], List[str]]:
    """``{task: trials}`` and the trials matching no expected task.

    With an expected list every trial is keyed by the expected name it matches
    (an ambiguous match refuses the merge); without one, by its own task name.
    """
    by_task: Dict[str, List[Trial]] = {}
    unexpected: List[str] = []
    for trial in trials:
        if expected is None:
            key = canonical_task_name(
                trial.task_name or _trial_task_prefix(trial.trial_name)
            )
        else:
            hits = [name for name in expected if _matches(trial, name)]
            if len(hits) > 1:
                raise MergeError(
                    f"trial {trial.trial_name} matches several expected tasks {hits}; "
                    "keep the trial dirs (they carry the full task name)"
                )
            if not hits:
                unexpected.append(f"{trial.trial_name} ({trial.job_dir})")
                continue
            key = hits[0]
        by_task.setdefault(key, []).append(trial)
    return by_task, unexpected


def _trial_time(trial: Trial, jobs: Mapping[str, Job]) -> datetime:
    job = jobs[trial.job_dir]
    for value in (trial.finished_at, job.finished_at, job.started_at):
        parsed = _parse_time(value)
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


def dedupe(
    by_task: Dict[str, List[Trial]], jobs: Mapping[str, Job]
) -> Tuple[Dict[str, List[Trial]], Dict[str, Dict[str, Any]]]:
    """Keep each task's trials from one job: the one that finished it last."""
    kept: Dict[str, List[Trial]] = {}
    duplicates: Dict[str, Dict[str, Any]] = {}
    for task, trials in by_task.items():
        by_job: Dict[str, List[Trial]] = {}
        for trial in trials:
            by_job.setdefault(trial.job_dir, []).append(trial)
        if len(by_job) == 1:
            kept[task] = trials
            continue
        latest = max(
            by_job,
            key=lambda job_dir: max(_trial_time(t, jobs) for t in by_job[job_dir]),
        )
        kept[task] = by_job[latest]
        duplicates[task] = {
            "kept_job": latest,
            "dropped_jobs": sorted(j for j in by_job if j != latest),
        }
    return kept, duplicates


# --- the eval config ---------------------------------------------------------


def eval_task(model: str, task_name: str):
    """The EvalConfig agentic task the runs are merged and graded as."""
    from reference_config.evals.eval_config import _eval_config_map

    config = _eval_config_map.get(model)
    tasks = {t.task_name: t for t in (config.tasks if config else [])}
    task = tasks.get(task_name)
    if task is None or getattr(task, "agentic_eval_config", None) is None:
        raise MergeError(f"{model} has no agentic EvalConfig task {task_name}")
    return task


def configured_expectation(task) -> Tuple[Optional[List[str]], Optional[int]]:
    """(expected names, expected count) from the task's agentic config."""
    cfg = task.agentic_eval_config
    names = list(getattr(cfg, "task_names", None) or [])
    if names and not any(_GLOB_CHARS & set(name) for name in names):
        return names, len(names)
    n_tasks = getattr(cfg, "n_tasks", None)
    return None, n_tasks if isinstance(n_tasks, int) else None


# --- scoring and grading -----------------------------------------------------


def score_trials(by_task: Mapping[str, List[Trial]]) -> Dict[str, Any]:
    """Harbor's full-run score over the merged trials (errored trials score 0)."""
    trials = [t for ts in by_task.values() for t in ts]
    n = len(trials)
    n_resolved = sum(t.outcome == RESOLVED for t in trials)
    n_errors = sum(t.exception_type is not None for t in trials)
    n_scored = sum(t.reward is not None for t in trials)
    rewards = [t.reward or 0.0 for t in trials]
    accuracy = sum(rewards) / n if n else None
    stderr = math.sqrt(accuracy * (1 - accuracy) / n) if n else None
    return {
        "n": n,
        "n_resolved": n_resolved,
        "n_not_resolved": sum(t.outcome == NOT_RESOLVED for t in trials),
        "n_errored": sum(t.outcome == ERRORED for t in trials),
        "n_errors": n_errors,
        "n_scored": n_scored,
        "accuracy": accuracy,
        "score": accuracy * 100 if accuracy is not None else None,
        "stderr": stderr * 100 if stderr is not None else None,
        "binary_rewards": all(r in (0.0, 1.0) for r in rewards),
    }


def grade(task, scores: Mapping[str, Any]) -> Dict[str, Any]:
    """Grade the merged score exactly as the parser grades a full run."""
    from llm_module.parsers.agentic import _build_evals_data
    from workflow_module.engine_types import ReportCheckTypes
    from workflow_module.target_pack import get_target_pack

    metrics: Dict[str, Any] = {}
    if scores["accuracy"] is not None:
        metrics = {"accuracy": scores["accuracy"], "n_trials": scores["n_scored"]}
    data = _build_evals_data(
        task_name=task.task_name,
        score=task.score,
        metrics=metrics,
        priority=getattr(task, "priority", "must"),
    )
    ref = (
        get_target_pack().resolve_eval_reference(task.score, None)
        if task.score is not None
        else {}
    )
    check = ReportCheckTypes(data["accuracy_check"])
    return {
        "reference": ref.get("reference_score"),
        "reference_ref": ref.get("reference_ref"),
        "published": data["published_score"],
        "published_ref": data["published_score_ref"],
        "tolerance": ref.get("tolerance", data["tolerance"]),
        "ratio_to_reference": data["ratio_to_reference"],
        "ratio_to_published": data["ratio_to_published"],
        "accuracy_check": check.name,
    }


# --- the merge ---------------------------------------------------------------


def merge(
    model: str,
    task_name: str,
    inputs: Sequence[Path],
    expected: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Merge and grade; returns the ``merged_result.json`` document.

    ``status`` is ``PASS``/``FAIL``/``NA`` when graded, ``INCOMPLETE`` when the
    runs do not form one complete set (``problems`` says why).
    """
    task = eval_task(model, task_name)
    expected_source = "--expected-tasks"
    expected_count = len(expected) if expected is not None else None
    if expected is None:
        expected, expected_count = configured_expectation(task)
        expected_source = (
            "config task_names" if expected is not None else "union of the runs"
        )
    expected = list(expected) if expected is not None else None

    jobs = {str(d): load_job(d) for d in find_job_dirs(inputs)}
    problems = [p for job in jobs.values() for p in job.problems]
    eval_keys = sorted({k for job in jobs.values() for k in job.eval_keys})
    if len(eval_keys) > 1:
        problems.append(
            f"the jobs ran different agent/model/dataset combinations: {eval_keys}"
        )
    trials = [t for job in jobs.values() for t in job.trials]
    by_task, unexpected = assign_tasks(trials, expected)
    if unexpected:
        problems.append(
            f"{len(unexpected)} trial(s) of tasks outside the expected set: "
            + ", ".join(unexpected[:20])
        )
    by_task, duplicates = dedupe(by_task, jobs)
    if expected is not None:
        missing = [name for name in expected if name not in by_task]
    else:
        missing = []
        if expected_count is not None and len(by_task) != expected_count:
            problems.append(
                f"merged {len(by_task)} distinct task(s) but the configured run has "
                f"n_tasks={expected_count}; pass --expected-tasks to name the set"
            )
    if missing:
        problems.append(f"{len(missing)} expected task(s) missing")

    scores = score_trials(by_task)
    if not problems and scores["n"] == 0:
        problems.append("no trials found")
    per_task = {
        name: {
            "outcome": trials_[0].outcome
            if len({t.outcome for t in trials_}) == 1
            else "mixed",
            "trials": [
                {
                    k: v
                    for k, v in asdict(t).items()
                    if k in ("trial_name", "outcome", "reward", "exception_type")
                }
                for t in trials_
            ],
            "job_dir": trials_[0].job_dir,
        }
        for name, trials_ in sorted(by_task.items())
    }
    graded = grade(task, scores)
    status = "INCOMPLETE" if problems else graded["accuracy_check"]
    if problems:
        graded["accuracy_check"] = "NA"
    return {
        "model": model,
        "task_name": task_name,
        "status": status,
        **graded,
        "score": scores["score"],
        "stderr": scores["stderr"],
        "accuracy": scores["accuracy"],
        "n": scores["n"],
        "n_tasks": len(by_task),
        "n_resolved": scores["n_resolved"],
        "n_not_resolved": scores["n_not_resolved"],
        "n_errored": scores["n_errored"],
        "n_errors": scores["n_errors"],
        "binary_rewards": scores["binary_rewards"],
        "expected_source": expected_source,
        "expected_count": expected_count,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "problems": problems,
        "eval_keys": eval_keys,
        "jobs": sorted(jobs),
        "per_task": per_task,
    }


def render_report(result: Mapping[str, Any]) -> str:
    def fmt(value, spec=".2f"):
        return "N/A" if value is None else format(value, spec)

    lines = [
        f"# Agentic merge: {result['task_name']} ({result['model']})",
        "",
        f"**Status: {result['status']}**",
        "",
        "| score % | stderr % | reference % | published % | tolerance | tasks | "
        "trials | resolved | errored |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {fmt(result['score'])} | {fmt(result['stderr'])} | "
        f"{fmt(result['reference'])} | {fmt(result['published'])} | "
        f"{fmt(result['tolerance'])} | {result['n_tasks']} | {result['n']} | "
        f"{result['n_resolved']} | {result['n_errored']} |",
        "",
        f"Expected task set: {result['expected_source']}"
        + (
            f" ({result['expected_count']} tasks)"
            if result["expected_count"] is not None
            else ""
        ),
        f"Jobs merged: {len(result['jobs'])}",
    ]
    if result["problems"]:
        lines += ["", "## Not gradable", ""]
        lines += [f"- {p}" for p in result["problems"]]
    if result["missing"]:
        lines += ["", "## Missing tasks", ""]
        lines += [f"- {name}" for name in result["missing"]]
    if result["duplicates"]:
        lines += ["", "## Duplicates (kept the latest job)", ""]
        lines += [
            f"- {task}: kept {d['kept_job']}, dropped {', '.join(d['dropped_jobs'])}"
            for task, d in sorted(result["duplicates"].items())
        ]
    if not result["binary_rewards"]:
        lines += ["", "Note: non-binary rewards; the score is the mean reward."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="HF repo id, e.g. zai-org/GLM-5.2"
    )
    parser.add_argument(
        "--task", required=True, help="agentic eval task, e.g. terminal_bench_2_1"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--expected-tasks",
        default=None,
        help="the benchmark's task names: a,b,... or @file (one per line)",
    )
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        expected = (
            parse_task_names(args.expected_tasks)
            if args.expected_tasks is not None
            else None
        )
        result = merge(args.model, args.task, args.runs, expected)
    except (MergeError, ValueError) as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / MERGED_RESULT_NAME).write_text(
        json.dumps(result, indent=2, default=str)
    )
    report = render_report(result)
    (args.out / MERGE_REPORT_NAME).write_text(report)
    print(report)
    if result["status"] == "INCOMPLETE":
        return 2
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
