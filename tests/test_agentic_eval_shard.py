# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Agentic (Harbor) evals selected by --eval-tasks and sharded by --eval-shard.

Selection: --eval-tasks may name an agentic task; --workflow evals then runs it
through AgenticWorkflow. Partition: shard i/N runs every N-th name of the sorted
task set from i, so the N shards cover it exactly once. Merge: the full-task
accuracy is the sum of every shard's rewards over the full trial count, and the
merge refuses missing, duplicate or foreign shards.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_module import DriverContext, ServerConnection
from llm_module.drivers.agentic import HarborAgenticDriver, harbor_command_sha
from llm_module.eval_configs import _select_tasks
from report_module import merge_eval_shards as merge_cli
from test_module.llm_tests.agentic_eval_tests import (
    _select_agentic_tasks,
    run_llm_agentic_eval,
)
from workflow_module.execution import OrchestratorMetadata, TaskOutcome
from workflow_module.workflows import EvalsWorkflow, ReleaseWorkflow
from workflows.eval_shard import (
    SHARD_RECORD_PREFIX,
    ShardError,
    agentic_shard_plan,
    agentic_shard_record,
    agentic_shard_task_names,
    check_agentic_shard_set,
    merge_agentic_shard_results,
)
from workflows.workflow_types import ModelType, WorkflowType, WorkflowVenvType

MODEL = "Qwen/Qwen3.8-27B"
TASK = "terminal_bench_2_1"
DATASET = "terminal-bench/terminal-bench-2-1"
NAMES = [f"terminal-bench/task-{i:02d}" for i in range(89)]
PROV = {
    "hf_model_repo": MODEL,
    "ttis_ref": "r",
    "package_manifest_sha256": "p",
    "docker_image": "i",
    "eval_command_sha": "c",
    "device": "p150x4",
    "model_status": "EXPERIMENTAL",
}


def _plan(names=NAMES, **overrides):
    kwargs = dict(
        dataset=DATASET,
        dataset_ref=None,
        resolved_ref="sha256:abc",
        task_names=names,
        n_attempts=1,
    )
    kwargs.update(overrides)
    return agentic_shard_plan(**kwargs)


# --- selection ---------------------------------------------------------------


def _task(name, venv=WorkflowVenvType.EVALS_COMMON):
    return SimpleNamespace(task_name=name, workflow_venv_type=venv)


LM_TASKS = [_task("r1_gpqa_diamond")]
ALL_TASKS = LM_TASKS + [
    _task(TASK, WorkflowVenvType.EVALS_AGENTIC),
    _task("swe_bench_verified", WorkflowVenvType.EVALS_AGENTIC),
]


def _rc(eval_tasks=None, **kw):
    return SimpleNamespace(
        eval_tasks=eval_tasks, eval_samples=None, limit_samples_mode=None, **kw
    )


def test_lm_eval_selection_accepts_an_agentic_name_and_selects_no_lm_task():
    configured = {t.task_name for t in ALL_TASKS}
    assert _select_tasks(LM_TASKS, _rc(TASK), configured_names=configured) == []
    both = _select_tasks(LM_TASKS, _rc(f"{TASK},r1_gpqa_diamond"), configured)
    assert [t.task_name for t in both] == ["r1_gpqa_diamond"]


def test_lm_eval_selection_still_refuses_a_name_nothing_configures():
    configured = {t.task_name for t in ALL_TASKS}
    with pytest.raises(ValueError, match="does not configure"):
        _select_tasks(LM_TASKS, _rc(f"{TASK},typo"), configured_names=configured)


def test_agentic_selection_honors_eval_tasks():
    ctx = SimpleNamespace(
        all_params=SimpleNamespace(tasks=ALL_TASKS),
        runtime_config=_rc(f"r1_gpqa_diamond,{TASK}", agentic_benchmark=None),
    )
    assert [t.task_name for t in _select_agentic_tasks(ctx)] == [TASK]
    ctx.runtime_config = _rc(None, agentic_benchmark=None)
    assert [t.task_name for t in _select_agentic_tasks(ctx)] == [
        TASK,
        "swe_bench_verified",
    ]
    ctx.runtime_config = _rc("typo", agentic_benchmark=None)
    with pytest.raises(ValueError, match="does not configure"):
        _select_agentic_tasks(ctx)


def _evals_ctx(eval_tasks):
    ctx = MagicMock()
    ctx.model_spec.model_type = ModelType.LLM
    ctx.all_params = SimpleNamespace(tasks=ALL_TASKS)
    ctx.runtime_config = _rc(eval_tasks)
    return ctx


@pytest.mark.parametrize(
    "eval_tasks, lm_eval_runs, agentic_runs",
    [
        (None, True, False),  # unchanged: a release runs the agentic set
        ("r1_gpqa_diamond", True, False),
        (TASK, False, True),
        (f"r1_gpqa_diamond,{TASK}", True, True),
    ],
)
def test_evals_workflow_runs_the_agentic_tasks_eval_tasks_names(
    eval_tasks, lm_eval_runs, agentic_runs
):
    lm = TaskOutcome("evaluation", 0, 0.0, "evals")
    ag = TaskOutcome("evaluation", 0, 0.0, "evals")
    with patch.object(
        EvalsWorkflow, "_run_llm_eval_task", return_value=lm
    ) as lm_eval, patch(
        "workflow_module.workflows.AgenticWorkflow.run_tasks", return_value=[ag]
    ) as agentic:
        outcomes = EvalsWorkflow(
            _evals_ctx(eval_tasks),
            accumulator=MagicMock(),
            orchestrator_metadata=OrchestratorMetadata(server_mode="local"),
        ).run_tasks()
    assert lm_eval.called is lm_eval_runs
    assert agentic.called is agentic_runs
    assert len(outcomes) == lm_eval_runs + agentic_runs


@pytest.mark.parametrize("eval_tasks, agentic_child", [(None, True), (TASK, False)])
def test_release_leaves_selected_agentic_tasks_to_the_evals_child(
    eval_tasks, agentic_child
):
    wf = ReleaseWorkflow(
        _evals_ctx(eval_tasks),
        accumulator=MagicMock(),
        orchestrator_metadata=OrchestratorMetadata(server_mode="local"),
    )
    assert ("agentic" in wf._llm_children()) is agentic_child


def test_evals_dispatch_provisions_the_agentic_venv_only_when_selected(monkeypatch):
    from workflows import workflow_dispatch as dispatch

    monkeypatch.setattr(
        dispatch, "_eval_config_for", lambda spec: SimpleNamespace(tasks=ALL_TASKS)
    )
    spec = SimpleNamespace(model_type=ModelType.LLM, hf_model_repo=MODEL)
    selected = dispatch._engine_dependency_venv_types(
        spec, WorkflowType.EVALS, _rc(TASK)
    )
    unselected = dispatch._engine_dependency_venv_types(
        spec, WorkflowType.EVALS, _rc("r1_gpqa_diamond")
    )
    assert WorkflowVenvType.EVALS_AGENTIC in selected
    assert WorkflowVenvType.EVALS_AGENTIC not in unselected


# --- partition ---------------------------------------------------------------


@pytest.mark.parametrize("n_tasks", [89, 500, 9])
@pytest.mark.parametrize("count", [2, 3, 4, 7, 8])
def test_shards_cover_every_task_exactly_once(n_tasks, count):
    names = [f"t{i:03d}" for i in reversed(range(n_tasks))]  # any input order
    plan = _plan(names)
    shards = [agentic_shard_task_names(plan, i, count) for i in range(count)]
    flat = [name for shard in shards for name in shard]
    assert sorted(flat) == sorted(names)
    assert len(flat) == len(set(flat)) == n_tasks
    # Deterministic: every N-th of the sorted set, from i.
    assert shards[1] == sorted(names)[1::count]


def test_plan_refuses_an_empty_set_duplicates_and_empty_shards():
    with pytest.raises(ShardError, match="no tasks"):
        _plan([])
    with pytest.raises(ShardError, match="twice"):
        _plan(["a", "b", "a"])
    # Harbor reads empty task_names as "no filter": an empty shard would run all.
    with pytest.raises(ShardError, match="empty"):
        agentic_shard_task_names(_plan(["a", "b"]), 2, 3)


def _harbor_task():
    cfg = SimpleNamespace(
        dataset=DATASET,
        dataset_ref=None,
        agent="terminus-2",
        model=None,
        n_concurrent_trials=10,
        n_attempts=1,
        n_tasks=89,
        task_names=[],
        exclude_task_names=[],
        task_names_map={},
        agent_kwargs={},
        environment_type="docker",
        override_cpus=16,
        override_memory_mb=48 * 1024,
        timeout_multiplier=None,
        agent_timeout_sec=6 * 60 * 60,
        agent_setup_timeout_multiplier=None,
        quiet=False,
        yes=True,
        agent_import_path=None,
        agent_env={},
        environment_env={},
        verifier_env={},
        environment_kwargs={},
        harbor_timeout_sec=None,
        llm_timeout_sec=600,
        per_task_overhead_sec=1200,
        startup_grace_sec=600,
        stall_grace_sec=300,
        progress_log_interval_sec=60,
        enforce_agent_deadline=False,
    )
    return SimpleNamespace(
        task_name=TASK,
        workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
        score=None,
        agentic_eval_config=cfg,
        limit_samples_map={},
        priority="must",
    )


def _run_shard(tmp_path, shard, rc=1):
    """Run the driver for one shard with Harbor stubbed; return (driver, config)."""
    task = _harbor_task()
    runtime = SimpleNamespace(eval_shard=shard, limit_samples_mode=None)
    driver = HarborAgenticDriver(task, runtime_config=runtime)
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return rc

    listing = {"task_names": list(reversed(NAMES)), "resolved_ref": "sha256:abc"}
    with patch(
        "llm_module.drivers.agentic.list_harbor_task_names", return_value=listing
    ) as lister, patch("llm_module.drivers.agentic.run_harbor", side_effect=fake_run):
        driver.run(
            SimpleNamespace(),
            ServerConnection(
                base_url="http://127.0.0.1", service_port=8000, model=MODEL
            ),
            DriverContext(output_dir=tmp_path, device="p150x4"),
        )
    # The listing is resolved with the run's own filters (n_tasks=89).
    assert lister.call_args.args[2] == 89
    return driver, captured["config"]


def test_driver_runs_exactly_its_shard_through_harbor_task_names(tmp_path):
    seen = []
    for i in range(4):
        driver, config = _run_shard(tmp_path, f"{i}/4")
        expected = sorted(NAMES)[i::4]
        assert config.task_names == [glob.escape(n) for n in expected]
        assert driver.shard["task_names"] == expected
        assert driver.shard["job_dir"] == config.jobs_dir / config.task_name
        seen.extend(expected)
    assert sorted(seen) == sorted(NAMES)


def test_command_hash_ignores_only_the_per_shard_fields(tmp_path):
    _, first = _run_shard(tmp_path, "0/4")
    _, second = _run_shard(tmp_path, "1/4")
    assert first.task_names != second.task_names
    assert harbor_command_sha(first) == harbor_command_sha(second)


def test_bridge_records_the_shard_and_reports_it_partial(tmp_path):
    from llm_module.parsers.agentic import AgenticEvalParser

    job_dir = tmp_path / "job"
    plan = _plan()
    shard = {
        "plan": plan,
        "index": 1,
        "count": 4,
        "task_names": agentic_shard_task_names(plan, 1, 4),
        "job_dir": job_dir,
        "command_sha": "cmd",
    }
    raw = {
        "n_total_trials": 22,
        "stats": {"evals": {"k": {"metrics": [{"mean": 0.5}]}}},
    }
    driver = MagicMock()
    driver.shard = shard
    driver.run.return_value = SimpleNamespace(return_code=0, raw=raw)
    driver.parse.return_value = AgenticEvalParser(task_name=TASK).parse(raw)
    ctx = MagicMock()
    ctx.all_params = SimpleNamespace(tasks=[_harbor_task()])
    ctx.runtime_config = SimpleNamespace(
        eval_tasks=TASK,
        eval_shard="1/4",
        limit_samples_mode=None,
        agentic_benchmark=None,
        run_id="run-1",
    )
    ctx.model_spec = SimpleNamespace(
        hf_model_repo=MODEL,
        model_name="Qwen3.8-27B",
        docker_image="img",
        status="EXPERIMENTAL",
        device_model_spec=None,
    )
    ctx.device = SimpleNamespace(name="p150x4")
    with patch(
        "test_module.llm_tests.agentic_eval_tests._require_openai_server"
    ), patch("test_module.llm_tests.agentic_eval_tests._configure_openai_env"), patch(
        "test_module.llm_tests.agentic_eval_tests.make_agentic_driver",
        return_value=driver,
    ), patch("test_module.llm_tests.agentic_eval_tests.accept_blocks"):
        blocks = run_llm_agentic_eval(ctx)

    assert blocks[0].data["status"] == "partial"
    record = json.loads((job_dir / f"{SHARD_RECORD_PREFIX}{TASK}.json").read_text())
    assert record["shard"] == "1/4"
    assert record["shard_task_names"] == sorted(NAMES)[1::4]
    assert record["expected_task_count"] == 89
    assert record["return_code"] == 0
    assert record["run_id"] == "run-1"
    assert record["provenance"]["eval_command_sha"] == "cmd"
    assert record["provenance"]["hf_model_repo"] == MODEL


# --- merge -------------------------------------------------------------------


def _harbor_result(names, passed, n_attempts=1, errors=0):
    """A Harbor result.json for ``names``; ``passed`` of them rewarded 1.0."""
    trials = [
        f"{n.split('/')[-1][:32].rstrip('_-')}__abc{i}" for i, n in enumerate(names)
    ]
    ok = [t for t, n in zip(trials, names) if n in passed]
    bad = [t for t, n in zip(trials, names) if n not in passed]
    total = len(names) * n_attempts
    return {
        "n_total_trials": total,
        "stats": {
            "n_completed_trials": total,
            "n_errored_trials": errors,
            "n_running_trials": 0,
            "n_pending_trials": 0,
            "evals": {
                f"terminus-2__{MODEL}__{DATASET}": {
                    "n_trials": len(trials),
                    "n_errors": errors,
                    "metrics": [{"mean": len(ok) / max(len(trials), 1)}],
                    "reward_stats": {"reward": {"1.0": ok, "0.0": bad}},
                }
            },
        },
    }


PASSED = set(sorted(NAMES)[::3])  # 30 of 89


def _write_shard(root, index, count, *, plan=None, prov=PROV, rc=0, result=None):
    plan = plan or _plan()
    names = agentic_shard_task_names(plan, index, count)
    job = Path(root) / f"shard{index}" / "eval_x" / "agentic" / f"{TASK}_stamp"
    job.mkdir(parents=True, exist_ok=True)
    record = agentic_shard_record(
        TASK, plan, index, count, prov, return_code=rc, run_id=f"run-{index}"
    )
    (job / f"{SHARD_RECORD_PREFIX}{TASK}.json").write_text(json.dumps(record))
    result = result if result is not None else _harbor_result(names, PASSED)
    (job / "result.json").write_text(json.dumps(result))
    return job


def test_merge_sums_passes_over_the_full_task_count():
    plan = _plan()
    results = {
        i: _harbor_result(agentic_shard_task_names(plan, i, 4), PASSED)
        for i in range(4)
    }
    doc = merge_agentic_shard_results(plan, 4, results)
    (stats,) = doc["stats"]["evals"].values()
    assert stats["metrics"][0]["mean"] == pytest.approx(len(PASSED) / 89)
    assert doc["n_total_trials"] == 89
    assert stats["n_trials"] == 89


def test_merge_cli_grades_the_merged_full_task(tmp_path, capsys):
    for i in range(4):
        _write_shard(tmp_path, i, 4)
    out = tmp_path / "out"
    code = merge_cli.main(
        ["--model", MODEL, "--task", TASK, "--out", str(out), str(tmp_path)]
    )
    assert code in (0, 1)  # graded (PASS or not); 2 would be a refused merge
    merged = json.loads((out / merge_cli.MERGED_RESULTS_NAME).read_text())
    (stats,) = merged["stats"]["evals"].values()
    assert stats["metrics"][0]["mean"] == pytest.approx(len(PASSED) / 89)
    assert merged["eval_shard_merge"]["shard_count"] == 4
    assert merged["eval_shard_merge"]["expected_task_count"] == 89
    report = json.loads((out / merge_cli.MERGED_REPORT_NAME).read_text())
    (block,) = report["evals"]
    assert block["data"]["score"] == pytest.approx(100 * len(PASSED) / 89)
    assert block["data"].get("status") != "partial"


def _refused(tmp_path, capsys, match):
    code = merge_cli.main(
        [
            "--model",
            MODEL,
            "--task",
            TASK,
            "--out",
            str(tmp_path / "out"),
            str(tmp_path),
        ]
    )
    assert code == 2
    assert match in capsys.readouterr().err


def test_merge_refuses_a_missing_shard(tmp_path, capsys):
    for i in (0, 1, 3):
        _write_shard(tmp_path, i, 4)
    _refused(tmp_path, capsys, "missing [2]")


def test_merge_refuses_a_duplicated_shard(tmp_path, capsys):
    for i in range(4):
        _write_shard(tmp_path, i, 4)
    _write_shard(tmp_path / "again", 2, 4)
    _refused(tmp_path, capsys, "duplicated [2]")


def test_merge_ignores_a_failed_attempt_so_its_shard_is_missing(tmp_path, capsys):
    for i in range(4):
        _write_shard(tmp_path, i, 4, rc=124 if i == 1 else 0)
    _refused(tmp_path, capsys, "missing [1]")


@pytest.mark.parametrize("key", ["docker_image", "package_manifest_sha256", "device"])
def test_merge_refuses_shards_of_different_identity(tmp_path, capsys, key):
    for i in range(4):
        prov = {**PROV, key: "other"} if i == 3 else PROV
        _write_shard(tmp_path, i, 4, prov=prov)
    _refused(tmp_path, capsys, key)


def test_merge_refuses_shards_of_different_plans(tmp_path, capsys):
    for i in range(4):
        plan = _plan(resolved_ref="sha256:def") if i == 3 else None
        _write_shard(tmp_path, i, 4, plan=plan)
    _refused(tmp_path, capsys, "plan_sha")


def test_merge_refuses_a_plan_of_another_dataset(tmp_path, capsys):
    plan = _plan(dataset="terminal-bench/terminal-bench-2")
    for i in range(4):
        _write_shard(tmp_path, i, 4, plan=plan)
    _refused(tmp_path, capsys, "dataset")


def test_merge_refuses_an_incomplete_shard(tmp_path, capsys):
    plan = _plan()
    short = agentic_shard_task_names(plan, 2, 4)[:-1]  # one trial never ran
    for i in range(4):
        result = _harbor_result(short, PASSED) if i == 2 else None
        _write_shard(tmp_path, i, 4, result=result)
    _refused(tmp_path, capsys, "trials")


def test_merge_refuses_a_trial_of_another_shards_task():
    plan = _plan()
    results = {
        i: _harbor_result(agentic_shard_task_names(plan, i, 2), PASSED)
        for i in range(2)
    }
    (stats,) = results[0]["stats"]["evals"].values()
    stats["reward_stats"]["reward"]["1.0"][0] = "task-01__zzz"  # shard 1's task
    with pytest.raises(ShardError, match="not one of the shard's tasks"):
        merge_agentic_shard_results(plan, 2, results)


def test_shard_set_check_refuses_a_tampered_task_list():
    plan = _plan()
    records = [
        agentic_shard_record(TASK, plan, i, 2, PROV, return_code=0) for i in range(2)
    ]
    records[1]["shard_task_names"] = records[1]["shard_task_names"][:-1]
    with pytest.raises(ShardError, match="slice of the plan"):
        check_agentic_shard_set(records)
