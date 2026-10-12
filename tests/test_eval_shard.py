# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sample-sharded full evals: partition, refusals, merge == unsharded, and the
rule that only a merged full set can PASS."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from report_module import merge_eval_shards as merge_cli
from report_module.acceptance_criteria import (
    acceptance_criteria_check,
    build_acceptance_export,
)
from report_module.schema import ReportSchema
from test_module._test_common import TestStatus
from test_module.llm_tests import llm_eval_tests as mod
from workflows.eval_shard import (
    ShardError,
    check_plan,
    check_shard_set,
    merge_shard_samples,
    parse_eval_shard,
    plan_sha,
    shard_indices,
    shard_record,
    shard_samples,
)

# --- partition ----------------------------------------------------------------


@pytest.mark.parametrize("n_docs", [0, 1, 7, 30, 198, 541, 12032])
@pytest.mark.parametrize("count", [2, 3, 5, 7])
def test_shards_are_disjoint_balanced_and_cover_the_full_set(n_docs, count):
    shards = [shard_indices(n_docs, i, count) for i in range(count)]
    flat = [d for shard in shards for d in shard]
    assert sorted(flat) == list(range(n_docs))  # union is exactly the full set
    assert len(flat) == len(set(flat))  # pairwise disjoint
    assert all(shard == sorted(shard) for shard in shards)  # lm-eval maps back by order
    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1
    # deterministic: same answer every time
    assert shards == [shard_indices(n_docs, i, count) for i in range(count)]


@pytest.mark.parametrize(
    "value, expected", [("0/2", (0, 2)), ("4/5", (4, 5)), (" 1/3 ", (1, 3))]
)
def test_parse_eval_shard(value, expected):
    assert parse_eval_shard(value) == expected


@pytest.mark.parametrize(
    "value", ["1/1", "2/2", "3/2", "-1/2", "1", "a/b", "1/2/3", ""]
)
def test_parse_eval_shard_rejects(value):
    with pytest.raises(ShardError):
        parse_eval_shard(value)


# --- plan refusals --------------------------------------------------------------


def _leaf(n_docs, shots=0, sampler=None, aggregations=None):
    return {
        "n_docs": n_docs,
        "num_fewshot": shots,
        "fewshot_sampler": sampler,
        "aggregations": aggregations or {"exact_match": "mean"},
        "repeats": 1,
    }


def test_zero_shot_and_first_n_few_shot_plan_is_accepted():
    check_plan({"leaves": {"a": _leaf(5), "b": _leaf(5, 4, "first_n")}, "groups": {}})


@pytest.mark.parametrize("sampler", ["default", None, "random"])
def test_random_few_shot_is_refused(sampler):
    with pytest.raises(ShardError, match="few-shot"):
        check_plan({"leaves": {"a": _leaf(5, 5, sampler)}, "groups": {}})


def test_non_mean_metric_is_refused():
    plan = {"leaves": {"a": _leaf(5, aggregations={"bleu": "bleu"})}, "groups": {}}
    with pytest.raises(ShardError, match="not a per-sample mean"):
        check_plan(plan)


def test_non_mean_group_aggregate_is_refused():
    plan = {
        "leaves": {"a": _leaf(5)},
        "groups": {
            "g": {
                "subtasks": ["a"],
                "aggregate_metric_list": [
                    {"metric": "exact_match", "aggregation": "median"}
                ],
            }
        },
    }
    with pytest.raises(ShardError, match="group metric"):
        check_plan(plan)


# --- merge == unsharded -------------------------------------------------------

PLAN = {
    "leaves": {
        "sub_a": _leaf(10, aggregations={"exact_match": "mean", "acc": "mean"}),
        "sub_b": _leaf(7, aggregations={"exact_match": "mean", "acc": "mean"}),
        "sub_c": _leaf(4, aggregations={"exact_match": "mean", "acc": "mean"}),
    },
    "groups": {
        # nested: outer = mean over (inner, sub_c), inner = size-weighted (sub_a, sub_b)
        "outer": {
            "subtasks": ["inner", "sub_c"],
            "aggregate_metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "weight_by_size": False,
                    "filter_list": ["none", "strict"],
                }
            ],
        },
        "inner": {
            "subtasks": ["sub_a", "sub_b"],
            "aggregate_metric_list": [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "weight_by_size": True,
                    "filter_list": ["none", "strict"],
                }
            ],
        },
    },
}


def _rows(leaf, n_docs):
    rows = []
    for doc in range(n_docs):
        for filt in ("none", "strict"):
            salt = sum(map(ord, leaf + filt))
            rows.append(
                {
                    "doc_id": doc,
                    "filter": filt,
                    "exact_match": float((doc * 7 + salt) % 3 == 0),
                    "acc": ((doc + salt) % 5) / 4,
                    "doc_hash": f"{leaf}-{doc}",
                }
            )
    return rows


def _unsharded():
    """What lm-eval reports for the full set: leaf means and lm-eval group means."""
    leaf = {}
    for name, info in PLAN["leaves"].items():
        rows = _rows(name, info["n_docs"])
        leaf[name] = {}
        for filt in ("none", "strict"):
            for metric in ("exact_match", "acc"):
                values = [r[metric] for r in rows if r["filter"] == filt]
                leaf[name][f"{metric},{filt}"] = sum(values) / len(values)
    for filt in ("none", "strict"):
        key = f"exact_match,{filt}"
        a, b, c = (leaf[x][key] for x in ("sub_a", "sub_b", "sub_c"))
        inner = (a * 10 + b * 7) / 17
        leaf.setdefault("inner", {})[key] = inner
        # The harness flattens `outer` to its descendant leaves (sub_a, sub_b,
        # sub_c) and, unweighted, averages those -- not (inner + sub_c) / 2.
        leaf.setdefault("outer", {})[key] = (a + b + c) / 3
    return leaf


def _shard_rows(count):
    return {
        i: {
            leaf: [r for r in _rows(leaf, info["n_docs"]) if r["doc_id"] % count == i]
            for leaf, info in PLAN["leaves"].items()
        }
        for i in range(count)
    }


@pytest.mark.parametrize("count", [2, 3, 5])
def test_merge_equals_unsharded(count):
    merged = merge_shard_samples(PLAN, count, _shard_rows(count))
    expected = _unsharded()
    for name, metrics in expected.items():
        for key, value in metrics.items():
            assert merged["results"][name][key] == pytest.approx(value, abs=1e-12), (
                name,
                key,
            )
    assert merged["n_samples"] == {"sub_a": 10, "sub_b": 7, "sub_c": 4}


def test_merge_is_independent_of_shard_count():
    one = merge_shard_samples(PLAN, 2, _shard_rows(2))["results"]
    other = merge_shard_samples(PLAN, 5, _shard_rows(5))["results"]
    for name in one:
        for key in one[name]:
            if "stderr" not in key:
                assert one[name][key] == pytest.approx(other[name][key], abs=1e-12)


def test_missing_shard_fails():
    rows = _shard_rows(3)
    del rows[2]
    with pytest.raises(ShardError, match="shards 0..2"):
        merge_shard_samples(PLAN, 3, rows)


def test_missing_doc_fails():
    rows = _shard_rows(3)
    rows[1]["sub_a"] = [r for r in rows[1]["sub_a"] if r["doc_id"] != 4]
    with pytest.raises(ShardError, match="missing"):
        merge_shard_samples(PLAN, 3, rows)


def test_foreign_or_duplicate_doc_fails():
    rows = _shard_rows(3)
    rows[0]["sub_a"].append(dict(rows[1]["sub_a"][0]))  # shard 1's doc in shard 0
    with pytest.raises(ShardError, match="not one of its own docs"):
        merge_shard_samples(PLAN, 3, rows)
    rows = _shard_rows(3)
    rows[0]["sub_a"].append(dict(rows[0]["sub_a"][0]))
    with pytest.raises(ShardError, match="logged twice"):
        merge_shard_samples(PLAN, 3, rows)


# --- shard-set checks ---------------------------------------------------------

PROV = {
    "hf_model_repo": "org/model",
    "ttis_ref": "abc",
    "package_manifest_sha256": "m",
    "docker_image": "img",
    "eval_command_sha": "c",
    "device": "P300X2",
    "model_status": "EXPERIMENTAL",
}


def _records(count, prov=PROV):
    return [shard_record("t", PLAN, i, count, prov) for i in range(count)]


def test_complete_shard_set_passes_the_set_check():
    assert check_shard_set(_records(3)) == PLAN


def test_duplicate_or_missing_shard_record_fails():
    records = _records(3)
    with pytest.raises(ShardError, match="duplicated \\[1\\]"):
        check_shard_set([records[0], records[1], records[1]])
    with pytest.raises(ShardError, match="missing \\[2\\]"):
        check_shard_set(records[:2])


@pytest.mark.parametrize("key", sorted(PROV))
def test_shards_from_different_runs_fail(key):
    records = _records(2)
    records[1] = shard_record("t", PLAN, 1, 2, {**PROV, key: "other"})
    with pytest.raises(ShardError, match=key):
        check_shard_set(records)


def test_shards_of_different_plans_fail():
    other = json.loads(json.dumps(PLAN))
    other["leaves"]["sub_a"]["n_docs"] = 11
    records = [
        shard_record("t", PLAN, 0, 2, PROV),
        shard_record("t", other, 1, 2, PROV),
    ]
    assert plan_sha(PLAN) != plan_sha(other)
    with pytest.raises(ShardError, match="plan_sha"):
        check_shard_set(records)


def test_shard_samples_match_the_partition():
    samples = shard_samples(PLAN, 1, 3)
    assert samples == {"sub_a": [1, 4, 7], "sub_b": [1, 4], "sub_c": [1]}


# --- end to end: N shard dirs -> merge CLI -> standard grade ------------------

MODEL, TASK = "Qwen/Qwen2.5-7B", "leaderboard_ifeval"
METRIC = "prompt_level_strict_acc"
N_DOCS, CORRECT = 541, 156  # 156/541 = 28.84 vs published 28.28 -> PASS


def _ifeval_plan():
    return {
        "leaves": {TASK: _leaf(N_DOCS, aggregations={METRIC: "mean"})},
        "groups": {},
    }


def _write_shard(root: Path, index: int, count: int, plan, prov=PROV, correct=CORRECT):
    attempt = root / f"run-{index}" / "evals" / f"eval-attempt-{index}"
    out = attempt / "eval_x" / "org__model"
    out.mkdir(parents=True)
    record = shard_record(TASK, plan, index, count, {**prov, "hf_model_repo": MODEL})
    (attempt / f"eval_shard_{TASK}.json").write_text(json.dumps(record))
    docs = shard_indices(N_DOCS, index, count)
    rows = [
        {"doc_id": d, "filter": "none", METRIC: d < correct, "doc_hash": str(d)}
        for d in docs
    ]
    (out / f"samples_{TASK}_2026-10-11T00-00-00.000000.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    shard_score = sum(r[METRIC] for r in rows) / len(rows)
    (out / "results_2026-10-11T00-00-00.000000.json").write_text(
        json.dumps(
            {
                "results": {TASK: {"alias": TASK, f"{METRIC},none": shard_score}},
                "configs": {TASK: {"task": TASK, "dataset_path": "google/IFEval"}},
                # lm-eval reports the full count as effective under --samples
                "n-samples": {TASK: {"original": N_DOCS, "effective": N_DOCS}},
            }
        )
    )
    return root / f"run-{index}"


def test_merge_cli_end_to_end_passes_only_on_the_full_set(tmp_path):
    count = 3
    dirs = [_write_shard(tmp_path, i, count, _ifeval_plan()) for i in range(count)]
    out = tmp_path / "merged"
    assert (
        merge_cli.main(
            ["--model", MODEL, "--task", TASK, "--out", str(out), *map(str, dirs)]
        )
        == 0
    )
    doc = json.loads((out / merge_cli.MERGED_RESULTS_NAME).read_text())
    assert doc["results"][TASK][f"{METRIC},none"] == pytest.approx(CORRECT / N_DOCS)
    assert doc["n-samples"][TASK] == {"original": N_DOCS, "effective": N_DOCS}
    assert len(doc["eval_shard_merge"]["sources"]) == count
    report = json.loads((out / merge_cli.MERGED_REPORT_NAME).read_text())
    assert report["acceptance_criteria"] is True
    assert report["evals"][0]["data"]["score"] == pytest.approx(100 * CORRECT / N_DOCS)


def test_merge_cli_refuses_an_incomplete_set(tmp_path):
    dirs = [_write_shard(tmp_path, i, 3, _ifeval_plan()) for i in range(2)]
    out = tmp_path / "merged"
    assert (
        merge_cli.main(
            ["--model", MODEL, "--task", TASK, "--out", str(out), *map(str, dirs)]
        )
        == 2
    )
    assert not (out / merge_cli.MERGED_REPORT_NAME).exists()


def test_merge_cli_refuses_a_rerun_duplicate(tmp_path):
    dirs = [_write_shard(tmp_path, i, 2, _ifeval_plan()) for i in range(2)]
    again = _write_shard(tmp_path / "rerun", 1, 2, _ifeval_plan())
    out = tmp_path / "merged"
    assert (
        merge_cli.main(
            [
                "--model",
                MODEL,
                "--task",
                TASK,
                "--out",
                str(out),
                *map(str, dirs + [again]),
            ]
        )
        == 2
    )


def test_merge_cli_grades_a_failing_full_set_as_fail(tmp_path):
    prov = {**PROV, "model_status": "FUNCTIONAL"}  # evals enforced
    dirs = [
        _write_shard(tmp_path, i, 2, _ifeval_plan(), prov=prov, correct=60)
        for i in range(2)
    ]
    out = tmp_path / "merged"
    assert (
        merge_cli.main(
            ["--model", MODEL, "--task", TASK, "--out", str(out), *map(str, dirs)]
        )
        == 1
    )
    report = json.loads((out / merge_cli.MERGED_REPORT_NAME).read_text())
    assert report["acceptance_criteria_metadata"]["enforcement_result"] == "FAIL"


# --- a subset run is PARTIAL, never PASS --------------------------------------


def _subset_ctx(**runtime):
    config = None
    if runtime:
        config = SimpleNamespace(
            **{
                "limit_samples_mode": None,
                "eval_shard": None,
                "eval_samples": None,
                **runtime,
            }
        )
    return SimpleNamespace(
        runtime_config=config,
        model_spec=SimpleNamespace(hf_model_repo=MODEL, model_name="Qwen2.5-7B"),
        device=SimpleNamespace(name="P300X2"),
    )


def _graded(ctx, model_status=None):
    from reference_config.evals.eval_config import _eval_config_map

    task = next(t for t in _eval_config_map[MODEL].tasks if t.task_name == TASK)
    results = {TASK: {f"{METRIC},none": CORRECT / N_DOCS}}  # a passing score
    blocks = mod.blocks_for_task(ctx, task, results, {TASK: N_DOCS})
    accepted, blockers, categories = acceptance_criteria_check(
        ReportSchema(metadata={"report_id": "r"}, sections=blocks),
        model_status=model_status,
    )
    return blocks, accepted, build_acceptance_export(accepted, blockers, categories)


def test_full_run_with_a_passing_score_passes():
    blocks, accepted, export = _graded(_subset_ctx())
    assert accepted is True
    assert "status" not in blocks[0].data


@pytest.mark.parametrize(
    "runtime",
    [{"eval_shard": "0/3"}, {"eval_samples": json.dumps({TASK: [0, 1, 2]})}],
)
def test_subset_run_with_a_passing_score_is_partial_not_pass(runtime):
    blocks, accepted, export = _graded(_subset_ctx(**runtime))
    assert blocks[0].data["status"] == TestStatus.PARTIAL.value
    assert accepted is False
    assert export["acceptance_criteria_metadata"]["enforcement_result"] == "PARTIAL"
    assert "PARTIAL" in export["acceptance_summary_markdown"]
    assert "PASS`" not in export["acceptance_summary_markdown"].split("\n")[2]


def test_partial_blocks_acceptance_even_where_evals_are_informational():
    # EXPERIMENTAL makes eval failures informational; a subset still never passes.
    _, accepted, export = _graded(
        _subset_ctx(eval_shard="1/2"), model_status="EXPERIMENTAL"
    )
    assert accepted is False
    assert export["acceptance_criteria_metadata"]["enforcement_result"] == "PARTIAL"


def test_partial_is_not_a_process_failure():
    assert TestStatus.PARTIAL.is_blocking is False


# --- run.py --eval-shard -> lm-eval --samples ----------------------------------


def test_build_eval_command_plans_and_passes_the_shard(tmp_path, monkeypatch):
    from llm_module import eval_command
    from reference_config.evals.eval_config import EvalTask

    plan = {"leaves": {"gpqa_x": _leaf(10)}, "groups": {}}

    planned = []

    def fake_run(cmd, **kwargs):
        planned.append(cmd)
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(plan))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(eval_command.subprocess, "run", fake_run)
    monkeypatch.setattr(eval_command, "_ttis_ref", lambda: "deadbeef")
    model_spec = SimpleNamespace(
        model_id="m",
        model_name="m",
        hf_model_repo="org/m",
        docker_image="img",
        status=SimpleNamespace(name="EXPERIMENTAL"),
        device_model_spec=SimpleNamespace(
            max_context=131072,
            max_concurrency=32,
            eval_max_retries=0,
            env_vars={"QUETZAL_BUNDLE_MANIFEST_SHA256": "man"},
            known_issues=[],
        ),
    )
    task = EvalTask(
        task_name="gpqa_x", custom_dataset_kwargs={"max_seq_lengths": [4096]}
    )
    rc = SimpleNamespace(eval_shard="1/3", eval_samples=None, limit_samples_mode=None)
    cmd = eval_command.build_eval_command(
        task, model_spec, "P300X2", str(tmp_path), 8000, runtime_config=rc
    )
    assert json.loads(cmd[cmd.index("--samples") + 1]) == {"gpqa_x": [1, 4, 7]}
    assert "--limit" not in cmd
    record = json.loads((tmp_path / "eval_shard_gpqa_x.json").read_text())
    assert record["shard"] == "1/3"
    assert record["provenance"]["package_manifest_sha256"] == "man"
    assert record["provenance"]["ttis_ref"] == "deadbeef"
    # The planner sees the eval's own lm_eval arguments verbatim, so the
    # harness resolves the same doc set: --metadata (merged with --model_args
    # by lm-eval) and --tasks reach it exactly as they reach the run.
    (plan_cmd,) = planned
    forwarded = plan_cmd[plan_cmd.index("--") + 1 :]
    run_args = cmd[: cmd.index("--samples")]
    assert run_args[-len(forwarded) :] == forwarded
    assert forwarded[forwarded.index("--metadata") + 1] == json.dumps(
        {"max_seq_lengths": [4096]}
    )
    assert "--model_args" in forwarded and "--tasks" in forwarded


def test_build_eval_command_refuses_random_few_shot(tmp_path, monkeypatch):
    from llm_module import eval_command
    from reference_config.evals.eval_config import EvalTask

    plan = {"leaves": {"t": _leaf(10, 5, "default")}, "groups": {}}

    def fake_run(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(plan))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(eval_command.subprocess, "run", fake_run)
    model_spec = SimpleNamespace(
        model_id="m",
        model_name="m",
        hf_model_repo="org/m",
        device_model_spec=SimpleNamespace(
            max_context=8192, max_concurrency=1, eval_max_retries=0
        ),
    )
    rc = SimpleNamespace(eval_shard="0/2", eval_samples=None, limit_samples_mode=None)
    with pytest.raises(ShardError, match="few-shot"):
        eval_command.build_eval_command(
            EvalTask(task_name="t"),
            model_spec,
            "P300X2",
            str(tmp_path),
            8000,
            runtime_config=rc,
        )


# --- the per-task deadline bounds each shard -----------------------------------


def _shard_eval_ctx(tmp_path, eval_shard):
    from unittest.mock import MagicMock

    ctx = MagicMock()
    ctx.model_spec.model_name = "m"
    ctx.model_spec.hf_model_repo = "org/m"
    ctx.model_spec.device_model_spec.max_context = 131072
    ctx.device.name = "P300X2"
    ctx.output_path = str(tmp_path)
    ctx.remote_server = False
    ctx.server_host = "http://127.0.0.1"
    ctx.server_port = 8000
    ctx.model_spec.device_model_spec.tensor_cache_timeout = 1
    ctx.runtime_config = SimpleNamespace(
        eval_shard=eval_shard, eval_samples=None, limit_samples_mode=None
    )
    return ctx


@pytest.mark.parametrize("eval_shard", [None, "0/4", "3/4"])
def test_each_shard_carries_the_task_deadline_unchanged(
    tmp_path, monkeypatch, eval_shard
):
    # wall_clock_timeout_seconds is per run, so per shard: sharding never
    # divides or stretches it, and a shard gets the whole budget as written.
    from reference_config.evals.eval_config import EvalTask

    seen = {}
    monkeypatch.setattr(mod, "build_eval_command", lambda *a, **k: ["echo", "x"])
    monkeypatch.setattr(mod, "run_command", lambda **kw: seen.update(kw) or 0)
    task = EvalTask(task_name="gpqa_x", wall_clock_timeout_seconds=50400)
    rc = mod._run_eval_task(
        _shard_eval_ctx(tmp_path, eval_shard), task, "", output_path=tmp_path
    )
    assert rc == 0
    assert seen["timeout_seconds"] == 50400


def test_a_shard_past_its_deadline_is_incomplete_not_partial(tmp_path, monkeypatch):
    # A shard killed at its deadline is a FAIL (incomplete), never a PARTIAL
    # measurement: its partial per-sample logs must not look mergeable.
    from reference_config.evals.eval_config import EvalTask

    task = EvalTask(task_name="gpqa_x", wall_clock_timeout_seconds=60)
    ctx = _shard_eval_ctx(tmp_path, "1/4")
    server = SimpleNamespace(
        wait_for_healthy=lambda timeout: True,
        get_health=lambda: SimpleNamespace(status_code=200),
    )
    monkeypatch.setattr(mod, "get_llm_eval_tasks", lambda *a, **k: [task])
    monkeypatch.setattr(mod, "HttpServerController", lambda **kw: server)
    monkeypatch.setattr(mod, "_run_eval_task", lambda *a, **k: mod._DEADLINE_RC)
    monkeypatch.setattr(
        mod, "_score_task", lambda *a, **k: pytest.fail("a timed-out shard was scored")
    )
    blocks = [b for b in mod.run_llm_eval(ctx) if b.kind == "evals"]
    assert blocks, "the timed-out shard must still emit a block"
    for block in blocks:
        assert block.data["accuracy_check"] == mod.ReportCheckTypes.FAIL
        assert block.data.get("status") != TestStatus.PARTIAL.value
        assert "deadline" in block.data["error"]
        assert block.data["subprocess_rc"] == mod._DEADLINE_RC


# --- review #5378: group config shape, dataset metadata, harness capability ----


def _harness_shaped_group(aggregate_metric_list):
    """A stand-in with the pinned harness's ConfigurableGroup shape (be23028b).

    ``ConfigurableGroup.config`` returns ``GroupConfig.to_dict()``, i.e.
    ``dataclasses.asdict``: a dict whose aggregate entries are dicts, never
    objects with attributes.
    """
    from dataclasses import asdict, dataclass, field

    @dataclass
    class AggMetricConfig:
        metric: str = None
        aggregation: str = "mean"
        weight_by_size: bool = False
        filter_list: list = field(default_factory=lambda: ["none"])

    @dataclass
    class GroupConfig:
        group: str = None
        task: list = None
        aggregate_metric_list: list = None

    class ConfigurableGroup:
        def __init__(self, **config):
            items = config.pop("aggregate_metric_list", None)
            self._config = GroupConfig(
                **config,
                aggregate_metric_list=None
                if items is None
                else [AggMetricConfig(**i) for i in items],
            )

        @property
        def group_name(self):
            return self._config.group

        @property
        def config(self):
            return asdict(self._config)

    return ConfigurableGroup(
        group="g", task=["a", "b"], aggregate_metric_list=aggregate_metric_list
    )


def test_group_plan_reads_the_harness_dict_shaped_group_config():
    from llm_module.eval_shard_plan import build_plan

    group = _harness_shaped_group(
        [{"metric": "exact_match", "aggregation": "mean", "weight_by_size": True}]
    )
    assert isinstance(group.config, dict)
    leaf = SimpleNamespace(
        config={"num_fewshot": 0},
        eval_docs=[0, 1, 2],
        aggregation=lambda: {"exact_match": "mean"},
    )
    plan = build_plan({group: {"a": leaf, "b": leaf}})
    assert plan["groups"]["g"]["aggregate_metric_list"] == [
        {
            "metric": "exact_match",
            "aggregation": "mean",
            "weight_by_size": True,
            "filter_list": ["none"],
        }
    ]


def test_group_plan_from_the_real_pinned_configurable_group():
    group_mod = pytest.importorskip("lm_eval.api.group")
    from llm_module.eval_shard_plan import _group_info

    group = group_mod.ConfigurableGroup(
        config={
            "group": "g",
            "task": ["a"],
            "aggregate_metric_list": [
                {"metric": "acc", "aggregation": "mean", "weight_by_size": True}
            ],
        }
    )
    assert _group_info(group) == [
        {
            "metric": "acc",
            "aggregation": "mean",
            "weight_by_size": True,
            "filter_list": ["none"],
        }
    ]


GROUP_LEAVES = {"sub_a": 6, "sub_b": 3}


def _group_plan(aggregates):
    return {
        "leaves": {
            leaf: _leaf(n, aggregations={"exact_match": "mean"})
            for leaf, n in GROUP_LEAVES.items()
        },
        "groups": {
            "grp": {"subtasks": list(GROUP_LEAVES), "aggregate_metric_list": aggregates}
        },
    }


def _correct(leaf, doc):
    return float((doc + len(leaf)) % 2 == 0)


def _write_group_shard(root: Path, index: int, count: int, plan):
    attempt = root / f"run-{index}" / f"eval-attempt-{index}"
    out = attempt / "eval_x" / "org__model"
    out.mkdir(parents=True)
    record = shard_record("grp", plan, index, count, {**PROV, "hf_model_repo": MODEL})
    (attempt / "eval_shard_grp.json").write_text(json.dumps(record))
    results = {}
    for leaf, n_docs in GROUP_LEAVES.items():
        rows = [
            {"doc_id": d, "filter": "none", "exact_match": _correct(leaf, d)}
            for d in shard_indices(n_docs, index, count)
        ]
        (out / f"samples_{leaf}_2026-10-11T00-00-00.000000.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        results[leaf] = {
            "alias": leaf,
            "exact_match,none": sum(r["exact_match"] for r in rows) / len(rows),
        }
    # lm-eval's own group line for this shard: a shard-only value.
    results["grp"] = {"alias": "grp", "exact_match,none": 0.123456}
    (out / "results_2026-10-11T00-00-00.000000.json").write_text(
        json.dumps({"results": results, "groups": {"grp": results["grp"]}})
    )
    return root / f"run-{index}"


def test_merge_recomputes_the_group_aggregate_from_the_full_set(tmp_path):
    plan = _group_plan(
        [{"metric": "exact_match", "aggregation": "mean", "weight_by_size": True}]
    )
    dirs = [_write_group_shard(tmp_path, i, 2, plan) for i in range(2)]
    doc = merge_cli.merge(dirs, "grp")
    full = {
        leaf: sum(_correct(leaf, d) for d in range(n)) / n
        for leaf, n in GROUP_LEAVES.items()
    }
    expected = (full["sub_a"] * 6 + full["sub_b"] * 3) / 9
    for section in ("results", "groups"):
        assert doc[section]["grp"]["exact_match,none"] == pytest.approx(expected)
        assert doc[section]["grp"]["exact_match,none"] != 0.123456


def test_merge_refuses_a_group_metric_it_cannot_recompute(tmp_path):
    # What the dict-shaped config bug produced: a group whose aggregate list
    # came back empty. The merge must refuse, never keep the shard's value.
    plan = _group_plan([])
    dirs = [_write_group_shard(tmp_path, i, 2, plan) for i in range(2)]
    with pytest.raises(ShardError, match="cannot recompute"):
        merge_cli.merge(dirs, "grp")


def test_harness_without_samples_is_refused(monkeypatch):
    # lm-eval 0.4.4 (the EVALS_META pin) has no doc selection: its
    # simple_evaluate takes no `samples`. Detected from the harness itself.
    import sys
    import types

    from llm_module import eval_shard_plan

    def install(simple_evaluate, version):
        pkg = types.ModuleType("lm_eval")
        pkg.__version__ = version
        evaluator = types.ModuleType("lm_eval.evaluator")
        evaluator.simple_evaluate = simple_evaluate
        pkg.evaluator = evaluator
        monkeypatch.setitem(sys.modules, "lm_eval", pkg)
        monkeypatch.setitem(sys.modules, "lm_eval.evaluator", evaluator)

    def v044(model, model_args=None, tasks=None, limit=None, log_samples=True):
        pass

    def pinned(model, model_args=None, tasks=None, limit=None, samples=None):
        pass

    install(v044, "0.4.4")
    why = eval_shard_plan.harness_supports_samples()
    assert why is not None and "0.4.4" in why
    assert eval_shard_plan.main(["--out", "unused.json", "--", "--tasks", "t"]) == (
        eval_shard_plan.UNSUPPORTED_HARNESS_RC
    )
    install(pinned, "0.4.9")
    assert eval_shard_plan.harness_supports_samples() is None


def test_build_eval_command_refuses_a_venv_whose_harness_lacks_samples(
    tmp_path, monkeypatch
):
    from llm_module import eval_command
    from reference_config.evals.eval_config import EvalTask
    from workflow_module.engine_types import WorkflowVenvType

    monkeypatch.setattr(
        eval_command.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            returncode=eval_command._SHARD_PLAN_UNSUPPORTED_RC,
            stderr="eval shard refused: lm-eval 0.4.4 has no doc selection",
        ),
    )
    model_spec = SimpleNamespace(
        model_id="m",
        model_name="m",
        hf_model_repo="org/m",
        device_model_spec=SimpleNamespace(
            max_context=8192, max_concurrency=1, eval_max_retries=0
        ),
    )
    rc = SimpleNamespace(eval_shard="0/2", eval_samples=None, limit_samples_mode=None)
    task = EvalTask(task_name="t", workflow_venv_type=WorkflowVenvType.EVALS_META)
    with pytest.raises(ShardError, match="EVALS_META.*0.4.4"):
        eval_command.build_eval_command(
            task, model_spec, "P300X2", str(tmp_path), 8000, runtime_config=rc
        )
    assert not (tmp_path / "eval_shard_t.json").exists()


def test_merge_grades_errored_requests_incomplete_not_pass(tmp_path):
    # Same as the passing end-to-end case, but one shard's requests errored:
    # the merged set is not a measurement, whatever the scores add up to.
    count = 2
    dirs = [_write_shard(tmp_path, i, count, _ifeval_plan()) for i in range(count)]
    samples = next((tmp_path / "run-1").rglob(f"samples_{TASK}_*.jsonl"))
    rows = [json.loads(line) for line in samples.read_text().splitlines() if line]
    rows[0]["resps"] = [["__INFERENCE_ERROR__: ClientResponseError(status=400)"]]
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "merged"
    assert (
        merge_cli.main(
            ["--model", MODEL, "--task", TASK, "--out", str(out), *map(str, dirs)]
        )
        == 1
    )
    report = json.loads((out / merge_cli.MERGED_REPORT_NAME).read_text())
    assert report["acceptance_criteria"] is False
    (block,) = report["evals"]
    assert block["data"]["incomplete"] is True
    assert block["data"]["inference_error_samples"] == 1


def test_nested_group_flattens_to_leaves_like_the_harness():
    # The review's example: outer = inner(A, B) + C, unweighted. A and B score
    # 100%, C 0%. The pinned harness gives (100 + 100 + 0) / 3 = 66.67%, not
    # the (100 + 0) / 2 = 50% an immediate-children average would give.
    plan = {
        "leaves": {name: _leaf(4) for name in ("A", "B", "C")},
        "groups": {
            "outer": {
                "subtasks": ["inner", "C"],
                "aggregate_metric_list": [
                    {
                        "metric": "exact_match",
                        "aggregation": "mean",
                        "weight_by_size": False,
                        "filter_list": ["none"],
                    }
                ],
            },
            "inner": {"subtasks": ["A", "B"], "aggregate_metric_list": []},
        },
    }
    score = {"A": 1.0, "B": 1.0, "C": 0.0}
    rows = {
        i: {
            leaf: [
                {"doc_id": d, "filter": "none", "exact_match": score[leaf]}
                for d in range(4)
                if d % 2 == i
            ]
            for leaf in score
        }
        for i in range(2)
    }
    merged = merge_shard_samples(plan, 2, rows)
    assert merged["results"]["outer"]["exact_match,none"] == pytest.approx(2 / 3)
    assert "inner" not in merged["results"]  # no aggregate list: no group score


@pytest.mark.parametrize(
    "eval_samples, partial",
    [
        (json.dumps({TASK: None}), False),  # selected with every sample
        (json.dumps({"some_other_task": [0, 1]}), False),  # no entry for this task
        (json.dumps({TASK: [0, 1, 2]}), True),  # a real restriction
    ],
)
def test_partial_follows_the_resolved_sample_set_not_the_option(eval_samples, partial):
    blocks, accepted, export = _graded(_subset_ctx(eval_samples=eval_samples))
    assert (blocks[0].data.get("status") == TestStatus.PARTIAL.value) is partial
    assert accepted is (not partial)
