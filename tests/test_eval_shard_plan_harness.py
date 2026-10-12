# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""The eval-shard planner against a real lm-eval (the EVALS_COMMON pin).

Skipped where lm_eval is not installed. Run from the eval venv, e.g.
``<evals-common venv>/bin/python -m pytest tests/test_eval_shard_plan_harness.py``.
The fixture tasks build their doc set from task metadata, like RULER.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("lm_eval")
pytest.importorskip("lm_eval._cli.harness")

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tests" / "fixtures" / "eval_shard_tasks"


def _planner():
    # By path: llm_module/__init__ pulls serving deps the eval venv lacks.
    spec = importlib.util.spec_from_file_location(
        "eval_shard_plan", ROOT / "llm_module" / "eval_shard_plan.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(tmp_path, *extra):
    out = tmp_path / "plan.json"
    args = [
        "--tasks",
        "shard_fixture_group",
        "--model",
        "local-completions",
        "--model_args",
        "model=m,base_url=http://127.0.0.1:1/v1/completions,shard_fixture_docs=7",
        "--include_path",
        str(TASKS),
        "--num_fewshot",
        "0",
        "--output_path",
        str(tmp_path / "out"),
        "--log_samples",
        *extra,
    ]
    assert _planner().main(["--out", str(out), "--", *args]) == 0
    return json.loads(out.read_text())


def test_metadata_selects_the_doc_set_the_run_will_see(tmp_path):
    # lm-eval merges --model_args into task metadata, then --metadata over it.
    base = _plan(tmp_path)
    assert {leaf: v["n_docs"] for leaf, v in base["leaves"].items()} == {
        "shard_fixture_leaf_a": 7,
        "shard_fixture_leaf_b": 7,
    }
    meta = _plan(tmp_path, "--metadata", json.dumps({"shard_fixture_docs": 9}))
    assert {v["n_docs"] for v in meta["leaves"].values()} == {9}


def test_group_aggregates_come_from_the_real_configurable_group(tmp_path):
    plan = _plan(tmp_path)
    assert plan["groups"]["shard_fixture_group"] == {
        "subtasks": ["shard_fixture_leaf_a", "shard_fixture_leaf_b"],
        "aggregate_metric_list": [
            {
                "metric": "exact_match",
                "aggregation": "mean",
                "weight_by_size": True,
                "filter_list": ["none"],
            }
        ],
    }


def test_this_harness_supports_doc_selection():
    assert _planner().harness_supports_samples() is None


def test_merge_matches_the_pinned_harness_group_aggregation():
    """The merged group score equals the harness's own consolidate_group_results
    on the same full-set leaf results, for a nested group."""
    import sys

    sys.path.insert(0, str(ROOT))
    from lm_eval.api.group import ConfigurableGroup
    from lm_eval.api.task import Task
    from lm_eval.evaluator_utils import consolidate_group_results

    from workflows.eval_shard import merge_shard_samples

    class _Leaf(Task):  # only identity matters to consolidate_group_results
        def __init__(self, name):
            self._name = name

        @property
        def task_name(self):
            return self._name

    _Leaf.__abstractmethods__ = frozenset()  # never run, only identified

    def group(name, weight_by_size):
        return ConfigurableGroup(
            config={
                "group": name,
                "aggregate_metric_list": [
                    {
                        "metric": "exact_match",
                        "aggregation": "mean",
                        "weight_by_size": weight_by_size,
                    }
                ],
            }
        )

    sizes = {"a": 10, "b": 7, "c": 4}
    score = {
        leaf: [float((d * 7 + len(leaf) * 3) % 3 == 0) for d in range(n)]
        for leaf, n in sizes.items()
    }
    for weight in (False, True):
        inner, outer = group("inner", True), group("outer", weight)
        task_dict = {
            outer: {inner: {"a": _Leaf("a"), "b": _Leaf("b")}, "c": _Leaf("c")}
        }
        results = {}
        for leaf, values in score.items():
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            results[leaf] = {
                "exact_match,none": mean,
                "exact_match_stderr,none": (var / len(values)) ** 0.5,
                "samples": len(values),
                "alias": leaf,
            }
        results.update({"inner": {"alias": "inner"}, "outer": {"alias": "outer"}})
        harness, _, _, _ = consolidate_group_results(results, {}, task_dict)

        plan = {
            "leaves": {
                leaf: {
                    "n_docs": n,
                    "num_fewshot": 0,
                    "fewshot_sampler": None,
                    "aggregations": {"exact_match": "mean"},
                    "repeats": 1,
                }
                for leaf, n in sizes.items()
            },
            "groups": {
                "outer": {
                    "subtasks": ["inner", "c"],
                    "aggregate_metric_list": [
                        {
                            "metric": "exact_match",
                            "aggregation": "mean",
                            "weight_by_size": weight,
                            "filter_list": ["none"],
                        }
                    ],
                },
                "inner": {
                    "subtasks": ["a", "b"],
                    "aggregate_metric_list": [
                        {
                            "metric": "exact_match",
                            "aggregation": "mean",
                            "weight_by_size": True,
                            "filter_list": ["none"],
                        }
                    ],
                },
            },
        }
        rows = {
            i: {
                leaf: [
                    {"doc_id": d, "filter": "none", "exact_match": v}
                    for d, v in enumerate(values)
                    if d % 3 == i
                ]
                for leaf, values in score.items()
            }
            for i in range(3)
        }
        merged = merge_shard_samples(plan, 3, rows)["results"]
        for name in ("inner", "outer"):
            assert merged[name]["exact_match,none"] == pytest.approx(
                harness[name]["exact_match,none"]
            ), (name, weight)
            assert merged[name]["exact_match_stderr,none"] == pytest.approx(
                harness[name]["exact_match_stderr,none"]
            ), (name, weight)
