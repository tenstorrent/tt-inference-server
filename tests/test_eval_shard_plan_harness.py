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
