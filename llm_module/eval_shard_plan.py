# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Describe one lm-eval task for sharding: leaf doc counts, few-shot sampler,
metric aggregations and group structure.

Runs with the eval venv's python (it imports lm_eval), invoked by path like
lm_eval_request_overrides.py, before the sharded lm_eval run. It loads the task
exactly as lm_eval will, so the doc counts are the ones lm_eval indexes:

    <venv>/bin/python eval_shard_plan.py --tasks T [--include_path P]
        [--num_fewshot K] --out plan.json

The plan is shard-independent; workflows/eval_shard.py turns it into each
shard's --samples mapping and refuses plans that cannot shard exactly.
"""

from __future__ import annotations

import argparse
import json
import sys


def _aggregation_name(fn) -> str:
    if isinstance(fn, str):
        return fn
    return getattr(fn, "__name__", repr(fn))


def _leaf_info(task, num_fewshot_override):
    config = task.config
    fewshot_config = getattr(config, "fewshot_config", None) or {}
    shots = num_fewshot_override
    if shots is None:
        shots = getattr(config, "num_fewshot", None) or 0
    aggregations = {
        metric: _aggregation_name(fn)
        for metric, fn in (task.aggregation() or {}).items()
    }
    return {
        "n_docs": len(task.eval_docs),
        "num_fewshot": int(shots),
        "fewshot_sampler": fewshot_config.get("sampler", "default") if shots else None,
        "aggregations": aggregations,
        "repeats": int(getattr(config, "repeats", 1) or 1),
    }


def _group_info(group):
    config = getattr(group, "config", None)
    items = getattr(config, "aggregate_metric_list", None) or []
    aggregate = []
    for item in items:
        filters = getattr(item, "filter_list", None) or ["none"]
        aggregate.append(
            {
                "metric": getattr(item, "metric", None),
                "aggregation": _aggregation_name(getattr(item, "aggregation", "mean")),
                "weight_by_size": bool(getattr(item, "weight_by_size", False)),
                "filter_list": [filters] if isinstance(filters, str) else list(filters),
            }
        )
    return aggregate


def build_plan(task_dict, num_fewshot_override=None) -> dict:
    leaves: dict = {}
    groups: dict = {}

    def name_of(key) -> str:
        return key if isinstance(key, str) else getattr(key, "group_name", str(key))

    def walk(mapping) -> list:
        names = []
        for key, value in mapping.items():
            name = name_of(key)
            names.append(name)
            if isinstance(value, dict):
                groups[name] = {
                    "subtasks": walk(value),
                    "aggregate_metric_list": _group_info(key),
                }
            else:
                leaves[name] = _leaf_info(value, num_fewshot_override)
        return names

    walk(task_dict)
    return {"leaves": leaves, "groups": groups}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--include_path", default=None)
    parser.add_argument("--num_fewshot", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(include_path=args.include_path)
    task_dict = get_task_dict([args.tasks], manager)
    shots = None
    if args.num_fewshot not in (None, "", "None"):
        shots = int(args.num_fewshot)
    plan = build_plan(task_dict, shots)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
