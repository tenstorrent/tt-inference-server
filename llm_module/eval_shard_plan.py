# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Describe one lm-eval task for sharding: leaf doc counts, few-shot sampler,
metric aggregations and group structure.

Runs with the eval venv's python (it imports lm_eval), invoked by path like
lm_eval_request_overrides.py, before the sharded lm_eval run. It gets the eval's
exact lm_eval arguments and loads the tasks through the harness's own CLI
parser and EvaluatorConfig, so the doc set is the one lm_eval will index.
``--metadata`` merged with ``--model_args``, ``--include_path`` and
``--num_fewshot`` are all resolved by the harness itself, not re-implemented
here:

    <venv>/bin/python eval_shard_plan.py --out plan.json -- <lm_eval args...>

Exit status: 0 plan written; 3 this venv's harness cannot shard (its evaluator
has no ``samples`` doc selection, e.g. lm-eval 0.4.4), and nothing is written.

The plan is shard-independent; workflows/eval_shard.py turns it into each
shard's --samples mapping and refuses plans that cannot shard exactly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any, Mapping, Optional

UNSUPPORTED_HARNESS_RC = 3


def _aggregation_name(fn) -> str:
    if isinstance(fn, str):
        return fn
    return getattr(fn, "__name__", repr(fn))


def _get(obj: Any, key: str, default=None):
    """A field of a harness config that may be a dict or an object."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def harness_supports_samples() -> Optional[str]:
    """None if this harness can select docs by index; else why it cannot.

    Capability, not version: ``--samples`` reaches lm-eval through
    ``simple_evaluate(samples=...)``; a harness whose evaluator has no such
    parameter (lm-eval 0.4.4, the EVALS_META pin) would run every doc in each
    shard, so sharding must refuse it.
    """
    try:
        import lm_eval
        from lm_eval.evaluator import simple_evaluate
    except ImportError as exc:  # not an lm-eval venv at all
        return f"lm_eval is not importable in this venv ({exc})"
    if "samples" not in inspect.signature(simple_evaluate).parameters:
        version = getattr(lm_eval, "__version__", "unknown")
        return (
            f"lm-eval {version} has no doc selection (simple_evaluate takes no "
            "'samples'), so it cannot run one shard of a task"
        )
    return None


def load_task_dict(lm_eval_args):
    """(task_dict, num_fewshot) exactly as ``lm_eval <lm_eval_args>`` loads them."""
    from lm_eval._cli.harness import HarnessCLI
    from lm_eval.config.evaluate_config import EvaluatorConfig
    from lm_eval.tasks import get_task_dict

    saved = sys.argv
    try:
        # HarnessCLI.parse_args reads sys.argv and inserts the legacy 'run'.
        sys.argv = ["lm_eval", *lm_eval_args]
        args = HarnessCLI().parse_args()
    finally:
        sys.argv = saved
    cfg = EvaluatorConfig.from_cli(args)
    task_manager = cfg.process_tasks(cfg.metadata)
    return get_task_dict(cfg.tasks, task_manager), cfg.num_fewshot


def _leaf_info(task, num_fewshot_override):
    config = task.config
    fewshot_config = _get(config, "fewshot_config") or {}
    shots = num_fewshot_override
    if shots is None:
        shots = _get(config, "num_fewshot") or 0
    aggregations = {
        metric: _aggregation_name(fn)
        for metric, fn in (task.aggregation() or {}).items()
    }
    return {
        "n_docs": len(task.eval_docs),
        "num_fewshot": int(shots),
        "fewshot_sampler": _get(fewshot_config, "sampler", "default")
        if shots
        else None,
        "aggregations": aggregations,
        "repeats": int(_get(config, "repeats", 1) or 1),
    }


def _group_info(group):
    """The group's aggregate_metric_list, read from its config.

    The pinned harness's ``ConfigurableGroup.config`` is a dict
    (``GroupConfig.to_dict()``) whose entries are dicts too; older or test
    objects may expose attributes. Both are read; a callable aggregation comes
    back as its source text, which is not "mean" and so is refused later.
    """
    config = _get(group, "config") or {}
    items = _get(config, "aggregate_metric_list") or []
    if isinstance(items, Mapping) or not isinstance(items, (list, tuple)):
        items = [items]
    aggregate = []
    for item in items:
        filters = _get(item, "filter_list") or ["none"]
        aggregate.append(
            {
                "metric": _get(item, "metric"),
                "aggregation": _aggregation_name(_get(item, "aggregation", "mean")),
                "weight_by_size": bool(_get(item, "weight_by_size", False)),
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
                    # A plain-string key is a tag: lm-eval reports no aggregate for it.
                    "aggregate_metric_list": []
                    if isinstance(key, str)
                    else _group_info(key),
                }
            else:
                leaves[name] = _leaf_info(value, num_fewshot_override)
        return names

    walk(task_dict)
    return {"leaves": leaves, "groups": groups}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "lm_eval_args",
        nargs=argparse.REMAINDER,
        help="after --: the eval's lm_eval arguments, verbatim",
    )
    args = parser.parse_args(argv)
    lm_eval_args = list(args.lm_eval_args)
    if lm_eval_args[:1] == ["--"]:
        lm_eval_args = lm_eval_args[1:]
    if not lm_eval_args:
        parser.error("pass the eval's lm_eval arguments after --")

    why = harness_supports_samples()
    if why is not None:
        print(f"eval shard refused: {why}", file=sys.stderr)
        return UNSUPPORTED_HARNESS_RC
    task_dict, shots = load_task_dict(lm_eval_args)
    plan = build_plan(task_dict, shots)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
