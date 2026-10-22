# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sample-sharded full evals: the partition, the shard record and the merge.

A task too long for one CI job runs as N shards (``run.py --eval-shard i/N``).
Each leaf lm-eval task has a doc set: every doc of the dataset, or, for a task
whose EvalConfig fixes it (``EvalTask.doc_ids``, e.g. the 131-doc LiveCodeBench
v6 window of a 1,055-doc dataset), exactly those ids. Shard i evaluates the
docs at positions ``p % N == i`` of that sorted set: deterministic, disjoint,
and together exactly the set an unsharded run evaluates. No sample is ever
capped. The set is part of the plan, so every shard record carries it and the
merge requires the shards' union to equal it.

A shard alone is never a verdict. ``merge_shard_samples`` joins N shards' per-
sample logs back into the full-set results after verifying the shards are
complete, disjoint and from the same plan; only that merged result is graded
(``python -m report_module.merge_eval_shards``).

Exactness is the rule, so sharding refuses what it cannot reproduce exactly:

- few-shot examples drawn by a random sampler (they depend on document order,
  so a shard would see different examples than an unsharded run); zero-shot
  and the fixed ``first_n`` sampler are order-independent;
- any metric whose aggregation is not a plain per-sample mean, at leaf or
  group level, since only a mean recombines exactly from shards.

Standard library only: run.py, the eval-command builder and the merge CLI all
import it, and none of them may need the eval venv.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SHARD_RECORD_PREFIX = "eval_shard_"
# 2: the plan carries a leaf's configured doc set (``doc_ids``) and each record
# its expected doc set; a version-1 shard of a doc_ids task partitioned the
# whole dataset instead, so it can never be merged.
SHARD_RECORD_VERSION = 2
ORDER_INDEPENDENT_SAMPLERS = frozenset({"first_n"})

_SHARD_RE = re.compile(r"^(\d+)/(\d+)$")


class ShardError(ValueError):
    """A shard plan or a set of shards that cannot give an exact full result."""


def parse_eval_shard(value: str) -> Tuple[int, int]:
    """``"i/N"`` -> ``(i, N)``, with 0 <= i < N and N >= 2."""
    match = _SHARD_RE.match(str(value).strip())
    if not match:
        raise ShardError(f"--eval-shard must look like i/N (e.g. 0/4); got {value!r}")
    index, count = int(match.group(1)), int(match.group(2))
    if count < 2:
        raise ShardError(f"--eval-shard needs N >= 2; got {value!r}")
    if index >= count:
        raise ShardError(f"--eval-shard needs 0 <= i < N; got {value!r}")
    return index, count


def shard_indices(n_docs: int, index: int, count: int) -> List[int]:
    """Sorted doc indices of shard ``index`` of ``count`` over ``n_docs`` docs."""
    if n_docs < 0:
        raise ShardError(f"doc count must be >= 0; got {n_docs}")
    return list(range(index, n_docs, count))


def leaf_doc_ids(info: Mapping[str, Any]) -> List[int]:
    """The sorted doc set an unsharded run of this plan leaf evaluates."""
    if "doc_ids" in info:
        return list(info["doc_ids"])
    return list(range(int(info["n_docs"])))


def shard_doc_ids(doc_ids: Sequence[int], index: int, count: int) -> List[int]:
    """Shard ``index`` of ``count`` of a sorted doc set: every count-th position."""
    return list(doc_ids)[index::count]


def restrict_plan_to_doc_ids(
    plan: Mapping[str, Any], task_name: str, doc_ids: Sequence[int]
) -> Dict[str, Any]:
    """The plan of a task whose EvalConfig fixes its doc set (EvalTask.doc_ids).

    An unsharded run sends ``--samples {task_name: doc_ids}``, so the set
    applies to the leaf named ``task_name``; it is recorded on that leaf, sorted
    and unique, so it is part of the plan (and its sha) every shard carries.
    Refused when the ids cannot be that leaf's set: the task is not a single
    leaf of that name, or an id is outside the leaf's dataset.
    """
    leaves = plan.get("leaves") or {}
    if set(leaves) != {task_name}:
        raise ShardError(
            f"{task_name}: its configured doc_ids select docs of the leaf task "
            f"{task_name!r}, but the plan's leaves are {sorted(leaves)}; cannot "
            "shard that doc set exactly"
        )
    ids = sorted(set(int(i) for i in doc_ids))
    if not ids:
        raise ShardError(f"{task_name}: configured doc_ids are empty")
    n_docs = int(leaves[task_name]["n_docs"])
    outside = [i for i in ids if not 0 <= i < n_docs]
    if outside:
        raise ShardError(
            f"{task_name}: configured doc_ids {outside[:5]} are outside the "
            f"dataset's {n_docs} docs"
        )
    restricted = json.loads(json.dumps(plan))
    restricted["leaves"][task_name]["doc_ids"] = ids
    return restricted


def check_configured_doc_ids(
    plan: Mapping[str, Any], task_name: str, doc_ids: Optional[Sequence[int]]
) -> None:
    """Refuse shards whose doc set is not the task's configured one.

    ``doc_ids`` is the task's EvalConfig doc set (None: the full dataset). A
    shard set planned over another set -- the whole dataset of a doc_ids task,
    a different window, or a window on a task that has none -- is a complete
    run of a different benchmark and must not be graded as this one.
    """
    planned = {
        leaf: list(info["doc_ids"])
        for leaf, info in (plan.get("leaves") or {}).items()
        if "doc_ids" in info
    }
    expected = {} if doc_ids is None else {task_name: sorted(set(doc_ids))}
    if planned != expected:

        def describe(sets):
            if not sets:
                return "every doc of each leaf"
            return ", ".join(
                f"{leaf}: {len(ids)} configured doc_ids (sha {indices_sha(ids)[:12]})"
                for leaf, ids in sorted(sets.items())
            )

        raise ShardError(
            f"{task_name}: the shards evaluate {describe(planned)}, but the task's "
            f"configured doc set is {describe(expected)}"
        )


def indices_sha(indices: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(indices)).encode()).hexdigest()


def plan_sha(plan: Mapping[str, Any]) -> str:
    """Hash of the shard-independent plan: same for every shard of one run set."""
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def check_plan(plan: Mapping[str, Any]) -> None:
    """Refuse a plan whose shards could not recombine to the unsharded result."""
    problems = []
    for leaf, info in sorted(plan.get("leaves", {}).items()):
        shots = info.get("num_fewshot") or 0
        sampler = info.get("fewshot_sampler")
        if shots > 0 and sampler not in ORDER_INDEPENDENT_SAMPLERS:
            problems.append(
                f"{leaf}: {shots}-shot with sampler {sampler!r} draws few-shot "
                "examples in document order, so shards would see different "
                "examples than an unsharded run"
            )
        for metric, aggregation in sorted(info.get("aggregations", {}).items()):
            if aggregation != "mean":
                problems.append(
                    f"{leaf}: metric {metric!r} aggregates by {aggregation!r}, "
                    "not a per-sample mean"
                )
    for group, info in sorted(plan.get("groups", {}).items()):
        for item in info.get("aggregate_metric_list") or []:
            if item.get("aggregation", "mean") != "mean":
                problems.append(
                    f"{group}: group metric {item.get('metric')!r} aggregates by "
                    f"{item.get('aggregation')!r}, not a mean"
                )
    if problems:
        raise ShardError(
            "eval task cannot be sharded exactly:\n  - " + "\n  - ".join(problems)
        )


def shard_samples(
    plan: Mapping[str, Any], index: int, count: int
) -> Dict[str, List[int]]:
    """lm-eval ``--samples`` mapping {leaf: sorted doc indices} for one shard."""
    check_plan(plan)
    return {
        leaf: shard_doc_ids(leaf_doc_ids(info), index, count)
        for leaf, info in sorted(plan["leaves"].items())
    }


def shard_record(
    task_name: str,
    plan: Mapping[str, Any],
    index: int,
    count: int,
    provenance: Mapping[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The per-shard record written next to the shard's lm-eval output.

    ``run_id`` is the run.py run (``RuntimeConfig.run_id``) that wrote it: the
    merge binds each shard's runtime model spec to its results by it. It is not
    provenance (every shard has its own) and is not compared across shards.
    """
    samples = shard_samples(plan, index, count)
    expected = {leaf: leaf_doc_ids(info) for leaf, info in plan["leaves"].items()}
    return {
        "version": SHARD_RECORD_VERSION,
        "task_name": task_name,
        "shard": f"{index}/{count}",
        "shard_index": index,
        "shard_count": count,
        "plan": dict(plan),
        "plan_sha": plan_sha(plan),
        "leaf_indices_sha": {leaf: indices_sha(ix) for leaf, ix in samples.items()},
        "leaf_sample_counts": {leaf: len(ix) for leaf, ix in samples.items()},
        # The doc set the N shards must cover together, exactly (the ids
        # themselves are in the plan when configured, else 0..n_docs-1).
        "expected_doc_ids_sha": {
            leaf: indices_sha(ids) for leaf, ids in sorted(expected.items())
        },
        "expected_doc_counts": {
            leaf: len(ids) for leaf, ids in sorted(expected.items())
        },
        "provenance": dict(provenance),
        "run_id": run_id,
    }


# --- merge -------------------------------------------------------------------

# Provenance that must agree across shards: one package on one ref and image.
IDENTITY_KEYS = (
    "hf_model_repo",
    "ttis_ref",
    "package_manifest_sha256",
    "docker_image",
    "eval_command_sha",
    "device",
    "model_status",
)


def check_shard_set(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Verify N shard records form one complete set; return the shared plan."""
    if not records:
        raise ShardError("no shard records")
    first = records[0]
    count = first.get("shard_count")
    problems = []
    for record in records:
        if record.get("version") != SHARD_RECORD_VERSION:
            problems.append(
                f"shard {record.get('shard')}: record version {record.get('version')!r}"
            )
        for key in ("task_name", "shard_count", "plan_sha", "expected_doc_ids_sha"):
            if record.get(key) != first.get(key):
                problems.append(
                    f"shard {record.get('shard')}: {key} {record.get(key)!r} != {first.get(key)!r}"
                )
        for key in IDENTITY_KEYS:
            mine = (record.get("provenance") or {}).get(key)
            theirs = (first.get("provenance") or {}).get(key)
            if mine != theirs:
                problems.append(
                    f"shard {record.get('shard')}: {key} {mine!r} != {theirs!r}"
                )
        if plan_sha(record.get("plan") or {}) != record.get("plan_sha"):
            problems.append(
                f"shard {record.get('shard')}: plan does not match its plan_sha"
            )
        planned_sets = {
            leaf: indices_sha(leaf_doc_ids(info))
            for leaf, info in sorted(
                ((record.get("plan") or {}).get("leaves") or {}).items()
            )
        }
        if planned_sets != record.get("expected_doc_ids_sha"):
            problems.append(
                f"shard {record.get('shard')}: expected doc set does not match its plan"
            )
    indices = sorted(record.get("shard_index") for record in records)
    if isinstance(count, int) and indices != list(range(count)):
        seen = [i for i in indices]
        missing = sorted(set(range(count)) - set(seen))
        duplicate = sorted({i for i in seen if seen.count(i) > 1})
        problems.append(
            f"shards must be exactly 0..{count - 1} once each; "
            f"missing {missing}, duplicated {duplicate}"
        )
    if problems:
        raise ShardError(
            "shard set is not one complete run:\n  - " + "\n  - ".join(problems)
        )
    check_plan(first["plan"])
    return first["plan"]


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    raise ShardError(
        f"doc {row.get('doc_id')}: metric {metric!r} is {value!r}, not a number"
    )


def _mean_and_stderr(values: Sequence[float]) -> Tuple[float, Optional[float]]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance / n)


def merge_shard_samples(
    plan: Mapping[str, Any],
    count: int,
    shard_rows: Mapping[int, Mapping[str, Iterable[Mapping[str, Any]]]],
) -> Dict[str, Any]:
    """Join N shards' per-sample rows into full-set leaf and group results.

    ``shard_rows[i][leaf]`` is shard i's logged samples for that leaf (lm-eval
    ``samples_<leaf>_*.jsonl`` rows: ``doc_id``, ``filter`` and one key per
    metric). Each shard must hold exactly its own doc indices and the union
    exactly the leaf's doc set (``leaf_doc_ids``: its configured doc_ids, else
    the full dataset); otherwise nothing is merged.

    Returns ``{"results": {name: {"<metric>,<filter>": value, ...}},
    "n_samples": {leaf: n}, "doc_hashes": {leaf: sha}}``.
    """
    leaves = plan["leaves"]
    if sorted(shard_rows) != list(range(count)):
        raise ShardError(
            f"need per-sample logs for shards 0..{count - 1}; got {sorted(shard_rows)}"
        )
    results: Dict[str, Dict[str, float]] = {}
    n_samples: Dict[str, int] = {}
    doc_hashes: Dict[str, str] = {}
    for leaf, info in sorted(leaves.items()):
        doc_ids = leaf_doc_ids(info)
        metrics = sorted(info.get("aggregations", {}))
        by_filter: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        for index in range(count):
            expected = set(shard_doc_ids(doc_ids, index, count))
            rows = list(shard_rows[index].get(leaf, ()))
            for row in rows:
                doc_id, filt = row.get("doc_id"), row.get("filter", "none")
                if doc_id not in expected:
                    raise ShardError(
                        f"{leaf}: shard {index}/{count} logged doc {doc_id}, "
                        "which is not one of its own docs"
                    )
                slot = by_filter.setdefault(filt, {})
                if doc_id in slot:
                    raise ShardError(
                        f"{leaf}: doc {doc_id} (filter {filt}) logged twice"
                    )
                slot[doc_id] = row
        if not by_filter:
            raise ShardError(f"{leaf}: no per-sample rows in any shard")
        for filt, rows_by_doc in sorted(by_filter.items()):
            missing = sorted(set(doc_ids) - set(rows_by_doc))
            if missing:
                raise ShardError(
                    f"{leaf}: filter {filt} is missing {len(missing)} of "
                    f"{len(doc_ids)} docs (first {missing[:5]})"
                )
            entry = results.setdefault(leaf, {})
            ordered = [rows_by_doc[d] for d in doc_ids]
            for metric in metrics:
                mean, stderr = _mean_and_stderr(
                    [_metric_value(r, metric) for r in ordered]
                )
                entry[f"{metric},{filt}"] = mean
                entry[f"{metric}_stderr,{filt}"] = (
                    stderr if stderr is not None else "N/A"
                )
            doc_hashes[leaf] = hashlib.sha256(
                "".join(str(r.get("doc_hash", "")) for r in ordered).encode()
            ).hexdigest()
        n_samples[leaf] = len(doc_ids)
    groups = plan.get("groups", {})

    def leaves_of(group: str, stack: Tuple[str, ...] = ()) -> List[str]:
        """Descendant leaf tasks in order, as the harness flattens a group."""
        if group in stack:
            raise ShardError(f"group cycle at {group}")
        out: List[str] = []
        for sub in groups[group].get("subtasks") or []:
            if sub in groups:
                out.extend(leaves_of(sub, stack + (group,)))
            elif sub in leaves:
                out.append(sub)
            else:
                raise ShardError(f"{group}: unknown subtask {sub}")
        return out

    # The pinned harness (lm_eval/evaluator_utils.py consolidate_group_results,
    # tstescoTT be23028b) aggregates a group over ALL its descendant leaf tasks
    # -- a nested subgroup is flattened, never averaged as one child -- using
    # aggregate_subtask_metrics(metrics, sizes, weight_by_size) over the leaves
    # that report the metric, and pooled_sample_stderr(stderrs, sizes).
    for group in sorted(groups):
        aggregates = groups[group].get("aggregate_metric_list") or []
        if not aggregates:
            continue
        task_list = leaves_of(group)
        entry = results.setdefault(group, {})
        for item in aggregates:
            metric = item["metric"]
            filters = item.get("filter_list") or ["none"]
            for filt in [filters] if isinstance(filters, str) else filters:
                key = f"{metric},{filt}"
                stderr_key = f"{metric}_stderr,{filt}"
                present = [t for t in task_list if key in results.get(t, {})]
                if not present:
                    raise ShardError(f"{group}: no leaf task reports {key}")
                metrics = [results[t][key] for t in present]
                sizes = [n_samples[t] for t in present]
                weights = sizes if item.get("weight_by_size") else [1] * len(sizes)
                entry[key] = sum(m * w for m, w in zip(metrics, weights)) / sum(weights)
                stderrs = [results[t].get(stderr_key, "N/A") for t in present]
                if "N/A" in stderrs or sum(sizes) <= len(sizes):
                    entry[stderr_key] = "N/A"
                else:
                    pooled = sum((n - 1) * e**2 * n for n, e in zip(sizes, stderrs)) / (
                        sum(sizes) - len(sizes)
                    )
                    entry[stderr_key] = math.sqrt(pooled / sum(sizes))
    return {"results": results, "n_samples": n_samples, "doc_hashes": doc_hashes}


# --- agentic (Harbor) tasks --------------------------------------------------
#
# An agentic eval's unit is a dataset task (one Harbor trial per attempt), not
# an lm-eval doc. The plan is the task set an unsharded run evaluates -- the
# dataset resolved through the run's own Harbor filters, in Harbor's order --
# sorted by name; shard i runs the names at positions ``p % N == i`` as
# Harbor ``task_names``. Harbor's score is the mean reward over trials, so the
# full-set accuracy is the sum of every shard's rewards over the full trial
# count, which recombines exactly.

AGENTIC_RECORD_KIND = "agentic"
AGENTIC_SHARD_RECORD_VERSION = 1


def agentic_shard_plan(
    *,
    dataset: str,
    dataset_ref: Optional[str],
    resolved_ref: Optional[str],
    task_names: Iterable[str],
    n_attempts: int,
) -> Dict[str, Any]:
    """The shard-independent plan of an agentic task: its sorted task set."""
    names = [str(name) for name in task_names]
    if not names:
        raise ShardError(f"{dataset}: the task selection resolved to no tasks")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ShardError(f"{dataset}: task names listed twice: {duplicates[:5]}")
    if (
        not isinstance(n_attempts, int)
        or isinstance(n_attempts, bool)
        or n_attempts < 1
    ):
        raise ShardError(f"{dataset}: n_attempts must be >= 1; got {n_attempts!r}")
    return {
        "kind": AGENTIC_RECORD_KIND,
        "dataset": dataset,
        "dataset_ref": dataset_ref,
        "resolved_ref": resolved_ref,
        "n_attempts": n_attempts,
        "task_names": sorted(names),
    }


def agentic_shard_task_names(
    plan: Mapping[str, Any], index: int, count: int
) -> List[str]:
    """Shard ``index`` of ``count``: every count-th name of the sorted set.

    An empty shard is refused: Harbor reads an empty ``task_names`` as "no
    filter", so it would run the whole dataset.
    """
    names = list(plan["task_names"])[index::count]
    if not names:
        raise ShardError(
            f"{plan.get('dataset')}: shard {index}/{count} of "
            f"{len(plan['task_names'])} tasks is empty; use fewer shards"
        )
    return names


def agentic_shard_record(
    task_name: str,
    plan: Mapping[str, Any],
    index: int,
    count: int,
    provenance: Mapping[str, Any],
    *,
    return_code: int,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """The per-shard record written into the shard's Harbor job dir."""
    names = agentic_shard_task_names(plan, index, count)
    return {
        "kind": AGENTIC_RECORD_KIND,
        "version": AGENTIC_SHARD_RECORD_VERSION,
        "task_name": task_name,
        "shard": f"{index}/{count}",
        "shard_index": index,
        "shard_count": count,
        "plan": dict(plan),
        "plan_sha": plan_sha(plan),
        "shard_task_names": names,
        "shard_task_count": len(names),
        "expected_task_count": len(plan["task_names"]),
        "return_code": return_code,
        "provenance": dict(provenance),
        "run_id": run_id,
    }


def check_agentic_shard_set(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Verify N agentic shard records form one complete set; return the plan.

    The same identity rules as :func:`check_shard_set`: one task, one plan,
    one package/ref/image/command/device/status, shards 0..N-1 once each, and
    each shard's task list exactly its slice of the plan.
    """
    if not records:
        raise ShardError("no shard records")
    first = records[0]
    count = first.get("shard_count")
    problems = []
    for record in records:
        shard = record.get("shard")
        if record.get("kind") != AGENTIC_RECORD_KIND:
            problems.append(f"shard {shard}: record kind {record.get('kind')!r}")
        if record.get("version") != AGENTIC_SHARD_RECORD_VERSION:
            problems.append(f"shard {shard}: record version {record.get('version')!r}")
        for key in ("task_name", "shard_count", "plan_sha"):
            if record.get(key) != first.get(key):
                problems.append(
                    f"shard {shard}: {key} {record.get(key)!r} != {first.get(key)!r}"
                )
        for key in IDENTITY_KEYS:
            mine = (record.get("provenance") or {}).get(key)
            theirs = (first.get("provenance") or {}).get(key)
            if mine != theirs:
                problems.append(f"shard {shard}: {key} {mine!r} != {theirs!r}")
        plan = record.get("plan") or {}
        if plan_sha(plan) != record.get("plan_sha"):
            problems.append(f"shard {shard}: plan does not match its plan_sha")
            continue
        index = record.get("shard_index")
        if (
            isinstance(index, int)
            and isinstance(count, int)
            and 0 <= index < count
            and list(record.get("shard_task_names") or [])
            != list(plan.get("task_names") or [])[index::count]
        ):
            problems.append(f"shard {shard}: task list is not its slice of the plan")
    indices = sorted(record.get("shard_index") for record in records)
    if isinstance(count, int) and indices != list(range(count)):
        missing = sorted(set(range(count)) - set(indices))
        duplicate = sorted({i for i in indices if indices.count(i) > 1})
        problems.append(
            f"shards must be exactly 0..{count - 1} once each; "
            f"missing {missing}, duplicated {duplicate}"
        )
    if problems:
        raise ShardError(
            "shard set is not one complete run:\n  - " + "\n  - ".join(problems)
        )
    return first["plan"]


def _harbor_trial_prefix(task_name: str) -> str:
    """The task part of a Harbor trial name (``<task[:32]>__<uuid>``)."""
    return task_name.split("/")[-1][:32].rstrip("_-")


def _count(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def merge_agentic_shard_results(
    plan: Mapping[str, Any],
    count: int,
    shard_results: Mapping[int, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Join N shards' Harbor ``result.json`` into one full-set Harbor result.

    Each shard must have finished every trial of exactly its own tasks (no
    pending/running trial, and every rewarded trial one of its tasks); the
    merged accuracy is the sum of rewards over the full trial count, as Harbor
    scores an unsharded run (a trial without a reward counts 0). Returns a
    Harbor-format result document (``stats.evals`` with one entry) that the
    agentic result parser reads as it reads a run's ``result.json``.
    """
    if sorted(shard_results) != list(range(count)):
        raise ShardError(
            f"need Harbor results for shards 0..{count - 1}; got {sorted(shard_results)}"
        )
    n_attempts = int(plan["n_attempts"])
    total_trials = len(plan["task_names"]) * n_attempts
    eval_key = None
    reward_key = None
    rewards: Dict[float, List[str]] = {}
    n_rewarded = n_errors = 0
    for index in range(count):
        raw = shard_results[index]
        names = agentic_shard_task_names(plan, index, count)
        where = f"shard {index}/{count}"
        expected = len(names) * n_attempts
        if _count(raw.get("n_total_trials")) != expected:
            raise ShardError(
                f"{where}: Harbor ran {raw.get('n_total_trials')!r} trials, "
                f"its {len(names)} tasks x {n_attempts} attempts need {expected}"
            )
        stats = raw.get("stats") or {}
        completed = _count(stats.get("n_completed_trials", stats.get("n_trials")))
        if completed != expected:
            raise ShardError(f"{where}: {completed!r} of {expected} trials completed")
        for key in ("n_running_trials", "n_pending_trials"):
            if _count(stats.get(key)) not in (None, 0):
                raise ShardError(f"{where}: {stats.get(key)} trial(s) {key[2:-7]}")
        evals = stats.get("evals") or {}
        if len(evals) != 1:
            raise ShardError(
                f"{where}: expected one agent/model/dataset eval, found {sorted(evals)}"
            )
        ((key, eval_stats),) = evals.items()
        if eval_key is None:
            eval_key = key
        elif key != eval_key:
            raise ShardError(f"{where}: eval {key!r} != shard 0's {eval_key!r}")
        reward_stats = eval_stats.get("reward_stats") or {}
        if len(reward_stats) > 1:
            raise ShardError(
                f"{where}: several reward keys {sorted(reward_stats)}; Harbor's "
                "accuracy is then per key, not one mean"
            )
        prefixes = {_harbor_trial_prefix(name) for name in names}
        rewarded = 0
        for rkey, by_value in reward_stats.items():
            if reward_key is None:
                reward_key = rkey
            elif rkey != reward_key:
                raise ShardError(f"{where}: reward key {rkey!r} != {reward_key!r}")
            for value, trials in (by_value or {}).items():
                try:
                    reward = float(value)
                except (TypeError, ValueError):
                    raise ShardError(
                        f"{where}: reward {value!r} is not a number"
                    ) from None
                if not math.isfinite(reward):
                    raise ShardError(f"{where}: reward {value!r} is not finite")
                for trial in trials or []:
                    if str(trial).rsplit("__", 1)[0] not in prefixes:
                        raise ShardError(
                            f"{where}: trial {trial!r} is not one of the shard's tasks"
                        )
                    rewards.setdefault(reward, []).append(str(trial))
                    rewarded += 1
        trials_rewarded = _count(eval_stats.get("n_trials"))
        if trials_rewarded is not None and trials_rewarded != rewarded:
            raise ShardError(
                f"{where}: n_trials {trials_rewarded} != {rewarded} rewarded trials"
            )
        if rewarded > expected:
            raise ShardError(f"{where}: {rewarded} rewards for {expected} trials")
        n_rewarded += rewarded
        n_errors += _count(eval_stats.get("n_errors")) or 0
    reward_sum = sum(value * len(trials) for value, trials in rewards.items())
    accuracy = reward_sum / total_trials
    return {
        "_result_format": "harbor",
        "n_total_trials": total_trials,
        "stats": {
            "n_completed_trials": total_trials,
            "n_errored_trials": n_errors,
            "evals": {
                eval_key: {
                    "n_trials": n_rewarded,
                    "n_errors": n_errors,
                    "metrics": [{"mean": accuracy}],
                    "reward_stats": (
                        {
                            reward_key: {
                                str(value): sorted(trials)
                                for value, trials in sorted(rewards.items())
                            }
                        }
                        if reward_key is not None
                        else {}
                    ),
                }
            },
        },
    }


__all__ = [
    "AGENTIC_RECORD_KIND",
    "AGENTIC_SHARD_RECORD_VERSION",
    "agentic_shard_plan",
    "agentic_shard_record",
    "agentic_shard_task_names",
    "check_agentic_shard_set",
    "merge_agentic_shard_results",
    "ShardError",
    "SHARD_RECORD_PREFIX",
    "check_configured_doc_ids",
    "check_plan",
    "check_shard_set",
    "indices_sha",
    "leaf_doc_ids",
    "merge_shard_samples",
    "parse_eval_shard",
    "plan_sha",
    "restrict_plan_to_doc_ids",
    "shard_doc_ids",
    "shard_indices",
    "shard_record",
    "shard_samples",
]
