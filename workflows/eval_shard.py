# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Sample-sharded full evals: the partition, the shard record and the merge.

A task too long for one CI job runs as N shards (``run.py --eval-shard i/N``).
Shard i evaluates, for every leaf lm-eval task, the docs whose index d has
``d % N == i``: deterministic, disjoint, and together exactly the full set.
No sample is ever capped.

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
SHARD_RECORD_VERSION = 1
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
        leaf: shard_indices(int(info["n_docs"]), index, count)
        for leaf, info in sorted(plan["leaves"].items())
    }


def shard_record(
    task_name: str,
    plan: Mapping[str, Any],
    index: int,
    count: int,
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """The per-shard record written next to the shard's lm-eval output."""
    samples = shard_samples(plan, index, count)
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
        "provenance": dict(provenance),
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
        for key in ("task_name", "shard_count", "plan_sha"):
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
    exactly the leaf's full set; otherwise nothing is merged.

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
        n_docs = int(info["n_docs"])
        metrics = sorted(info.get("aggregations", {}))
        by_filter: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        for index in range(count):
            expected = set(shard_indices(n_docs, index, count))
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
            missing = sorted(set(range(n_docs)) - set(rows_by_doc))
            if missing:
                raise ShardError(
                    f"{leaf}: filter {filt} is missing {len(missing)} of {n_docs} "
                    f"docs (first {missing[:5]})"
                )
            entry = results.setdefault(leaf, {})
            ordered = [rows_by_doc[d] for d in range(n_docs)]
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
        n_samples[leaf] = n_docs
    groups = plan.get("groups", {})
    done: set = set()

    def aggregate(group: str, stack: Tuple[str, ...] = ()) -> None:
        if group in done:
            return
        if group in stack:
            raise ShardError(f"group cycle at {group}")
        info = groups[group]
        subtasks = info.get("subtasks") or []
        for sub in subtasks:
            if sub in groups:
                aggregate(sub, stack + (group,))
        done.add(group)
        aggregates = info.get("aggregate_metric_list") or []
        if not aggregates:
            return
        entry = results.setdefault(group, {})
        for item in aggregates:
            metric = item["metric"]
            filters = item.get("filter_list") or ["none"]
            for filt in [filters] if isinstance(filters, str) else filters:
                key = f"{metric},{filt}"
                values, weights = [], []
                for sub in subtasks:
                    if key not in results.get(sub, {}):
                        raise ShardError(f"{group}: subtask {sub} has no {key}")
                    values.append(results[sub][key])
                    weights.append(
                        _group_size(sub, plan, n_samples)
                        if item.get("weight_by_size")
                        else 1
                    )
                entry[key] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                entry[f"{metric}_stderr,{filt}"] = "N/A"

    for group in sorted(groups):
        aggregate(group)
    return {"results": results, "n_samples": n_samples, "doc_hashes": doc_hashes}


def _group_size(
    name: str, plan: Mapping[str, Any], n_samples: Mapping[str, int]
) -> int:
    if name in n_samples:
        return n_samples[name]
    group = plan.get("groups", {}).get(name)
    if not group:
        raise ShardError(f"unknown subtask {name}")
    return sum(_group_size(sub, plan, n_samples) for sub in group.get("subtasks") or [])


__all__ = [
    "ShardError",
    "SHARD_RECORD_PREFIX",
    "check_plan",
    "check_shard_set",
    "indices_sha",
    "merge_shard_samples",
    "parse_eval_shard",
    "plan_sha",
    "shard_indices",
    "shard_record",
    "shard_samples",
]
