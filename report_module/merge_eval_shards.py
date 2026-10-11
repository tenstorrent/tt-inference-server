# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merge N ``run.py --eval-shard i/N`` runs of one eval task and grade it.

This is the only way a sharded task can PASS. Each shard's own report is
PARTIAL by construction; this tool checks that the N shards are one complete,
disjoint run of one plan on one package/ref/image/command, recomputes every
metric from the shards' per-sample logs, writes the merged full-set results,
and grades them with the standard rule (published or GPU reference, noise-
aware) into the usual evals blocks and acceptance export.

    python -m report_module.merge_eval_shards --model <hf_repo> --task <task> \\
        --out <dir> <shard_dir> [<shard_dir> ...]

Each shard dir is searched recursively for ``eval_shard_<task>.json`` (written
by run.py next to that shard's lm-eval output), its ``results_*.json`` and its
``samples_<leaf>_*.jsonl``. Exit 0 PASS, 1 graded but not PASS, 2 the shards do
not form one complete run (nothing is graded).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from workflows.eval_shard import (
    SHARD_RECORD_PREFIX,
    ShardError,
    check_shard_set,
    merge_shard_samples,
)

MERGED_RESULTS_NAME = "results_merged.json"
MERGED_REPORT_NAME = "eval_shard_merge_report.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_shard_runs(dirs: Sequence[Path], task: str) -> List[Tuple[Path, dict]]:
    """(attempt dir, record) for every completed shard attempt under ``dirs``.

    An attempt counts only if lm-eval wrote results next to its record; an
    attempt that died before results is ignored, but two completed attempts of
    the same shard are both returned so the set check rejects the ambiguity.
    """
    found = []
    name = f"{SHARD_RECORD_PREFIX}{task}.json"
    for root in dirs:
        for record_path in sorted(Path(root).rglob(name)):
            attempt = record_path.parent
            if not list(attempt.rglob("results_*.json")):
                continue
            found.append((attempt, json.loads(record_path.read_text())))
    return found


def _results_file(attempt: Path) -> Path:
    files = sorted(attempt.rglob("results_*.json"))
    if len(files) != 1:
        raise ShardError(f"{attempt}: expected one results_*.json, found {len(files)}")
    return files[0]


def _samples_rows(attempt: Path, leaf: str) -> List[dict]:
    pattern = re.compile(rf"^samples_{re.escape(leaf)}_\d{{4}}-\d\d-\d\dT.*\.jsonl$")
    files = [p for p in attempt.rglob("samples_*.jsonl") if pattern.match(p.name)]
    if len(files) != 1:
        raise ShardError(
            f"{attempt}: expected one per-sample log for {leaf}, found {len(files)}"
        )
    with files[0].open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(dirs: Sequence[Path], task: str) -> Dict[str, Any]:
    """Verify and merge; returns the merged lm-eval results document."""
    runs = find_shard_runs(dirs, task)
    if not runs:
        raise ShardError(f"no completed shard of {task} under {list(map(str, dirs))}")
    plan = check_shard_set([record for _, record in runs])
    count = runs[0][1]["shard_count"]
    shard_rows: Dict[int, Dict[str, List[dict]]] = {}
    sources = []
    template = None
    for attempt, record in sorted(runs, key=lambda r: r[1]["shard_index"]):
        index = record["shard_index"]
        shard_rows[index] = {
            leaf: _samples_rows(attempt, leaf) for leaf in plan["leaves"]
        }
        results_path = _results_file(attempt)
        results_doc = json.loads(results_path.read_text())
        for leaf, info in plan["leaves"].items():
            original = (results_doc.get("n-samples", {}).get(leaf) or {}).get(
                "original"
            )
            if original is not None and original != info["n_docs"]:
                raise ShardError(
                    f"shard {record['shard']}: lm-eval saw {original} docs for {leaf}, "
                    f"plan says {info['n_docs']}"
                )
        if template is None:
            template = results_doc
        sources.append(
            {
                "shard": record["shard"],
                "attempt_dir": str(attempt),
                "results_sha256": _sha(results_path),
                "record_sha256": _sha(attempt / f"{SHARD_RECORD_PREFIX}{task}.json"),
            }
        )
    merged = merge_shard_samples(plan, count, shard_rows)
    doc = copy.deepcopy(template)
    for name, metrics in merged["results"].items():
        for section in ("results", "groups"):
            if name in doc.get(section, {}):
                alias = doc[section][name].get("alias", name)
                doc[section][name] = {**metrics, "alias": alias}
    doc["n-samples"] = {
        leaf: {"original": n, "effective": n} for leaf, n in merged["n_samples"].items()
    }
    first = runs[0][1]
    doc["eval_shard_merge"] = {
        "task_name": task,
        "shard_count": count,
        "plan_sha": first["plan_sha"],
        "provenance": first["provenance"],
        "doc_hashes": merged["doc_hashes"],
        "sources": sources,
    }
    return doc


def grade(
    model: str, task_name: str, doc: Mapping[str, Any], results_path: Path
) -> dict:
    """Grade the merged results exactly as a full run would be graded."""
    from reference_config.evals.eval_config import _eval_config_map
    from report_module.acceptance_criteria import (
        acceptance_criteria_check,
        build_acceptance_export,
    )
    from report_module.schema import ReportSchema
    from test_module.llm_tests.llm_eval_tests import blocks_for_task, load_eval_results

    config = _eval_config_map.get(model)
    tasks = {t.task_name: t for t in (config.tasks if config else [])}
    if task_name not in tasks:
        raise ShardError(f"{model} has no EvalConfig task {task_name}")
    provenance = doc["eval_shard_merge"]["provenance"]
    if provenance.get("hf_model_repo") not in (None, model):
        raise ShardError(
            f"shards are for {provenance.get('hf_model_repo')}, not {model}"
        )
    ctx = SimpleNamespace(
        runtime_config=None,  # the merged set is the full set: not a subset run
        model_spec=SimpleNamespace(
            hf_model_repo=model, model_name=model.split("/")[-1]
        ),
        device=SimpleNamespace(name=provenance.get("device") or "unknown"),
    )
    results, counts = load_eval_results([str(results_path)])
    blocks = blocks_for_task(ctx, tasks[task_name], results, counts)
    for block in blocks:
        block.data["eval_shard_merge"] = {
            "shard_count": doc["eval_shard_merge"]["shard_count"],
            "plan_sha": doc["eval_shard_merge"]["plan_sha"],
        }
    schema = ReportSchema(
        metadata={"model_repo": model, "report_id": f"merge_{task_name}"},
        sections=blocks,
    )
    accepted, blockers, categories = acceptance_criteria_check(
        schema,
        known_issues=provenance.get("known_issues") or None,
        model_status=provenance.get("model_status"),
    )
    return {
        "model": model,
        "task_name": task_name,
        "eval_shard_merge": doc["eval_shard_merge"],
        "evals": [
            {"title": b.title, "data": b.data, "targets": b.targets} for b in blocks
        ],
        **build_acceptance_export(
            accepted, blockers, categories, provenance.get("model_status")
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="HF repo id, e.g. openai/gpt-oss-120b"
    )
    parser.add_argument("--task", required=True, help="eval task name")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        doc = merge(args.shard_dirs, args.task)
    except ShardError as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / MERGED_RESULTS_NAME
    results_path.write_text(json.dumps(doc, indent=2, default=str))
    try:
        report = grade(args.model, args.task, doc, results_path)
    except ShardError as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    (args.out / MERGED_REPORT_NAME).write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(report["acceptance_summary_markdown"])
    return 0 if report["acceptance_criteria"] else 1


if __name__ == "__main__":
    sys.exit(main())
