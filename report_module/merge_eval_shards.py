# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Merge N ``run.py --eval-shard i/N`` runs of one eval task and grade it.

This is the only way a sharded task can PASS. Each shard's own report is
PARTIAL by construction; this tool checks that the N shards are one complete,
disjoint run of one plan on one package/ref/image/command, over exactly the
task's configured doc set (its EvalConfig ``doc_ids``, else the full dataset),
recomputes every
metric from the shards' per-sample logs, writes the merged full-set results,
and grades them with the standard rule (published or GPU reference, noise-
aware) into the usual evals blocks and acceptance export.

    python -m report_module.merge_eval_shards --model <hf_repo> --task <task> \\
        --out <dir> <shard_dir> [<shard_dir> ...]

Each shard dir is searched recursively for ``eval_shard_<task>.json`` (written
by run.py next to that shard's lm-eval output), its ``results_*.json`` and its
``samples_<leaf>_*.jsonl``. Exit 0 PASS, 1 graded but not PASS, 2 the shards do
not form one complete run (nothing is graded).

Besides its own ``results_merged.json`` and ``eval_shard_merge_report.json``,
the merge writes the two files an unsharded ``run.py --workflow evals`` writes,
with run.py's own writers and in run.py's layout under ``--out``:

- ``reports_output/evals/data/report_data_<id>.json`` (and its ``.md``), by
  ``ReportGenerator`` after the evals workflow's stage tagging and metadata
  injection;
- ``runtime_model_specs/runtime_model_spec_<...>.json``, by
  ``RuntimeConfig.to_json``: the runtime model spec of the merged shards
  (found under the shard dirs and bound to each merged shard's results by the
  run id in its shard record; see :func:`find_shard_runtime_model_spec`) with
  ``eval_shard`` None, ``eval_tasks`` the merged task and ``eval_shard_merge``
  the merge provenance.

So a CI that collects a run.py run collects the merged verdict the same way.
Without a shard runtime model spec under the shard dirs these two files are
not written (a report with no run config could not be told apart from an
unsharded run); the grade and exit code are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from workflows.eval_shard import (
    SHARD_RECORD_PREFIX,
    ShardError,
    check_configured_doc_ids,
    check_shard_set,
    leaf_doc_ids,
    merge_shard_samples,
    parse_eval_shard,
)

MERGED_RESULTS_NAME = "results_merged.json"
MERGED_REPORT_NAME = "eval_shard_merge_report.json"
# run.py's layout under its workflow log root: the evals workflow's report dir
# (ReportGenerator writes data/report_data_<id>.json there) and the runtime
# model spec dir (RuntimeConfig.to_json).
RUN_REPORT_DIR = Path("reports_output") / "evals"
RUN_RUNTIME_MODEL_SPEC_DIR = Path("runtime_model_specs")
RUNTIME_MODEL_SPEC_GLOB = "runtime_model_spec_*.json"


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


def eval_task(model: str, task_name: str):
    """The EvalConfig task the shards are merged and graded as."""
    from reference_config.evals.eval_config import _eval_config_map

    config = _eval_config_map.get(model)
    tasks = {t.task_name: t for t in (config.tasks if config else [])}
    if task_name not in tasks:
        raise ShardError(f"{model} has no EvalConfig task {task_name}")
    return tasks[task_name]


def merge(
    dirs: Sequence[Path], task: str, doc_ids: Optional[Sequence[int]] = None
) -> Dict[str, Any]:
    """Verify and merge; returns the merged lm-eval results document.

    ``doc_ids`` is the task's configured doc set (``EvalTask.doc_ids``; None
    for the full dataset). The shards' plan must be over exactly that set, so a
    complete run of another set is refused rather than graded as this task.
    """
    runs = find_shard_runs(dirs, task)
    if not runs:
        raise ShardError(f"no completed shard of {task} under {list(map(str, dirs))}")
    plan = check_shard_set([record for _, record in runs])
    check_configured_doc_ids(plan, task, doc_ids)
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
                "run_id": record.get("run_id"),
                "attempt_dir": str(attempt),
                "results_sha256": _sha(results_path),
                "record_sha256": _sha(attempt / f"{SHARD_RECORD_PREFIX}{task}.json"),
            }
        )
    merged = merge_shard_samples(plan, count, shard_rows)
    # An errored request is scored as a wrong answer; count them so the grade
    # can refuse to treat the merged set as a measurement.
    from test_module.llm_tests.llm_eval_tests import INFERENCE_ERROR_MARKER

    logged = errored = 0
    for rows_by_leaf in shard_rows.values():
        for rows in rows_by_leaf.values():
            for row in rows:
                logged += 1
                answers = json.dumps([row.get("resps"), row.get("filtered_resps")])
                errored += INFERENCE_ERROR_MARKER in answers
    doc = copy.deepcopy(template)
    # Every metric lm-eval reported must be recomputed from the full set; a
    # value left from the template would be one shard's score in a "merged"
    # result. So each entry is replaced wholesale, and an entry the merge
    # cannot recompute (e.g. a group aggregate the plan does not describe)
    # refuses the merge instead of keeping the shard value.
    recomputed = merged["results"]
    for section in ("results", "groups"):
        for name, entry in list(doc.get(section, {}).items()):
            reported = {key for key in entry if "," in key}
            if not reported:
                continue
            metrics = recomputed.get(name)
            missing = sorted(reported - set(metrics or {}))
            if missing:
                raise ShardError(
                    f"{section}.{name}: lm-eval reported {missing} but the merge "
                    "cannot recompute them from the shards' samples and the plan; "
                    "refusing to keep one shard's values"
                )
            doc[section][name] = {**metrics, "alias": entry.get("alias", name)}
    # lm-eval's convention: original is the dataset, effective what was graded
    # (a doc_ids task: its configured set, not the whole dataset).
    doc["n-samples"] = {
        leaf: {"original": int(plan["leaves"][leaf]["n_docs"]), "effective": n}
        for leaf, n in merged["n_samples"].items()
    }
    first = runs[0][1]
    doc["eval_shard_merge"] = {
        "task_name": task,
        "shard_count": count,
        "plan_sha": first["plan_sha"],
        "expected_doc_counts": {
            leaf: len(leaf_doc_ids(info)) for leaf, info in plan["leaves"].items()
        },
        "provenance": first["provenance"],
        "doc_hashes": merged["doc_hashes"],
        "sources": sources,
        "inference_error_samples": errored,
        "samples_logged": logged,
    }
    return doc


def _graded(model: str, task_name: str, doc: Mapping[str, Any], results_path: Path):
    """Grade the merged results exactly as a full run would be graded.

    Returns ``(report, workflow, schema)``: the merge report, and the evals
    workflow and report schema run.py would hold at this point (blocks tagged
    with their delivery stages, acceptance export in the metadata), for
    :func:`write_run_report`.
    """
    from report_module.acceptance_criteria import (
        acceptance_criteria_check,
        build_acceptance_export,
        stage_results,
    )
    from test_module.llm_tests.llm_eval_tests import (
        _envelope,
        blocks_for_task,
        load_eval_results,
        mark_incomplete,
    )
    from workflow_module.blocks_sink import BlockAccumulator
    from workflow_module.workflows import EvalsWorkflow

    task = eval_task(model, task_name)
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
    blocks = blocks_for_task(ctx, task, results, counts)
    if not blocks:
        # load_eval_results logs and skips a results document it cannot read
        # (schema mismatch, empty configs, mixed dataset paths), and
        # blocks_for_task then returns no block. Grading an empty block list
        # would read PASS with nothing measured, so fail closed instead.
        raise ShardError(
            f"INCOMPLETE: the merged results for {task_name} could not be loaded "
            "or scored (no eval block was produced); refusing to grade an empty "
            "report"
        )
    merge_info = doc["eval_shard_merge"]
    if merge_info.get("inference_error_samples"):
        mark_incomplete(
            blocks, merge_info["inference_error_samples"], merge_info["samples_logged"]
        )
    for block in blocks:
        block.data["eval_shard_merge"] = {
            "shard_count": doc["eval_shard_merge"]["shard_count"],
            "plan_sha": doc["eval_shard_merge"]["plan_sha"],
        }
    # The same accumulator, envelope and stage tagging as run.py's evals
    # workflow, so the schema graded here is the one its report is written from.
    accumulator = BlockAccumulator()
    accumulator.accept(blocks, envelope=_envelope(ctx))
    workflow = EvalsWorkflow(ctx, accumulator=accumulator)
    schema = workflow.format_results()
    known_issues = provenance.get("known_issues") or None
    model_status = provenance.get("model_status")
    accepted, blockers, categories = acceptance_criteria_check(
        schema, known_issues=known_issues, model_status=model_status
    )
    stages = stage_results(
        schema, blockers, known_issues=known_issues, model_status=model_status
    )
    export = build_acceptance_export(
        accepted, blockers, categories, model_status, stages
    )
    schema.metadata.update(export)
    report = {
        "model": model,
        "task_name": task_name,
        "eval_shard_merge": doc["eval_shard_merge"],
        "evals": [
            {"title": b.title, "data": b.data, "targets": b.targets} for b in blocks
        ],
        **export,
    }
    return report, workflow, schema


def grade(
    model: str, task_name: str, doc: Mapping[str, Any], results_path: Path
) -> dict:
    """Grade the merged results exactly as a full run would be graded."""
    return _graded(model, task_name, doc, results_path)[0]


# Run config that legitimately differs between the shards of one merge.
PER_SHARD_RUNTIME_CONFIG_KEYS = frozenset({"eval_shard", "run_id"})


def _spec_identity_problems(spec: Mapping[str, Any], provenance: Mapping[str, Any]):
    """Where a runtime model spec disagrees with the shards' recorded provenance."""
    env_vars = (spec.get("device_model_spec") or {}).get("env_vars") or {}
    pairs = (
        ("hf_model_repo", spec.get("hf_model_repo")),
        ("docker_image", spec.get("docker_image")),
        ("device", spec.get("device_type")),
        ("model_status", spec.get("status")),
        ("package_manifest_sha256", env_vars.get("QUETZAL_BUNDLE_MANIFEST_SHA256")),
    )
    return [
        f"{key} {mine!r} != shards' {provenance.get(key)!r}"
        for key, mine in pairs
        if mine != provenance.get(key)
    ]


def find_shard_runtime_model_spec(
    dirs: Sequence[Path], model: str, merge_info: Mapping[str, Any]
) -> Optional[Path]:
    """The runtime model spec the merged run's identity comes from, or None.

    run.py writes one ``runtime_model_spec_*.json`` per run and stamps the
    run's ``run_id`` into the shard record next to that run's results, so each
    spec is bound to the shard attempt it belongs to. Fail closed: when any
    spec is present, every merged shard must have exactly one spec with its
    run id and shard id, every spec must belong to a merged shard (a spec from
    an aborted or foreign attempt is refused, never picked), all specs must
    agree except on per-shard keys, and their identity (model, image, device,
    status, package) must match the merged provenance. None only when the
    shard dirs hold no spec at all. Returns the lowest shard's spec.
    """
    specs = []
    for root in dirs:
        for path in sorted(Path(root).rglob(RUNTIME_MODEL_SPEC_GLOB)):
            data = json.loads(path.read_text())
            spec = data.get("runtime_model_spec") if isinstance(data, dict) else None
            config = data.get("runtime_config") if isinstance(data, dict) else None
            if not isinstance(spec, dict) or not isinstance(config, dict):
                raise ShardError(f"{path}: not a run.py runtime model spec")
            specs.append((path, spec, config))
    if not specs:
        return None
    by_run: Dict[str, List[Tuple[Path, dict, dict]]] = {}
    for entry in specs:
        by_run.setdefault(entry[2].get("run_id"), []).append(entry)
    sources = merge_info["sources"]
    merged_runs = {source.get("run_id") for source in sources}
    problems = []
    for run_id, entries in sorted(by_run.items(), key=lambda kv: str(kv[0])):
        if run_id is None or run_id not in merged_runs:
            problems.extend(
                f"{path}: run {run_id!r} contributed no merged shard"
                for path, _, _ in entries
            )
    bound = []
    for source in sorted(sources, key=lambda s: parse_eval_shard(s["shard"])[0]):
        run_id = source.get("run_id")
        entries = by_run.get(run_id, []) if run_id else []
        if len(entries) != 1:
            problems.append(
                f"shard {source['shard']} (run {run_id!r}): expected one runtime "
                f"model spec, found {len(entries)}"
            )
            continue
        path, spec, config = entries[0]
        if config.get("eval_shard") != source["shard"]:
            problems.append(
                f"{path}: eval_shard {config.get('eval_shard')!r} != {source['shard']!r}"
            )
        if spec.get("hf_model_repo") != model:
            problems.append(
                f"{path}: spec is for {spec.get('hf_model_repo')}, not {model}"
            )
        problems.extend(
            f"{path}: {problem}"
            for problem in _spec_identity_problems(spec, merge_info["provenance"])
        )
        bound.append((path, spec, config))
    if not problems and bound:
        first_path, first_spec, first_config = bound[0]
        for path, spec, config in bound[1:]:
            if spec != first_spec:
                problems.append(f"{path}: runtime model spec differs from {first_path}")
            keys = (set(config) | set(first_config)) - PER_SHARD_RUNTIME_CONFIG_KEYS
            differing = sorted(k for k in keys if config.get(k) != first_config.get(k))
            if differing:
                problems.append(
                    f"{path}: runtime config differs from {first_path} on {differing}"
                )
    if problems:
        raise ShardError(
            "runtime model specs do not bind to the merged shards:\n  - "
            + "\n  - ".join(problems)
        )
    return bound[0][0]


class _SerializedModelSpec:
    """A shard's serialized ``runtime_model_spec``, re-emitted unchanged.

    ``RuntimeConfig.to_json`` only calls ``get_serialized_dict()``; passing the
    shard's dict (not a ModelSpec rebuilt from it) keeps the merged spec
    byte-for-byte the shards' model spec.
    """

    def __init__(self, serialized: dict) -> None:
        self._serialized = serialized

    def get_serialized_dict(self) -> dict:
        return self._serialized


def write_runtime_model_spec(
    out_dir: Path, shard_spec: Path, task_name: str, merge_info: Mapping[str, Any]
) -> Path:
    """The merged run's runtime model spec, written by ``RuntimeConfig.to_json``.

    The shard's model spec and run config, except ``eval_shard`` (None: the
    merged set is the full task), ``eval_tasks`` (the merged task) and
    ``eval_shard_merge`` (the merge provenance).
    """
    from workflows.runtime_config import RuntimeConfig

    data = json.loads(shard_spec.read_text())
    config = RuntimeConfig.from_dict(data["runtime_config"])
    config.eval_shard = None
    config.eval_tasks = task_name
    config.eval_shard_merge = dict(merge_info)
    spec = data["runtime_model_spec"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return config.to_json(
        _SerializedModelSpec(spec),
        timestamp,
        spec.get("model_id") or "merged",
        str(out_dir),
    )


def write_run_report(
    workflow, schema, out_dir: Path, run_command: str, spec_path: Path
) -> Path:
    """Write ``report_data_<id>.json`` as run.py's evals workflow does.

    Same metadata injection (identity fields from the runtime model spec) and
    the same ``ReportGenerator``; returns the JSON path.
    """
    from report_module.generator import ReportGenerator
    from workflow_module.execution import OrchestratorMetadata

    workflow.orchestrator_metadata = dataclasses.replace(
        workflow.orchestrator_metadata or OrchestratorMetadata(),
        run_command=run_command,
        runtime_model_spec_json=str(spec_path),
    )
    workflow.inject_metadata(schema)
    return ReportGenerator().generate(schema, out_dir).json_path


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
        task = eval_task(args.model, args.task)
        doc = merge(args.shard_dirs, args.task, task.doc_ids)
        shard_spec = find_shard_runtime_model_spec(
            args.shard_dirs, args.model, doc["eval_shard_merge"]
        )
    except ShardError as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / MERGED_RESULTS_NAME
    results_path.write_text(json.dumps(doc, indent=2, default=str))
    try:
        report, workflow, schema = _graded(args.model, args.task, doc, results_path)
    except ShardError as exc:
        print(f"merge refused: {exc}", file=sys.stderr)
        return 2
    (args.out / MERGED_REPORT_NAME).write_text(
        json.dumps(report, indent=2, default=str)
    )
    if shard_spec is None:
        print(
            f"warning: no {RUNTIME_MODEL_SPEC_GLOB} under the shard dirs; not writing "
            "the run.py report layout (report_data / runtime model spec)",
            file=sys.stderr,
        )
    else:
        spec_path = write_runtime_model_spec(
            args.out / RUN_RUNTIME_MODEL_SPEC_DIR,
            shard_spec,
            args.task,
            doc["eval_shard_merge"],
        )
        run_command = " ".join(
            ["python -m report_module.merge_eval_shards", *(argv or sys.argv[1:])]
        )
        write_run_report(
            workflow, schema, args.out / RUN_REPORT_DIR, run_command, spec_path
        )
    print(report["acceptance_summary_markdown"])
    return 0 if report["acceptance_criteria"] else 1


if __name__ == "__main__":
    sys.exit(main())
