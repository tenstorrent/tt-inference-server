# SPDX-License-Identifier: Apache-2.0
"""Resolve the frozen goal12 SWE population without a mutable registry lookup."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "reference_config/evals"


def selected(config):
    return config.dataset == "swebench-verified" and config.model_name in (
        "Qwen/Qwen3.8-27B", "openai/Qwen/Qwen3.8-27B")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_dataset(config, *, root=ROOT):
    path = root / "qwen38_goal12_swe_registry.json"
    manifest = json.loads((root / "qwen38_goal12_samples.json").read_text())["evals"]["swe_bench_verified"]
    entries = json.loads(path.read_text())
    if len(entries) != 1 or entries[0]["name"] != "swebench-verified" or entries[0]["version"] != "1.0":
        raise ValueError("Frozen SWE registry identity mismatch")
    tasks = {t["name"]: t for t in entries[0]["tasks"]}
    names = sorted(tasks)
    if (len(names) != 500 or len(entries[0]["tasks"]) != 500
            or digest(names) != manifest["population_names_sha256"]
            or digest(tasks) != manifest["population_sources_sha256"]):
        raise ValueError("Frozen SWE population/source hash mismatch")
    if config.dataset_ref:
        raise ValueError("SWE legacy registry uses version, never a Hub ref")
    if not set(config.task_names).issubset(tasks):
        raise ValueError("Requested SWE task missing from frozen population")
    receipt = {"dataset": "swebench-verified", "version": "1.0", "population": len(names),
               "population_names_sha256": digest(names), "population_sources_sha256": digest(tasks),
               "source_commits": sorted({t["git_commit_id"] for t in tasks.values()}),
               "selected_task_names": config.task_names or names, "n_tasks_limit": config.n_tasks,
               "registry_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (config.jobs_dir / f"{config.task_name}_dataset_identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"version": "1.0", "registry_path": str(path)}
