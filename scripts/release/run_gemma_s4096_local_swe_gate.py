#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Run a predeclared, pinned Gemma S4096 local behavioral SWE gate.

This is intentionally not a Models CI graded result.  It exercises the real
mini-swe agent and isolated SWE-bench verifier against a generated-only local
endpoint, while binding the selected instance bytes before the first request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

MODEL = "google/gemma-4-31B-it"
DATASET = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
# First S4096 candidate in a predeclared TTIS nightly list.  This was fixed
# before this run and was not chosen from its gold patch or Gemma output.
INSTANCE_IDS = ["scikit-learn__scikit-learn-14629"]
MAX_CONTEXT = 4096
MAX_INPUT = 3584
MAX_OUTPUT = 512
STEP_LIMIT = 12
OBSERVATION_CHARS = 2048
INSTANCE_TEMPLATE = """<pr_description>
{{task}}
</pr_description>

Fix this bug in /testbed. Work concisely: inspect only likely files, do not run
recursive repository listings, and do not install dependencies. Modify source
files only (not tests or configuration). Use at least one bash tool call in
each response. When the fix is ready, create patch.txt with `git diff --` over
only modified source files, inspect it in a separate command, then submit with
exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.
"""


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_ttis_source_revision(repo_root: Path) -> str:
    """Return the exact tracked TTIS revision used by this local gate."""
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise RuntimeError(f"invalid TTIS Git revision: {revision!r}")
    subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--quiet", "HEAD", "--"],
        check=True,
    )
    return revision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    ttis_source_revision = resolve_ttis_source_revision(repo_root)

    if MAX_INPUT + MAX_OUTPUT != MAX_CONTEXT:
        raise RuntimeError("Gemma local gate no longer exactly binds S4096")
    with urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=30) as response:
        models = json.load(response)
    rows = models.get("data", [])
    if [row.get("id") for row in rows] != [MODEL]:
        raise RuntimeError(f"endpoint model mismatch: {rows!r}")
    if rows[0].get("max_model_len") != MAX_CONTEXT:
        raise RuntimeError(f"endpoint context mismatch: {rows[0]!r}")

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="test", revision=DATASET_REVISION)
    by_id = {
        row["instance_id"]: dict(row)
        for row in dataset
        if row["instance_id"] in INSTANCE_IDS
    }
    selected_rows = [by_id[instance_id] for instance_id in INSTANCE_IDS]
    selected_sha = hashlib.sha256(canonical(selected_rows)).hexdigest()
    ordered_ids_sha = hashlib.sha256(
        json.dumps(INSTANCE_IDS, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    predeclared = {
        "schema": "ttis.gemma-s4096-local-swe-gate/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qualification_claim": "local_behavioral_only",
        "quality_status": "ungraded",
        "selection_policy": "predeclared_ordered_subset",
        "selection_provenance": (
            "fixed TTIS nightly instance; chosen before this model call without "
            "gold-patch ranking or prior Gemma output"
        ),
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "instance_ids": INSTANCE_IDS,
        "ordered_instance_ids_sha256": ordered_ids_sha,
        "selected_instances_sha256": selected_sha,
        "serving": {
            "concurrency": 1,
            "max_context": MAX_CONTEXT,
            "max_input_tokens": MAX_INPUT,
            "max_output_tokens": MAX_OUTPUT,
        },
        "agent": {
            "backend": "mini-swe-agent",
            "step_limit": STEP_LIMIT,
            "observation_char_limit": OBSERVATION_CHARS,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "verifier": "swebench.harness.run_evaluation",
        "slurm": {"job_id": args.slurm_job_id, "node": args.node},
        "source": {"ttis_revision": ttis_source_revision},
        "endpoint_models_response": models,
    }
    write_json(out / "predeclared-contract.json", predeclared)
    print(json.dumps(predeclared, sort_keys=True), flush=True)

    os.environ["MODEL_SPECS_ENV"] = "dev"
    os.environ.setdefault("TTIS_REPO_ROOT", str(repo_root))
    from llm_module.agentic.swebench import SWEbenchRunConfig, run
    from reference_config.evals.eval_config import EVAL_CONFIGS

    matching_configs = [
        config for config in EVAL_CONFIGS.values() if config.hf_model_repo == MODEL
    ]
    if len(matching_configs) != 1:
        raise RuntimeError(
            f"expected one eval config for canonical model {MODEL!r}, "
            f"found {len(matching_configs)}"
        )
    task = next(
        task
        for task in matching_configs[0].tasks
        if task.task_name == "swe_bench_verified"
    )
    cfg = task.swebench_eval_config
    run_cfg = SWEbenchRunConfig(
        task_name="gemma-s4096-local-swe-gate",
        dataset_name=DATASET,
        dataset_split="test",
        sweagent_subset=cfg.sweagent_subset,
        agent_backend="mini-swe-agent",
        model_name=f"openai/{MODEL}",
        api_base=args.endpoint.rstrip("/") + "/v1",
        output_dir=out,
        sweagent_config=cfg.sweagent_config,
        mini_config=cfg.mini_config,
        mini_model_class=cfg.mini_model_class,
        mini_environment_class="docker",
        n_concurrent_trials=1,
        max_workers=1,
        n_tasks=None,
        temperature=0.0,
        top_p=1.0,
        max_input_tokens=MAX_INPUT,
        max_output_tokens=MAX_OUTPUT,
        completion_kwargs={
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
        swebench_timeout_sec=1800,
        agent_generation_timeout_sec=3600,
        shuffle=False,
        random_delay_multiplier=0.0,
        score_existing_predictions=False,
        instance_ids=INSTANCE_IDS,
        mini_agent_kwargs={
            "step_limit": STEP_LIMIT,
            "instance_template": INSTANCE_TEMPLATE,
        },
        mini_observation_chars=OBSERVATION_CHARS,
        qualification_claim="local_behavioral_only",
        selection_policy="predeclared_ordered_subset",
        instance_selection_provenance=predeclared["selection_provenance"],
        dataset_revision=DATASET_REVISION,
        ordered_instance_ids_sha256=ordered_ids_sha,
        selected_instances_sha256=selected_sha,
        eval_limit_mode="local_behavioral",
        tokenizer_name=str(args.tokenizer_root.resolve()),
        venv_python=Path(sys.executable),
    )
    rc = run(run_cfg)
    terminal = dict(predeclared)
    terminal["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    terminal["run_return_code"] = rc
    result = out / "result.json"
    terminal["result_present"] = result.is_file()
    terminal["result_sha256"] = (
        hashlib.sha256(result.read_bytes()).hexdigest() if result.is_file() else None
    )
    terminal["status"] = "pass" if rc == 0 and result.is_file() else "fail"
    write_json(out / "terminal-receipt.json", terminal)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
