#!/usr/bin/env python3
"""Run one arm of the predeclared GPT-OSS S8192 reasoning A/B.

Both arms use the same generated-only endpoint, pinned SWE-bench row, sampling
seed, token envelope, retained-observation budget, and isolated verifier.  The
only model-request difference is whether ``reasoning_effort=high`` is present.
This is causal local evidence, not a Models-CI graded subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL = "openai/gpt-oss-120b"
DATASET = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
INSTANCE_IDS = ["django__django-11299"]
MAX_CONTEXT = 8192
MAX_INPUT = 7168
MAX_OUTPUT = 1024
STEP_LIMIT = 16
OBSERVATION_RETAINED_PAYLOAD_CHARS = 2048
SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
BOUNDED_INSTANCE_TEMPLATE = """<pr_description>
{{task}}
</pr_description>

Fix this bug in /testbed. Work concisely: inspect only likely files and do not
run recursive repository listings. Avoid rereading the same file range unless
new evidence requires it. By turn 8, if the likely source path is known, make
the smallest plausible source edit. Modify source files only (not tests or
configuration). Use at least one bash tool call in each response. When the fix
is ready, create patch.txt with `git diff --` over only modified source files,
inspect it in a separate command, then submit with exactly
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.
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


def completion_kwargs(reasoning_mode: str) -> dict[str, object]:
    result: dict[str, object] = {"seed": SEED}
    if reasoning_mode == "high":
        result["reasoning_effort"] = "high"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reasoning-mode", choices=("default", "high"), required=True)
    parser.add_argument(
        "--agent-workflow",
        choices=("upstream", "bounded"),
        default="upstream",
    )
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()

    if MAX_INPUT + MAX_OUTPUT != MAX_CONTEXT:
        raise RuntimeError("GPT causal A/B no longer exactly binds S8192")
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
    request_kwargs = completion_kwargs(args.reasoning_mode)
    predeclared = {
        "schema": "ttis.gpt120-s8192-causal-swe-arm/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qualification_claim": "local_behavioral_only",
        "quality_status": "ungraded",
        "causal_factor": (
            "reasoning_effort"
            if args.agent_workflow == "upstream"
            else "generic_bounded_agent_workflow"
        ),
        "reasoning_mode": args.reasoning_mode,
        "agent_workflow": args.agent_workflow,
        "selection_policy": "predeclared_ordered_subset",
        "selection_provenance": (
            "fixed before both A/B arms; no gold-patch or model-output selection"
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
            "observation_retained_payload_chars": (OBSERVATION_RETAINED_PAYLOAD_CHARS),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "completion_kwargs": request_kwargs,
            "instance_template_sha256": (
                hashlib.sha256(BOUNDED_INSTANCE_TEMPLATE.encode()).hexdigest()
                if args.agent_workflow == "bounded"
                else None
            ),
        },
        "preregistered_gates": {
            "first_mutation_turn_lte": 8,
            "duplicate_file_range_reads": 0,
            "nonempty_patch_submitted": True,
            "isolated_verifier_required": True,
        },
        "verifier": "swebench.harness.run_evaluation",
        "slurm": {"job_id": args.slurm_job_id, "node": args.node},
        "endpoint_models_response": models,
    }
    write_json(out / "predeclared-contract.json", predeclared)
    print(json.dumps(predeclared, sort_keys=True), flush=True)

    os.environ["MODEL_SPECS_ENV"] = "dev"
    os.environ.setdefault("TTIS_REPO_ROOT", str(REPO_ROOT))
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
        task_name=f"gpt120-s8192-causal-{args.reasoning_mode}",
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
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_input_tokens=MAX_INPUT,
        max_output_tokens=MAX_OUTPUT,
        completion_kwargs=request_kwargs,
        swebench_timeout_sec=30 * 60,
        agent_generation_timeout_sec=6 * 60 * 60,
        shuffle=False,
        random_delay_multiplier=0.0,
        score_existing_predictions=False,
        instance_ids=INSTANCE_IDS,
        mini_agent_kwargs={
            "step_limit": STEP_LIMIT,
            **(
                {"instance_template": BOUNDED_INSTANCE_TEMPLATE}
                if args.agent_workflow == "bounded"
                else {}
            ),
        },
        mini_observation_chars=OBSERVATION_RETAINED_PAYLOAD_CHARS,
        qualification_claim="local_behavioral_only",
        selection_policy="predeclared_ordered_subset",
        instance_selection_provenance=predeclared["selection_provenance"],
        dataset_revision=DATASET_REVISION,
        ordered_instance_ids_sha256=ordered_ids_sha,
        selected_instances_sha256=selected_sha,
        eval_limit_mode="causal-ab",
        tokenizer_name=str(args.tokenizer_root.resolve()),
        venv_python=Path(sys.executable),
    )
    rc = run(run_cfg)
    terminal = dict(predeclared)
    terminal["ended_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    terminal["run_return_code"] = rc
    result = out / "result.json"
    predictions = out / "predictions.jsonl"
    terminal["result_present"] = result.is_file()
    terminal["result_sha256"] = (
        hashlib.sha256(result.read_bytes()).hexdigest() if result.is_file() else None
    )
    terminal["predictions_present"] = predictions.is_file()
    terminal["predictions_sha256"] = (
        hashlib.sha256(predictions.read_bytes()).hexdigest()
        if predictions.is_file()
        else None
    )
    terminal["status"] = (
        "pass" if rc == 0 and result.is_file() and predictions.is_file() else "fail"
    )
    write_json(out / "terminal-receipt.json", terminal)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
