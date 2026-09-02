#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Run a predeclared, pinned local behavioral SWE gate for any catalogue model.

One SWE contract for every model. Everything model-specific — the canonical
checkpoint, sampler settings, catalogue-owned completion kwargs, the
predeclared instance list — is loaded from the model's catalogue entry in
``reference_config.evals.eval_config``. This file carries zero model branches;
a model whose catalogue entry cannot satisfy the requested envelope fails
closed instead of being special-cased here.

This is intentionally not a Models CI graded result. It exercises the real
mini-swe agent and isolated SWE-bench verifier against a generated-only local
endpoint, while binding the selected instance bytes before the first request.
Prompt templates are never overridden per model: the shared budget-discipline
system prompt installed by ``llm_module.agentic.swebench`` is the only system
prompt, and any per-model prompt override passed through the agent kwargs is
rejected by the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_config.evals.eval_config import SHARED_SWE_STEP_LIMIT  # noqa: E402

DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
OBSERVATION_RETAINED_PAYLOAD_CHARS = 2048
SCHEMA = "ttis.local-swe-gate/v1"


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


def resolve_token_budget(max_context: int, max_output: int) -> tuple[int, int, int]:
    """Return an exact context budget without silently truncating SWE input."""
    if max_context <= 0:
        raise ValueError("max context must be positive")
    if max_output <= 0 or max_output >= max_context:
        raise ValueError("max output must be positive and smaller than max context")
    return max_context, max_context - max_output, max_output


@dataclass(frozen=True)
class GateConfig:
    """The complete per-run SWE gate contract, one shape for every model."""

    model: str
    hf_model_repo: str
    max_context: int
    max_input_tokens: int
    max_output_tokens: int
    step_limit: int
    temperature: float
    top_p: float
    completion_kwargs: dict[str, Any]
    mini_agent_kwargs: dict[str, Any]
    mini_observation_chars: int
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    sweagent_subset: str
    sweagent_config: str
    mini_config: str
    mini_model_class: str
    instance_ids: list[str] = field(default_factory=list)
    selection_policy: Optional[str] = None
    instance_selection_provenance: Optional[str] = None
    catalogue_min_context_required: Optional[int] = None
    qualification_claim: str = "local_behavioral_only"
    # A local gate never grades: no score exists until CS supplies thresholds,
    # so both references stay None and any accuracy reports NA downstream.
    published_score: Optional[float] = None
    gpu_reference_score: Optional[float] = None


def _swe_task(model: str):
    from reference_config.evals.eval_config import EVAL_CONFIGS

    if model not in EVAL_CONFIGS:
        raise ValueError(
            f"unknown catalogue model {model!r}; known: {sorted(EVAL_CONFIGS)}"
        )
    eval_config = EVAL_CONFIGS[model]
    tasks = [
        task for task in eval_config.tasks if task.task_name == "swe_bench_verified"
    ]
    if len(tasks) != 1:
        raise ValueError(
            f"catalogue model {model!r} declares {len(tasks)} swe_bench_verified "
            "rows; exactly one is required"
        )
    return eval_config, tasks[0]


def _declared_instance_ids(cfg) -> list[str]:
    from workflows.workflow_types import EvalLimitMode

    for mode in (EvalLimitMode.CI_NIGHTLY, EvalLimitMode.SMOKE_TEST):
        declared = cfg.instance_ids_map.get(mode)
        if declared:
            return list(declared)
    raise ValueError(
        "catalogue row declares no predeclared SWE instance list; a gate run "
        "cannot select its own instances"
    )


def build_gate_config(
    model: str,
    *,
    max_context: int,
    max_output_tokens: int,
    step_limit: int,
    instance_ids: Optional[list[str]] = None,
) -> GateConfig:
    """Bind one gate run's contract from the model's catalogue entry.

    Every model-specific value comes from the catalogue; the caller owns only
    the request envelope (context/output budget), the step limit, and an
    optional subset of the catalogue's predeclared instance list.
    """
    if not isinstance(step_limit, int) or isinstance(step_limit, bool):
        raise ValueError("step_limit must be an integer")
    if step_limit <= 0:
        raise ValueError("step_limit must be positive")
    ctx, max_input, max_output = resolve_token_budget(max_context, max_output_tokens)

    eval_config, task = _swe_task(model)
    cfg = task.swebench_eval_config
    if cfg is None:
        raise ValueError(f"catalogue model {model!r} has no swebench_eval_config")

    declared = _declared_instance_ids(cfg)
    if instance_ids is None:
        selected = declared
    else:
        unknown = [i for i in instance_ids if i not in declared]
        if unknown:
            raise ValueError(
                "requested instances are not in the catalogue's predeclared "
                f"list: {unknown}"
            )
        # Keep the catalogue's declared order; the CLI may only bound, never
        # reorder or introduce, the predeclared selection.
        selected = [i for i in declared if i in set(instance_ids)]
    if not selected:
        raise ValueError("gate run selected zero instances")

    return GateConfig(
        model=model,
        hf_model_repo=eval_config.hf_model_repo,
        max_context=ctx,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        step_limit=step_limit,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        completion_kwargs=json.loads(json.dumps(cfg.completion_kwargs)),
        mini_agent_kwargs={"step_limit": step_limit},
        mini_observation_chars=(
            cfg.mini_observation_chars
            if cfg.mini_observation_chars is not None
            else OBSERVATION_RETAINED_PAYLOAD_CHARS
        ),
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        dataset_revision=cfg.dataset_revision or DATASET_REVISION,
        sweagent_subset=cfg.sweagent_subset,
        sweagent_config=cfg.sweagent_config,
        mini_config=cfg.mini_config,
        mini_model_class=cfg.mini_model_class,
        instance_ids=selected,
        selection_policy=cfg.selection_policy,
        instance_selection_provenance=cfg.instance_selection_provenance,
        catalogue_min_context_required=task.min_context_required,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument(
        "--step-limit",
        type=int,
        default=SHARED_SWE_STEP_LIMIT,
        help=(
            "agent step budget; defaults to the one shared "
            "SHARED_SWE_STEP_LIMIT used by every catalogue SWE row"
        ),
    )
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--instance-id",
        action="append",
        dest="instance_ids",
        default=None,
        help=(
            "bound the run to a subset of the catalogue's predeclared instance "
            "list (repeatable); catalogue order is preserved"
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("MODEL_SPECS_ENV", "dev")
    os.environ.setdefault("TTIS_REPO_ROOT", str(REPO_ROOT))

    gate = build_gate_config(
        args.model,
        max_context=args.max_context,
        max_output_tokens=args.max_output_tokens,
        step_limit=args.step_limit,
        instance_ids=args.instance_ids,
    )
    ttis_source_revision = resolve_ttis_source_revision(REPO_ROOT)

    with urlopen(args.endpoint.rstrip("/") + "/v1/models", timeout=30) as response:
        models = json.load(response)
    rows = models.get("data", [])
    if [row.get("id") for row in rows] != [gate.hf_model_repo]:
        raise RuntimeError(f"endpoint model mismatch: {rows!r}")
    if rows[0].get("max_model_len") != gate.max_context:
        raise RuntimeError(f"endpoint context mismatch: {rows[0]!r}")

    from datasets import load_dataset

    dataset = load_dataset(
        gate.dataset_name, split=gate.dataset_split, revision=gate.dataset_revision
    )
    by_id = {
        row["instance_id"]: dict(row)
        for row in dataset
        if row["instance_id"] in gate.instance_ids
    }
    missing = [i for i in gate.instance_ids if i not in by_id]
    if missing:
        raise RuntimeError(f"pinned dataset lacks selected instances: {missing}")
    selected_rows = [by_id[instance_id] for instance_id in gate.instance_ids]
    selected_sha = hashlib.sha256(canonical(selected_rows)).hexdigest()
    ordered_ids_sha = hashlib.sha256(
        json.dumps(
            gate.instance_ids, ensure_ascii=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    predeclared = {
        "schema": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "qualification_claim": gate.qualification_claim,
        "quality_status": "ungraded",
        "selection_policy": gate.selection_policy,
        "selection_provenance": gate.instance_selection_provenance,
        "dataset": gate.dataset_name,
        "dataset_revision": gate.dataset_revision,
        "instance_ids": gate.instance_ids,
        "ordered_instance_ids_sha256": ordered_ids_sha,
        "selected_instances_sha256": selected_sha,
        "gate_config": asdict(gate),
        "serving": {
            "concurrency": 1,
            "max_context": gate.max_context,
            "max_input_tokens": gate.max_input_tokens,
            "max_output_tokens": gate.max_output_tokens,
        },
        "agent": {
            "backend": "mini-swe-agent",
            "step_limit": gate.step_limit,
            "observation_retained_payload_chars": gate.mini_observation_chars,
            "temperature": gate.temperature,
            "top_p": gate.top_p,
            "completion_kwargs": gate.completion_kwargs,
        },
        "verifier": "swebench.harness.run_evaluation",
        "slurm": {"job_id": args.slurm_job_id, "node": args.node},
        "source": {"ttis_revision": ttis_source_revision},
        "endpoint_models_response": models,
    }
    write_json(out / "predeclared-contract.json", predeclared)
    print(json.dumps(predeclared, sort_keys=True), flush=True)

    from llm_module.agentic.swebench import SWEbenchRunConfig, run

    run_cfg = SWEbenchRunConfig(
        task_name=f"{gate.model}-s{gate.max_context}-local-swe-gate",
        dataset_name=gate.dataset_name,
        dataset_split=gate.dataset_split,
        sweagent_subset=gate.sweagent_subset,
        agent_backend="mini-swe-agent",
        model_name=f"openai/{gate.hf_model_repo}",
        api_base=args.endpoint.rstrip("/") + "/v1",
        output_dir=out,
        sweagent_config=gate.sweagent_config,
        mini_config=gate.mini_config,
        mini_model_class=gate.mini_model_class,
        mini_environment_class="docker",
        n_concurrent_trials=1,
        max_workers=1,
        n_tasks=None,
        temperature=gate.temperature,
        top_p=gate.top_p,
        max_input_tokens=gate.max_input_tokens,
        max_output_tokens=gate.max_output_tokens,
        completion_kwargs=gate.completion_kwargs,
        swebench_timeout_sec=30 * 60,
        agent_generation_timeout_sec=6 * 60 * 60,
        shuffle=False,
        random_delay_multiplier=0.0,
        score_existing_predictions=False,
        instance_ids=gate.instance_ids,
        mini_agent_kwargs=gate.mini_agent_kwargs,
        mini_observation_chars=gate.mini_observation_chars,
        qualification_claim=gate.qualification_claim,
        selection_policy=gate.selection_policy,
        instance_selection_provenance=gate.instance_selection_provenance,
        dataset_revision=gate.dataset_revision,
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
