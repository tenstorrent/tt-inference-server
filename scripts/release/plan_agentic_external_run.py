#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Build a fail-closed launch contract for an external agentic endpoint.

This does not contact or start a server.  The caller must supply the capacities
admitted by the serving artifact; values smaller than the configured workload
are rejected before a long-running SWE/Terminal-Bench harness is launched.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_module.drivers.agentic import (  # noqa: E402
    resolve_instance_ids,
    resolve_n_tasks,
)
from reference_config.evals.eval_config import EVAL_CONFIGS  # noqa: E402
from workflows.model_spec import get_runtime_model_spec  # noqa: E402
from workflows.runtime_config import RuntimeConfig  # noqa: E402
from workflows.workflow_types import WorkflowVenvType  # noqa: E402


class ContractError(ValueError):
    """The endpoint admission evidence cannot satisfy the configured eval."""


@dataclass(frozen=True)
class AgenticLaunchContract:
    model: str
    hf_model_repo: str
    device: str
    task: str
    limit_samples_mode: str
    concurrency: int
    instance_ids: list[str]
    n_tasks: int | None
    max_input_tokens: int
    max_output_tokens: int
    required_context_tokens: int
    catalog_max_context_tokens: int
    admitted_max_input_tokens: int
    admitted_max_context_tokens: int
    server_url: str
    service_port: int


def build_contract(
    *,
    model: str,
    device: str,
    task_name: str,
    limit_samples_mode: str,
    admitted_max_input_tokens: int,
    admitted_max_context_tokens: int,
    server_url: str,
    service_port: int,
) -> tuple[AgenticLaunchContract, object]:
    model_spec, _, _ = get_runtime_model_spec(model=model, device=device)
    eval_config = EVAL_CONFIGS.get(model_spec.model_name)
    if eval_config is None:
        raise ContractError(f"no eval config for {model_spec.model_name!r}")
    matches = [task for task in eval_config.tasks if task.task_name == task_name]
    if len(matches) != 1:
        raise ContractError(
            f"expected exactly one task {task_name!r}, found {len(matches)}"
        )
    task = matches[0]
    if task.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC:
        raise ContractError(f"task {task_name!r} is not an agentic eval")
    cfg = task.swebench_eval_config or task.agentic_eval_config
    if cfg is None:
        raise ContractError(f"task {task_name!r} has no agentic configuration")

    max_output = cfg.max_output_tokens
    if max_output is None or max_output <= 0:
        raise ContractError(f"task {task_name!r} needs a finite positive output cap")
    if cfg.max_input_tokens <= 0:
        raise ContractError(f"task {task_name!r} needs a positive input cap")
    required_context = cfg.max_input_tokens + max_output
    catalog_context = int(model_spec.device_model_spec.max_context)
    if required_context > catalog_context:
        raise ContractError(
            f"task requires {required_context} tokens but catalog declares {catalog_context}"
        )
    if admitted_max_input_tokens < cfg.max_input_tokens:
        raise ContractError(
            f"artifact admits {admitted_max_input_tokens} input tokens; "
            f"task requires {cfg.max_input_tokens}"
        )
    if admitted_max_context_tokens < required_context:
        raise ContractError(
            f"artifact admits {admitted_max_context_tokens} total tokens; "
            f"task requires {required_context}"
        )
    if cfg.n_concurrent_trials <= 0:
        raise ContractError("agentic concurrency must be positive")
    if cfg.n_concurrent_trials > model_spec.device_model_spec.max_concurrency:
        raise ContractError(
            f"task concurrency {cfg.n_concurrent_trials} exceeds catalog maximum "
            f"{model_spec.device_model_spec.max_concurrency}"
        )

    runtime = SimpleNamespace(limit_samples_mode=limit_samples_mode)
    instance_ids = resolve_instance_ids(task, runtime)
    n_tasks = resolve_n_tasks(task, runtime)
    if task.swebench_eval_config is not None and not instance_ids and n_tasks is None:
        raise ContractError(
            f"mode {limit_samples_mode!r} selects neither fixed instances nor a task cap"
        )

    contract = AgenticLaunchContract(
        model=model_spec.model_name,
        hf_model_repo=model_spec.hf_model_repo,
        device=device,
        task=task_name,
        limit_samples_mode=limit_samples_mode,
        concurrency=cfg.n_concurrent_trials,
        instance_ids=instance_ids,
        n_tasks=n_tasks,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=max_output,
        required_context_tokens=required_context,
        catalog_max_context_tokens=catalog_context,
        admitted_max_input_tokens=admitted_max_input_tokens,
        admitted_max_context_tokens=admitted_max_context_tokens,
        server_url=server_url.rstrip("/"),
        service_port=service_port,
    )
    return contract, model_spec


def write_plan(
    contract: AgenticLaunchContract, model_spec: object, output_dir: Path
) -> tuple[Path, Path, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = RuntimeConfig(
        model=contract.model,
        workflow="agentic",
        device=contract.device,
        service_port=str(contract.service_port),
        server_url=contract.server_url,
        limit_samples_mode=contract.limit_samples_mode,
    )
    runtime_path = runtime.to_json(
        model_spec,
        "external-agentic",
        model_spec.model_id,
        output_dir,
    )
    command = [
        sys.executable,
        str(_REPO_ROOT / "launchers" / "run_agentic.py"),
        "--model",
        contract.model,
        "--workflow",
        "agentic",
        "--device",
        contract.device,
        "--server-url",
        contract.server_url,
        "--service-port",
        str(contract.service_port),
        "--runtime-model-spec-json",
        str(runtime_path),
        "--output-dir",
        str(output_dir / "results"),
    ]
    plan_path = output_dir / "agentic_launch_contract.json"
    plan_path.write_text(
        json.dumps(
            {"contract": asdict(contract), "argv": command}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path, runtime_path, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit-samples-mode", default="ci-nightly")
    parser.add_argument("--admitted-max-input-tokens", type=int, required=True)
    parser.add_argument("--admitted-max-context-tokens", type=int, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--service-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        contract, model_spec = build_contract(
            model=args.model,
            device=args.device,
            task_name=args.task,
            limit_samples_mode=args.limit_samples_mode,
            admitted_max_input_tokens=args.admitted_max_input_tokens,
            admitted_max_context_tokens=args.admitted_max_context_tokens,
            server_url=args.server_url,
            service_port=args.service_port,
        )
        plan_path, runtime_path, command = write_plan(
            contract, model_spec, args.output_dir
        )
    except (ContractError, ValueError) as exc:
        parser.error(str(exc))
    print(f"contract: {plan_path}")
    print(f"runtime: {runtime_path}")
    print(f"command: {shlex.join(command)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
