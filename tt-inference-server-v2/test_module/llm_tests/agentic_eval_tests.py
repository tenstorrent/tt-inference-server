# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Native v2 agentic eval runner.

Invokes llm_module.agentic harness wrappers (SWE-bench, Terminal-Bench)
in-process. All harness code lives in v2 under llm_module/agentic/ — v2
is self-contained and does not depend on v1 for agentic evals.

Emits Block(kind="evals") so acceptance_criteria._check_evals applies
as-is. No new MediaTaskType, no new dispatch table — AgenticWorkflow
bypasses the dispatcher and calls this runner directly.

Result parsing delegates to llm_module.agentic.report helpers to avoid
duplicating harbor-format logic.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_module.agentic.report import (  # noqa: E402
    _add_harbor_pass_at_metrics,
    _extract_harbor_summary_metrics,
)
from llm_module.agentic.swebench import SWEbenchRunConfig, run as run_swebench  # noqa: E402
from llm_module.agentic.terminal_bench import (  # noqa: E402
    TerminalBenchRunConfig,
    run as run_terminal_bench,
)
from workflows.workflow_types import WorkflowVenvType  # noqa: E402

from report_module.schema import Block
from workflow_module import accept_blocks

from .._test_common import block_id, sweep_envelope
from ..context import MediaContext, require_health

logger = logging.getLogger(__name__)


def _agentic_output_dir(ctx: MediaContext, task) -> Path:
    """Canonical output dir for one agentic task, matching v1's convention.

    v1's build_agentic_eval_command (run_evals.py:659 / 722) writes to:
      <output_path>/eval_<safe_model_id>/agentic/<task_name>/
    """
    safe_model_id = ctx.model_spec.model_id.replace("/", "__")
    return Path(ctx.output_path) / f"eval_{safe_model_id}" / "agentic" / task.task_name


def _build_swebench_config(task, ctx: MediaContext) -> SWEbenchRunConfig:
    """Build SWEbenchRunConfig from a v1 EvalTask.

    Field mapping mirrors v1's build_agentic_eval_command swebench branch
    (evals/run_evals.py:656). Keep in sync if v1 adds new fields.
    """
    cfg = task.swebench_eval_config
    return SWEbenchRunConfig(
        task_name=task.task_name,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        sweagent_subset=cfg.sweagent_subset,
        agent_backend=cfg.agent_backend,
        model_name=cfg.model or f"openai/{ctx.model_spec.hf_model_repo}",
        api_base=f"http://127.0.0.1:{ctx.service_port}/v1",
        output_dir=_agentic_output_dir(ctx, task),
        sweagent_config=cfg.sweagent_config,
        mini_config=cfg.mini_config,
        mini_model_class=cfg.mini_model_class,
        mini_environment_class=cfg.mini_environment_class,
        n_concurrent_trials=cfg.n_concurrent_trials,
        max_workers=cfg.max_workers,
        n_tasks=cfg.n_tasks,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_input_tokens=cfg.max_input_tokens,
        max_output_tokens=cfg.max_output_tokens,
        completion_kwargs=cfg.completion_kwargs,
        swebench_timeout_sec=cfg.swebench_timeout_sec,
        shuffle=cfg.shuffle,
        random_delay_multiplier=cfg.random_delay_multiplier,
        score_existing_predictions=False,
        instance_ids=[],  # CLI-level subsetting deferred (plan non-goals)
    )


def _build_terminal_bench_config(task, ctx: MediaContext) -> TerminalBenchRunConfig:
    """Build TerminalBenchRunConfig from a v1 EvalTask.

    Field mapping mirrors v1's build_agentic_eval_command terminal-bench
    branch (evals/run_evals.py:720).
    """
    cfg = task.agentic_eval_config
    # terminal_bench writes to jobs_dir/<task_name>/result.json, so
    # jobs_dir is one level above the task output dir.
    jobs_dir = _agentic_output_dir(ctx, task).parent
    return TerminalBenchRunConfig(
        task_name=task.task_name,
        dataset=cfg.dataset,
        agent=cfg.agent,
        model_name=cfg.model or f"openai/{ctx.model_spec.hf_model_repo}",
        jobs_dir=jobs_dir,
        api_base=f"http://127.0.0.1:{ctx.service_port}/v1",
        n_concurrent_trials=cfg.n_concurrent_trials,
        n_attempts=cfg.n_attempts,
        environment_type=cfg.environment_type,
        agent_kwargs=cfg.agent_kwargs,
        n_tasks=cfg.n_tasks,
        override_cpus=cfg.override_cpus,
        override_memory_mb=cfg.override_memory_mb,
        timeout_multiplier=cfg.timeout_multiplier,
        agent_timeout_sec=cfg.agent_timeout_sec,
        task_names=cfg.task_names,
        exclude_task_names=cfg.exclude_task_names,
        quiet=cfg.quiet,
        yes=cfg.yes,
    )


def _result_path(task, ctx: MediaContext) -> Path:
    """Canonical result.json path for a task.

    Both harnesses write to the same logical path:
      swebench:       output_dir/result.json  (= _agentic_output_dir/result.json)
      terminal_bench: jobs_dir/task_name/result.json  (= _agentic_output_dir/result.json)
    """
    return _agentic_output_dir(ctx, task) / "result.json"


def _parse_harbor_result(result: dict) -> dict:
    """Extract accuracy metrics from a harbor-format result dict.

    Delegates to evals/agentic/report.py helpers to avoid duplicating
    the harbor-format parsing logic.
    """
    metrics = _extract_harbor_summary_metrics(result)
    _add_harbor_pass_at_metrics(result, metrics)
    return metrics


def _compute_accuracy_check(metrics: dict, task) -> int:
    """Map accuracy to acceptance_criteria scale: PASS=1, MARGINAL=2, FAIL=3.

    _check_evals fails on accuracy_check == 3; anything else passes.
    """
    accuracy = metrics.get("accuracy")
    if accuracy is None or task.score is None:
        return 2
    target = task.score.gpu_reference_score or task.score.published_score
    tol = task.score.tolerance or 0.05
    if accuracy >= target * (1 - tol):
        return 1
    if accuracy >= target * (1 - 2 * tol):
        return 2
    return 3


def _block_from_result(task, ctx: MediaContext, harbor_result: dict) -> Block:
    metrics = _parse_harbor_result(harbor_result)
    return Block(
        kind="evals",
        task_type="llm",
        title=f"Agentic Eval — {task.task_name}",
        id=block_id(ctx),
        targets={
            "task_name": task.task_name,
            "tolerance": task.score.tolerance if task.score else None,
            "published_score": task.score.published_score if task.score else None,
            "published_score_ref": task.score.published_score_ref if task.score else None,
        },
        data={
            "success": True,
            "accuracy_check": _compute_accuracy_check(metrics, task),
            **metrics,
        },
    )


def _select_agentic_tasks(ctx: MediaContext) -> list:
    """Return EVALS_AGENTIC tasks; raise loudly if mixed with non-agentic."""
    tasks = getattr(ctx.all_params, "tasks", []) or []
    agentic = [t for t in tasks if t.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC]
    non_agentic = [t for t in tasks if t.workflow_venv_type != WorkflowVenvType.EVALS_AGENTIC]
    if agentic and non_agentic:
        raise RuntimeError(
            f"v2 agentic runner only supports EVALS_AGENTIC tasks. "
            f"Got non-agentic tasks: {[t.task_name for t in non_agentic]}. "
            f"Either port those to v2, remove {ctx.model_spec.model_name!r} from "
            f"_V2_ROUTED_MODELS, or use --eval-samples to select agentic tasks only."
        )
    return agentic


def run_llm_agentic_eval(ctx: MediaContext) -> List[Block]:
    """Run every EVALS_AGENTIC task for this model; return one Block per task.

    All Blocks are forwarded to the accumulator here so they all land in
    the report. AgenticWorkflow.run_tasks translates the list into a
    single TaskOutcome (accepting multi-block output is the designed
    departure from the v2 tasks[0] convention — see plan).
    """
    require_health(ctx)

    agentic_tasks = _select_agentic_tasks(ctx)
    if not agentic_tasks:
        raise RuntimeError(
            f"No EVALS_AGENTIC tasks configured for {ctx.model_spec.model_name!r}. "
            "Check evals/eval_config.py."
        )

    blocks: List[Block] = []
    for task in agentic_tasks:
        if task.swebench_eval_config is not None:
            logger.info("Running SWE-bench task: %s", task.task_name)
            rc = run_swebench(_build_swebench_config(task, ctx))
        elif task.agentic_eval_config is not None:
            logger.info("Running terminal-bench task: %s", task.task_name)
            rc = run_terminal_bench(_build_terminal_bench_config(task, ctx))
        else:
            raise RuntimeError(
                f"EVALS_AGENTIC task {task.task_name!r} has neither "
                "swebench_eval_config nor agentic_eval_config set."
            )

        if rc != 0:
            logger.error("Task %s exited with rc=%d", task.task_name, rc)
            blocks.append(
                Block(
                    kind="evals",
                    task_type="llm",
                    title=f"Agentic Eval — {task.task_name} (FAILED)",
                    id=block_id(ctx),
                    targets={"task_name": task.task_name},
                    data={"success": False, "accuracy_check": 3, "subprocess_rc": rc},
                )
            )
            continue

        rpath = _result_path(task, ctx)
        if not rpath.exists():
            raise RuntimeError(
                f"Result JSON not found at {rpath} after rc=0 for task {task.task_name!r}. "
                "Check harness output_dir / jobs_dir path mapping."
            )
        with open(rpath) as f:
            harbor_result = json.load(f)

        blocks.append(_block_from_result(task, ctx, harbor_result))
        logger.info(
            "Task %s done: accuracy=%s",
            task.task_name,
            blocks[-1].data.get("accuracy"),
        )

    accept_blocks(blocks, envelope=sweep_envelope(ctx))
    return blocks


__all__ = ["run_llm_agentic_eval"]
