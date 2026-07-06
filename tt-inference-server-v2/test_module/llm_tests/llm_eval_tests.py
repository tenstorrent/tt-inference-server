# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Standard LLM evals — run lm-eval / lmms-eval tasks and emit Blocks."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

from llm_module import HttpServerController
from llm_module.eval_command import build_eval_command
from llm_module.eval_configs import get_llm_eval_tasks
from report_module.schema import Block
from workflow_module import accept_blocks
from workflows.utils import run_command

from .._test_common import ReportCheckTypes, TestStatus, block_id
from ..context import MediaContext

logger = logging.getLogger(__name__)

# Fallback health-wait budget when the model spec doesn't set one. The per-model
# value comes from DeviceModelSpec.tensor_cache_timeout (first-compile/warmup for
# large forge LLMs can exceed 1200s); bump it per model in the model spec.
_DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 3600.0


def _device_label(ctx: MediaContext) -> str:
    return ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)


# --- reading lm-eval's result JSON off disk (it runs as a subprocess) --------


def discover_eval_results(output_path, model_spec) -> List[str]:
    """Find lm-eval / lmms-eval result JSON for this model under output_path.

    lm-eval (text) writes ``results_*.json``; lmms-eval (vision/audio) writes
    ``*_results.json``. Both land under ``eval_<model_id>/<hf_repo__>/`` where
    ``hf_repo__`` is the repo with ``/`` replaced by ``__`` (mirrors v1's
    per-model-type globs in run_reports.py).
    """
    repo = model_spec.hf_model_repo.replace("/", "__")
    base = f"eval_{model_spec.model_id}/{repo}"
    patterns = [
        f"{output_path}/{base}/results_*.json",
        f"{output_path}/{base}/*_results.json",
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return sorted(set(files))


def _extract_json(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    configs = data.get("configs", {})

    first_key = list(results.keys())[0]

    first_results = results[first_key]
    extracted_metrics = {
        k: v
        for k, v in first_results.items()
        if "alias" not in k and "_stderr" not in k
    }
    extracted = [{first_key: extracted_metrics}]

    config = configs.get(first_key, {})
    task_name = config.get("task", first_key)

    dataset_path = list(configs.values())[0]["dataset_path"]
    for config in configs.values():
        assert dataset_path == config.get("dataset_path")
    assert task_name == first_key, f"Task name mismatch: {task_name} != {first_key}"

    return extracted, {"task_name": task_name, "dataset_path": dataset_path}


def merge_eval_results(files) -> dict:
    """Merge per-task lm-eval result files into one {task_name: metrics} dict."""
    files = sorted(files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    results: dict = {}
    for json_file in files:
        res, _meta = _extract_json(Path(json_file))
        for task_dict in res:
            for specific_task_name, metrics in task_dict.items():
                results.setdefault(specific_task_name, metrics)
    return results


# --- scoring one task's results into Block(kind="evals") ---------------------


def _target_keys(task, results: dict) -> List[str]:
    if task.task_name in results:
        return [task.task_name]
    prefix = f"{task.task_name}_"
    return sorted(k for k in results if k.startswith(prefix))


def _score_one(
    task, results: dict, t_key: str
) -> Tuple[float, Union[float, str], Union[float, str], ReportCheckTypes]:
    """Compute (score, ratio_to_published, ratio_to_reference, accuracy_check)
    for one task/subtask. Real copy of the v1 evals_release scoring."""
    # Shallow-copy so kwargs["task_name"] = t_key doesn't mutate the shared
    # config dict for subsequent tasks in this process.
    kwargs = dict(task.score.score_func_kwargs)
    kwargs["task_name"] = t_key
    configured_keys = kwargs.get("result_keys", [])
    actual_data = results.get(t_key, {})
    key_found = any(k in actual_data for k in configured_keys)
    if not key_found:
        valid_candidates = [
            k
            for k, v in actual_data.items()
            if isinstance(v, (int, float)) and "stderr" not in k and "alias" not in k
        ]
        if valid_candidates:
            logger.info(
                "  Metric mismatch for %s. Auto-detected replacement: %s",
                t_key,
                valid_candidates[0],
            )
            kwargs["result_keys"] = [valid_candidates[0]]
    try:
        score = task.score.score_func(results, task_name=t_key, kwargs=kwargs)
    except Exception as e:
        logger.warning("  Could not calculate score for %s: %s", t_key, e)
        # WER=100 is worst-case; score=0.0 would invert to 100 and wrongly pass.
        score = 100.0 if kwargs.get("unit") == "WER" else 0.0
    if kwargs.get("unit") == "WER":
        score = 100 - score

    published = task.score.published_score
    reference = task.score.gpu_reference_score
    tolerance = task.score.tolerance

    if published:
        assert published > 0, "Published score is not > 0"
        ratio_to_published: Union[float, str] = score / published
    else:
        ratio_to_published = "N/A"

    if reference:
        assert reference > 0, "Reference score is not > 0"
        ratio_to_reference: Union[float, str] = score / reference
        accuracy_check = ReportCheckTypes.from_result(
            ratio_to_reference >= (1.0 - tolerance)
        )
    else:
        ratio_to_reference = "N/A"
        if published:
            accuracy_check = ReportCheckTypes.from_result(
                ratio_to_published >= (1.0 - tolerance)
            )
        else:
            accuracy_check = ReportCheckTypes.NA

    return score, ratio_to_published, ratio_to_reference, accuracy_check


def blocks_for_task(ctx: MediaContext, task, results: dict) -> List[Block]:
    """Score ``task`` against ``results`` and build one Block per task/subtask.

    A task that ran but has no score defined is not gradable -> one NA Bloc.
    A task with a score but no matching results still returns ``[]`` so the
    caller can surface a FAIL block for a task that ran but scored nothing.
    """
    if not task.score:
        reason = "no eval score defined"
        logger.info("%s ran but is not gradable: %s.", task.task_name, reason)
        return [_status_block(ctx, task, TestStatus.NA, reason)]

    blocks: List[Block] = []
    for t_key in _target_keys(task, results):
        score, ratio_pub, ratio_ref, accuracy_check = _score_one(task, results, t_key)
        blocks.append(
            Block(
                kind="evals",
                task_type="llm",
                title=f"LLM Eval — {t_key}",
                id=block_id(ctx) or None,
                targets={
                    "task_name": t_key,
                    "tolerance": task.score.tolerance,
                    "published_score": task.score.published_score,
                    "published_score_ref": task.score.published_score_ref,
                },
                data={
                    "task_name": t_key,
                    "tolerance": task.score.tolerance,
                    "published_score": task.score.published_score,
                    "published_score_ref": task.score.published_score_ref,
                    "gpu_reference_score": task.score.gpu_reference_score,
                    "score": score,
                    "ratio_to_published": ratio_pub,
                    "ratio_to_reference": ratio_ref,
                    "accuracy_check": accuracy_check,
                },
            )
        )
    return blocks


def _fail_block(ctx: MediaContext, task, error: str) -> Block:
    score = getattr(task, "score", None)
    return Block(
        kind="evals",
        task_type="llm",
        title=f"LLM Eval — {task.task_name}",
        id=block_id(ctx) or None,
        targets={"task_name": task.task_name},
        data={
            "task_name": task.task_name,
            "tolerance": getattr(score, "tolerance", None),
            "published_score": getattr(score, "published_score", None),
            "published_score_ref": getattr(score, "published_score_ref", None),
            "score": None,
            "accuracy_check": ReportCheckTypes.FAIL,
            "error": error,
        },
    )


def _status_block(ctx: MediaContext, task, status: TestStatus, reason: str) -> Block:
    """Build a non-graded evals Block carrying an explicit ``status``.

    Keeps a task that was intentionally not run (SKIP) or ran but couldn't be
    graded (NA) *visible* in the report instead of silently vanishing. The
    explicit ``status`` short-circuits acceptance grading, so the block is
    non-blocking.
    """
    score = getattr(task, "score", None)
    return Block(
        kind="evals",
        task_type="llm",
        title=f"LLM Eval — {task.task_name}",
        id=block_id(ctx) or None,
        targets={"task_name": task.task_name},
        data={
            "task_name": task.task_name,
            "status": status.value,
            "skipped": status is TestStatus.SKIP,
            "reason": reason,
            "tolerance": getattr(score, "tolerance", None),
            "published_score": getattr(score, "published_score", None),
            "score": None,
        },
    )


# --- running one task --------------------------------------------------------


def _run_eval_task(ctx: MediaContext, task, auth_token: str) -> int:
    cmd = build_eval_command(
        task,
        ctx.model_spec,
        _device_label(ctx),
        ctx.output_path,
        ctx.server_port,
        runtime_config=ctx.runtime_config,
        deploy_url=ctx.server_host,
    )
    env = dict(os.environ)
    if auth_token:
        # lm-eval local-completions reads the bearer token from OPENAI_API_KEY.
        env["OPENAI_API_KEY"] = auth_token
    logger.info("Running eval task=%s", task.task_name)
    return run_command(command=cmd, logger=logger, env=env)


def run_llm_eval(ctx: MediaContext, *, auth_token: str = "") -> List[Block]:
    """Run standard evals for ``ctx`` and return the emitted Blocks.

    Returns ``[]`` when the model has no standard eval tasks (e.g. agentic-only
    models) so the caller can no-op. On server-health or per-task failure it
    emits FAIL Blocks rather than silently dropping the task, so a release run
    surfaces the failure.
    """
    tasks = get_llm_eval_tasks(ctx.model_spec, ctx.runtime_config)
    if not tasks:
        logger.info(
            "No standard eval tasks for model=%s; nothing to run.",
            ctx.model_spec.model_name,
        )
        return []

    server = HttpServerController(
        base_url=ctx.server_host,
        service_port=ctx.server_port,
        auth_token=auth_token,
    )
    health_timeout = (
        getattr(
            getattr(ctx.model_spec, "device_model_spec", None),
            "tensor_cache_timeout",
            None,
        )
        or _DEFAULT_WAIT_HEALTHY_TIMEOUT_S
    )
    if not server.wait_for_healthy(timeout=health_timeout):
        logger.error("⛔ inference server not healthy; aborting evals.")
        blocks = [_fail_block(ctx, t, "inference server not healthy") for t in tasks]
        _accept(ctx, blocks)
        return blocks

    # Trace capture is skipped for evals (it's a perf warm-up; eval correctness
    # doesn't depend on it). lm-eval carries its own per-request timeout.
    device_max_context = getattr(
        getattr(ctx.model_spec, "device_model_spec", None), "max_context", None
    )
    ran_tasks = []
    rc_by_task = {}
    skipped_blocks: List[Block] = []
    for task in tasks:
        min_ctx = getattr(task, "min_context_required", None)
        if min_ctx and device_max_context and device_max_context < min_ctx:
            reason = (
                f"requires max_context >= {min_ctx}, device provides "
                f"{device_max_context}"
            )
            logger.warning("⏭  Skipping %s: %s.", task.task_name, reason)
            skipped_blocks.append(_status_block(ctx, task, TestStatus.SKIP, reason))
            continue
        health = server.get_health()
        if getattr(health, "status_code", 200) != 200:
            logger.error(
                "⛔ server unhealthy mid-eval (status %s); aborting.",
                getattr(health, "status_code", "?"),
            )
            rc_by_task[task.task_name] = 1
            ran_tasks.append(task)
            break
        rc_by_task[task.task_name] = _run_eval_task(ctx, task, auth_token)
        ran_tasks.append(task)

    results = merge_eval_results(discover_eval_results(ctx.output_path, ctx.model_spec))
    blocks: List[Block] = list(skipped_blocks)
    for task in ran_tasks:
        task_blocks = blocks_for_task(ctx, task, results)
        if task_blocks:
            blocks.extend(task_blocks)
        else:
            # Ran but scored nothing (command failed or results unparseable) —
            # v1's report path silently drops these; we surface a FAIL block.
            rc = rc_by_task.get(task.task_name)
            blocks.append(_fail_block(ctx, task, f"no eval results parsed (rc={rc})"))

    _accept(ctx, blocks)
    return blocks


def _accept(ctx: MediaContext, blocks: List[Block]) -> None:
    if not blocks:
        return
    accept_blocks(
        blocks,
        envelope={
            "model_name": ctx.model_spec.hf_model_repo,
            "device": _device_label(ctx),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )


__all__ = ["run_llm_eval"]
