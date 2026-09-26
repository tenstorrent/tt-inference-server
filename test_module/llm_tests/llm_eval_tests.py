# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Standard LLM evals — run lm-eval / lmms-eval tasks and emit Blocks."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

from llm_module import HttpServerController, RemoteOpenAIController
from llm_module.eval_command import build_eval_command
from llm_module.eval_configs import get_llm_eval_tasks
from report_module.schema import Block
from utils.model_naming import slugify_model_id
from workflow_module import accept_blocks
from workflow_module.engine_types import EvalLimitMode
from workflow_module.proc import run_command
from workflow_module.target_pack import get_target_pack

from .._test_common import ReportCheckTypes, TestStatus, block_id, report_model_fields
from ..context import MediaContext

logger = logging.getLogger(__name__)

# Fallback health-wait budget when the model spec doesn't set one. The per-model
# value comes from DeviceModelSpec.tensor_cache_timeout (first-compile/warmup for
# large forge LLMs can exceed 1200s); bump it per model in the model spec.
_DEFAULT_WAIT_HEALTHY_TIMEOUT_S = 3600.0


def _limit_mode(ctx: MediaContext):
    """Resolve the run's EvalLimitMode (from --ci-mode / --limit-samples-mode).

    Returns None for unrestricted full runs, in which case scoring compares
    against the full-dataset gpu_reference_score.
    """
    rc = getattr(ctx, "runtime_config", None)
    mode_str = getattr(rc, "limit_samples_mode", None) if rc is not None else None
    return EvalLimitMode.from_string(mode_str) if mode_str else None


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
    repo = slugify_model_id(model_spec.hf_model_repo)
    base = f"eval_{model_spec.model_id}/{repo}"
    patterns = [
        f"{output_path}/{base}/results_*.json",
        f"{output_path}/{base}/*_results.json",
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return sorted(set(files))


def _extract_json(json_path: Path) -> tuple[str, dict, int | None]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("eval results must be an object")  # noqa: TRY004 -- invalid file data

    results = data.get("results")
    configs = data.get("configs")
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a non-empty object")
    if not isinstance(configs, dict) or not configs:
        raise ValueError("configs must be a non-empty object")
    if any(not isinstance(config, dict) for config in configs.values()):
        raise ValueError("each task config must be an object")

    task_name = next(iter(results))
    metrics = results[task_name]
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics for {task_name} must be an object")  # noqa: TRY004 -- invalid file data
    config = configs.get(task_name, {})
    if config.get("task", task_name) != task_name:
        raise ValueError(f"task name mismatch for {task_name}")
    first_config = next(iter(configs.values()))
    if "dataset_path" not in first_config or any(
        config.get("dataset_path") != first_config["dataset_path"]
        for config in configs.values()
    ):
        raise ValueError("task configs must share a dataset_path")

    metrics = {
        k: v for k, v in metrics.items() if "alias" not in k and "_stderr" not in k
    }
    sample_counts = data.get("n-samples")
    sample_info = (
        sample_counts.get(task_name) if isinstance(sample_counts, dict) else None
    )
    count = sample_info.get("effective") if isinstance(sample_info, dict) else None
    if not isinstance(count, int) or isinstance(count, bool):
        count = None
    return task_name, metrics, count


def load_eval_results(files) -> tuple[dict, dict]:
    """Read each file once; use the newest valid metrics and count for each task.

    Missing sample counts stay absent so scoring can use its ratio fallback.
    Invalid files are skipped without preventing other tasks from running.
    """
    loaded = []
    for json_file in files:
        try:
            path = Path(json_file)
            modified = path.stat().st_mtime
            loaded.append((modified, _extract_json(path)))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping invalid eval results %s: %s", json_file, exc)

    results: dict = {}
    counts: dict = {}
    for _, (task_name, metrics, count) in sorted(
        loaded, key=lambda item: item[0], reverse=True
    ):
        if task_name in results:
            continue
        results[task_name] = metrics
        if count is not None:
            counts[task_name] = count
    return results, counts


# --- scoring one task's results into Block(kind="evals") ---------------------


def _target_keys(task, results: dict) -> List[str]:
    if task.task_name in results:
        return [task.task_name]
    prefix = f"{task.task_name}_"
    return sorted(k for k in results if k.startswith(prefix))


def _score_one(
    task, results: dict, t_key: str, ref: dict, n_total=None
) -> Tuple[float, Union[float, str], Union[float, str], ReportCheckTypes]:
    """Compute (score, ratio_to_published, ratio_to_reference, accuracy_check)
    for one task/subtask. Real copy of the v1 evals_release scoring.

    ``ref`` is the resolved reference dict from
    ``evals.eval_config.resolve_eval_reference`` (full-set or, under a limit
    mode, the matching subset reference). ``n_total`` is the effective sample
    count, used for the sample-count-aware acceptance check on subset
    references."""
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
    reference = ref["reference_score"]
    tolerance = ref["tolerance"]

    if published:
        assert published > 0, "Published score is not > 0"
        ratio_to_published: Union[float, str] = score / published
    else:
        ratio_to_published = "N/A"

    if reference:
        ratio_to_reference: Union[float, str] = score / reference
        # Sample-count-aware for subset references, ratio for full-set.
        accuracy_check = ReportCheckTypes.from_result(
            get_target_pack().accept_eval_score(ref, score, n_total=n_total)
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


def blocks_for_task(
    ctx: MediaContext,
    task,
    results: dict,
    sample_counts: Optional[dict] = None,
    elapsed_seconds: Optional[float] = None,
) -> List[Block]:
    """Score ``task`` against ``results`` and build one Block per task/subtask.

    A task that ran but has no score defined is not gradable -> one NA Block.
    A task with a score but no matching results still returns ``[]`` so the
    caller can surface a FAIL block for a task that ran but scored nothing.
    ``sample_counts`` maps task_name -> effective sample count for the
    sample-count-aware acceptance check on subset references. When the caller
    supplies the task subprocess wall time, the same counts produce the mean
    wall-clock seconds per evaluated sample.
    """
    if not task.score:
        reason = "no eval score defined"
        logger.info("%s ran but is not gradable: %s.", task.task_name, reason)
        return [_status_block(ctx, task, TestStatus.NA, reason)]

    sample_counts = sample_counts or {}

    # Under --ci-mode / --limit-samples-mode, compare the subset score against
    # the matching subset reference (mode_reference_scores) instead of the
    # full-dataset gpu_reference_score.
    ref = get_target_pack().resolve_eval_reference(task.score, _limit_mode(ctx))

    target_keys = _target_keys(task, results)
    total_samples = sum(
        count
        for key in target_keys
        if isinstance((count := sample_counts.get(key)), int)
        and not isinstance(count, bool)
        and count > 0
    )
    mean_seconds_per_task = (
        elapsed_seconds / total_samples
        if isinstance(elapsed_seconds, (int, float))
        and not isinstance(elapsed_seconds, bool)
        and elapsed_seconds >= 0
        and total_samples > 0
        else None
    )

    blocks: List[Block] = []
    for t_key in target_keys:
        score, ratio_pub, ratio_ref, accuracy_check = _score_one(
            task, results, t_key, ref, n_total=sample_counts.get(t_key)
        )
        data = {
            "task_name": t_key,
            "tolerance": ref["tolerance"],
            "published_score": task.score.published_score,
            "published_score_ref": task.score.published_score_ref,
            "gpu_reference_score": ref["reference_score"],
            "gpu_reference_score_ref": ref["reference_ref"],
            "score": score,
            "ratio_to_published": ratio_pub,
            "ratio_to_reference": ratio_ref,
            "accuracy_check": accuracy_check,
            "priority": getattr(task, "priority", "must"),
        }
        if mean_seconds_per_task is not None:
            data["mean_seconds_per_task"] = mean_seconds_per_task
        blocks.append(
            Block(
                kind="evals",
                task_type="llm",
                title=f"LLM Eval — {t_key}",
                id=block_id(ctx) or None,
                targets={
                    "task_name": t_key,
                    "tolerance": ref["tolerance"],
                    "published_score": task.score.published_score,
                    "published_score_ref": task.score.published_score_ref,
                },
                data=data,
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
            "priority": getattr(task, "priority", "must"),
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


def _run_eval_task(
    ctx: MediaContext, task, auth_token: str, *, output_path: Path
) -> int:
    cmd = build_eval_command(
        task,
        ctx.model_spec,
        _device_label(ctx),
        output_path,
        ctx.server_port,
        runtime_config=ctx.runtime_config,
        deploy_url=ctx.server_host,
    )
    env = dict(os.environ)
    if auth_token:
        # lm-eval local-completions reads the bearer token from OPENAI_API_KEY.
        env["OPENAI_API_KEY"] = auth_token
    if getattr(task, "allow_code_execution", False):
        # HF evaluate's code_eval metric refuses to run without this. Scoped to
        # this subprocess only -- never os.environ, the image, or the workflow
        # env -- so it cannot leak to another task in the same run.
        env["HF_ALLOW_CODE_EVAL"] = "1"
        logger.warning(
            "task=%s runs model-generated code on this host (allow_code_execution).",
            task.task_name,
        )
    if getattr(task, "capture_reasoning", False):
        # Opt-in: tell the (patched) lm-eval to keep the model's separate
        # reasoning_content trace in the per-sample logs. Scoped to this
        # subprocess only so it never leaks to another task in the same run.
        env["LM_EVAL_PRESERVE_REASONING"] = "1"
        logger.info(
            "task=%s will preserve reasoning_content in sample logs.", task.task_name
        )
    logger.info("Running eval task=%s", task.task_name)
    return run_command(command=cmd, logger=logger, env=env)


def run_llm_eval(ctx: MediaContext, *, auth_token: str = "") -> List[Block]:
    """Run standard evals for ``ctx`` and return the emitted Blocks.

    Returns ``[]`` when the model has no standard eval tasks (e.g. agentic-only
    models) so the caller can no-op. On server-health or per-task failure it
    emits FAIL Blocks rather than silently dropping the task, so a release run
    surfaces the failure.
    """
    tasks = get_llm_eval_tasks(ctx.model_spec, ctx.runtime_config, device=ctx.device)
    if not tasks:
        logger.info(
            "No standard eval tasks for model=%s; nothing to run.",
            ctx.model_spec.model_name,
        )
        return []

    if ctx.remote_server:
        server = RemoteOpenAIController(
            base_url=ctx.base_url,
            auth_token=auth_token,
        )
    else:
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
    envelope = _envelope(ctx)
    if not server.wait_for_healthy(timeout=health_timeout):
        logger.error("⛔ inference server not healthy; aborting evals.")
        blocks = [_fail_block(ctx, t, "inference server not healthy") for t in tasks]
        _accept(blocks, envelope)
        return blocks

    # Trace capture is skipped for evals (it's a perf warm-up; eval correctness
    # doesn't depend on it). lm-eval carries its own per-request timeout.
    device_max_context = getattr(
        getattr(ctx.model_spec, "device_model_spec", None), "max_context", None
    )
    # Each task is scored and accepted as soon as it finishes: accepting is what
    # checkpoints the report, so a job cancelled during task N keeps 1..N-1.
    blocks: List[Block] = []
    for task in tasks:
        min_ctx = getattr(task, "min_context_required", None)
        if min_ctx and device_max_context and device_max_context < min_ctx:
            reason = (
                f"requires max_context >= {min_ctx}, device provides "
                f"{device_max_context}"
            )
            logger.warning("⏭  Skipping %s: %s.", task.task_name, reason)
            task_blocks = [_status_block(ctx, task, TestStatus.SKIP, reason)]
            _accept(task_blocks, envelope)
            blocks.extend(task_blocks)
            continue
        health = server.get_health()
        if getattr(health, "status_code", 200) != 200:
            logger.error(
                "⛔ server unhealthy mid-eval (status %s); aborting.",
                getattr(health, "status_code", "?"),
            )
            task_blocks = [_fail_block(ctx, task, "inference server not healthy")]
            _accept(task_blocks, envelope)
            blocks.extend(task_blocks)
            break
        # Keep each attempt's raw outputs, but never grade files from an older
        # attempt when this subprocess fails or produces malformed results.
        output_path = Path(ctx.output_path) / f"eval-attempt-{uuid.uuid4().hex}"
        output_path.mkdir(parents=True)
        started_at = time.perf_counter()
        rc = _run_eval_task(ctx, task, auth_token, output_path=output_path)
        elapsed_seconds = time.perf_counter() - started_at
        task_blocks = _score_task(
            ctx, task, output_path=output_path, rc=rc, elapsed_seconds=elapsed_seconds
        )
        _accept(task_blocks, envelope)
        blocks.extend(task_blocks)

    return blocks


def _score_task(
    ctx: MediaContext,
    task,
    *,
    output_path: Path,
    rc: int,
    elapsed_seconds: Optional[float],
) -> List[Block]:
    """Score one finished task using only its current attempt's result files."""
    result_files = discover_eval_results(output_path, ctx.model_spec)
    results, sample_counts = load_eval_results(result_files)
    task_blocks = blocks_for_task(
        ctx,
        task,
        results,
        sample_counts,
        elapsed_seconds=elapsed_seconds,
    )
    if task_blocks:
        return task_blocks
    # Ran but scored nothing (command failed or results unparseable) —
    # v1's report path silently drops these; we surface a FAIL block.
    return [_fail_block(ctx, task, f"no eval results parsed (rc={rc})")]


def _envelope(ctx: MediaContext) -> dict:
    return {
        **report_model_fields(ctx.model_spec),
        "device": _device_label(ctx),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _accept(blocks: List[Block], envelope: dict) -> None:
    if not blocks:
        return
    accept_blocks(blocks, envelope=envelope)


__all__ = ["run_llm_eval"]
