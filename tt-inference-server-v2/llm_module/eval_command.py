# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""lm-eval / lmms-eval command builder for standard LLM evals."""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from utils.url_helpers import build_base_url
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

if TYPE_CHECKING:
    from evals.eval_config import EvalTask

logger = logging.getLogger(__name__)

SMOKE_TEST_EVAL_LIMIT = 3

# Per-request budget reserved for the prompt when clamping max_gen_toks against
# device_model_spec.max_context. A floor prevents pathological clamps on
# devices with very small max_context.
_MAX_GEN_TOKS_PROMPT_RESERVE = 1024
_MIN_OUTPUT_TOKENS = 256


def _clamp_max_gen_toks(
    gen_kwargs: dict, device_max_context: Optional[int], task_name: str
) -> dict:
    """Return a copy of gen_kwargs with max_gen_toks clamped to fit within
    device_max_context after reserving room for the prompt. Returns the
    original dict (no copy) when there is nothing to clamp."""
    if not device_max_context or gen_kwargs.get("max_gen_toks") is None:
        return gen_kwargs
    try:
        requested = int(gen_kwargs["max_gen_toks"])
    except (TypeError, ValueError):
        return gen_kwargs
    ceiling = max(_MIN_OUTPUT_TOKENS, device_max_context - _MAX_GEN_TOKS_PROMPT_RESERVE)
    if requested <= ceiling:
        return gen_kwargs
    out = dict(gen_kwargs)
    out["max_gen_toks"] = ceiling
    logger.info(
        f"Clamping {task_name} max_gen_toks: "
        f"{requested} -> {ceiling} "
        f"(device_model_spec.max_context={device_max_context}, "
        f"reserving {_MAX_GEN_TOKS_PROMPT_RESERVE} tokens for the prompt)"
    )
    return out


def _inject_seed_into_gen_kwargs(gen_kwargs: dict, seed) -> dict:
    """Return a copy of gen_kwargs with the task seed added if not already set.

    lm-eval's ``--seed`` only seeds the harness RNG (dataset/fewshot shuffling);
    it does not reach the server's SamplingParams. For ``do_sample=true`` tasks
    the request must carry its own seed to be reproducible, so propagate
    ``task.seed`` into gen_kwargs (the per-request sampling params). A seed
    already present in gen_kwargs wins."""
    if seed is None or "seed" in gen_kwargs:
        return gen_kwargs
    out = dict(gen_kwargs)
    out["seed"] = str(seed)
    return out


def _get_limit_mode(runtime_config) -> Optional[EvalLimitMode]:
    if runtime_config is None or not getattr(
        runtime_config, "limit_samples_mode", None
    ):
        return None
    return EvalLimitMode.from_string(runtime_config.limit_samples_mode)


def _parse_eval_samples_mapping(value: Optional[str]) -> Optional[dict]:
    """Parse the --eval-samples value into a dict.

    Accepts either a JSON string of shape ``{"task_name": [int, ...], ...}``
    or a path to a JSON file containing the same shape.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except TypeError as exc:
        raise ValueError(
            "--eval-samples must be a JSON string or a path to a JSON file; "
            f"got {type(value).__name__}"
        ) from exc
    except json.JSONDecodeError:
        path = Path(value)
        if not path.is_file():
            raise ValueError(
                f"--eval-samples value is not valid JSON and not an existing file: {value}"
            )
        try:
            parsed = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"--eval-samples file does not contain valid JSON: {path}"
            ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "--eval-samples must decode to a JSON object mapping task_name -> "
            f"[int, ...]; got {type(parsed).__name__}"
        )
    return parsed


def _resolve_eval_limit(task: "EvalTask", runtime_config) -> Optional[object]:
    limit_mode = _get_limit_mode(runtime_config)
    if limit_mode is None:
        return None
    if limit_mode == EvalLimitMode.SMOKE_TEST:
        return SMOKE_TEST_EVAL_LIMIT
    return task.limit_samples_map.get(limit_mode)


def _resolve_eval_samples(task: "EvalTask", runtime_config) -> Optional[str]:
    """Resolve --eval-samples to a per-task JSON string for lm-eval's --samples.

    Returns ``None`` when eval_samples is unset, when the current task has no
    entry in the user-supplied mapping, or when the task uses a vision/audio
    (lmms-eval) backend that does not support the --samples flag.
    """
    eval_samples = getattr(runtime_config, "eval_samples", None)
    if not eval_samples:
        return None
    if task.workflow_venv_type in (
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ):
        logger.warning(
            "--eval-samples is not supported for vision/audio evals; "
            "ignoring for task %s",
            task.task_name,
        )
        return None
    mapping = _parse_eval_samples_mapping(eval_samples)
    if mapping is None:
        return None
    indices = mapping.get(task.task_name)
    if indices is None:
        logger.info(
            "--eval-samples has no entry for task %s; skipping --samples for this task",
            task.task_name,
        )
        return None
    if not isinstance(indices, (list, tuple)) or not all(
        isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in indices
    ):
        raise ValueError(
            f"--eval-samples entry for task '{task.task_name}' must be a list of "
            f"non-negative integers; got {indices!r}"
        )
    logger.info(
        "Filtering task %s to %d doc_id(s) via --samples",
        task.task_name,
        len(indices),
    )
    return json.dumps({task.task_name: list(indices)})


def build_eval_command(
    task: "EvalTask",
    model_spec,
    device,
    output_path,
    service_port,
    runtime_config=None,
    deploy_url: str = "http://127.0.0.1",
) -> List[str]:
    """Build the lm_eval / lmms-eval command for one standard eval task."""
    if task.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC:
        raise ValueError(
            "build_eval_command does not handle EVALS_AGENTIC tasks; agentic "
            "evals run through AgenticWorkflow / llm_module.drivers.agentic."
        )

    # Audio models use tt-media-server which has endpoints at /audio (not /v1/audio)
    # Other models use vLLM which has endpoints at /v1
    host_with_port = build_base_url(deploy_url, service_port)
    if task.workflow_venv_type == WorkflowVenvType.EVALS_AUDIO:
        base_url = host_with_port
    else:
        base_url = f"{host_with_port}/v1"
    eval_class = task.eval_class
    task_venv_config = VENV_CONFIGS[task.workflow_venv_type]
    if task.use_chat_api:
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/completions"

    # Clamp client concurrency/batch_size to the server's device_model_spec.max_concurrency.
    device_max_concurrency = getattr(
        getattr(model_spec, "device_model_spec", None), "max_concurrency", None
    )

    effective_max_concurrent = task.max_concurrent
    if effective_max_concurrent and device_max_concurrency:
        effective_max_concurrent = min(effective_max_concurrent, device_max_concurrency)
        if effective_max_concurrent != task.max_concurrent:
            logger.info(
                f"Clamping {task.task_name} num_concurrent: "
                f"{task.max_concurrent} -> {effective_max_concurrent} "
                f"(device_model_spec.max_concurrency={device_max_concurrency})"
            )

    effective_batch_size = task.batch_size
    if effective_batch_size and device_max_concurrency:
        effective_batch_size = min(effective_batch_size, device_max_concurrency)
        if effective_batch_size != task.batch_size:
            logger.info(
                f"Clamping {task.task_name} batch_size: "
                f"{task.batch_size} -> {effective_batch_size} "
                f"(device_model_spec.max_concurrency={device_max_concurrency})"
            )

    # Clamp gen_kwargs.max_gen_toks so prompt + max_tokens fits within the
    # server's max_context.
    device_max_context = getattr(
        getattr(model_spec, "device_model_spec", None), "max_context", None
    )
    effective_gen_kwargs = _clamp_max_gen_toks(
        task.gen_kwargs, device_max_context, task.task_name
    )
    effective_gen_kwargs = _inject_seed_into_gen_kwargs(
        effective_gen_kwargs, getattr(task, "seed", None)
    )

    optional_model_args = []
    if effective_max_concurrent:
        optional_model_args.append(f"num_concurrent={effective_max_concurrent}")
    # Fast-fail 4xx when DeviceModelSpec opts in (forge LLMs at tight
    # max_context). EVALS_COMMON only: lm-eval 0.4.3 in EVALS_META rejects
    # the kwarg.
    eval_max_retries = getattr(
        getattr(model_spec, "device_model_spec", None), "eval_max_retries", None
    )
    if (
        eval_max_retries is not None
        and task.workflow_venv_type == WorkflowVenvType.EVALS_COMMON
    ):
        optional_model_args.append(f"max_retries={eval_max_retries}")

    # lm-eval (text) expects full completions api route in base_url
    # lmms-eval (vision) expects base_url WITHOUT the endpoint path
    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
    ]:
        _base_url = base_url
    else:
        _base_url = api_url

    # Set OPENAI_API_BASE for vision and audio models
    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ]:
        os.environ["OPENAI_API_BASE"] = base_url

    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ]:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lmms-eval"
    else:
        lm_eval_exec = task_venv_config.venv_path / "bin" / "lm_eval"

    model_kwargs_list = [f"{k}={v}" for k, v in task.model_kwargs.items()]
    model_kwargs_list += optional_model_args
    model_kwargs_str = ",".join(model_kwargs_list)

    # build gen_kwargs string
    gen_kwargs_list = [f"{k}={v}" for k, v in effective_gen_kwargs.items()]
    gen_kwargs_str = ",".join(gen_kwargs_list)

    # set output_dir
    # results go to {output_dir_path}/{hf_repo}/results_{timestamp}
    output_dir_path = Path(output_path) / f"eval_{model_spec.model_id}"

    # fmt: off
    if task.workflow_venv_type == WorkflowVenvType.EVALS_VISION:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model_version={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
            "--seed", task.seed,
            "--num_fewshot", task.num_fewshot,
            "--batch_size", effective_batch_size,
            "--log_samples",
            "--show_config",
        ]
    elif task.workflow_venv_type == WorkflowVenvType.EVALS_AUDIO:
        cmd = [
            str(lm_eval_exec),
            "--model", eval_class,
            "--model_args", (
                f"model={model_spec.hf_model_repo},"
                f"base_url={base_url},"
                f"{model_kwargs_str}"
            ),
            "--tasks", task.task_name,
            "--batch_size", str(effective_batch_size),
            "--output_path", str(output_dir_path),
            "--log_samples",
        ]
    else:
        cmd = [
            str(lm_eval_exec),
            "--tasks", task.task_name,
            "--model", eval_class,
            "--model_args", (
                f"model={model_spec.hf_model_repo},"
                f"base_url={_base_url},"
                f"tokenizer_backend={task.tokenizer_backend},"
                f"{model_kwargs_str}"
            ),
            "--gen_kwargs", gen_kwargs_str,
            "--output_path", output_dir_path,
            "--seed", task.seed,
            "--num_fewshot", task.num_fewshot,
            "--batch_size", effective_batch_size,
            "--log_samples",
            "--show_config",
        ]
    # fmt: on

    if task.include_path:
        cmd.append("--include_path")
        if task.workflow_venv_type == WorkflowVenvType.EVALS_META:
            # lm-eval meta_* task YAMLs hardcode `./work_dir/joined_*.parquet`
            # relative to cwd. Give each invocation its own staging dir with a
            # symlink that masquerades as it, so parallel runs don't race.
            meta_data_dir = (
                task_venv_config.venv_path
                / "llama-cookbook/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval"
                / f"work_dir_{model_spec.model_name}"
            )
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"meta_eval_{model_spec.model_name}_",
                    dir=task_venv_config.venv_path,
                )
            )
            atexit.register(shutil.rmtree, staging_dir, ignore_errors=True)
            staging_work_dir = staging_dir / "work_dir"
            os.symlink(meta_data_dir, staging_work_dir)
            cmd.append(staging_work_dir)
            os.chdir(staging_dir)
        else:
            cmd.append(task_venv_config.venv_path / task.include_path)
            os.chdir(task_venv_config.venv_path)
    if task.apply_chat_template:
        cmd.append("--apply_chat_template")  # Flag argument (no value)

    # Add metadata parameter if specified (needed for tasks like RULER)
    if getattr(task, "custom_dataset_kwargs", None):
        cmd.append("--metadata")
        cmd.append(json.dumps(task.custom_dataset_kwargs))

    # Add safety flags for code evaluation tasks
    if task.workflow_venv_type == WorkflowVenvType.EVALS_COMMON:
        cmd.append("--trust_remote_code")
        cmd.append("--confirm_run_unsafe_code")

    samples_arg = _resolve_eval_samples(task, runtime_config)
    if samples_arg is not None:
        cmd.extend(["--samples", samples_arg])

    limit_arg = _resolve_eval_limit(task, runtime_config)
    if limit_arg is not None:
        cmd.extend(["--limit", str(limit_arg)])

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


__all__ = ["build_eval_command", "SMOKE_TEST_EVAL_LIMIT"]
