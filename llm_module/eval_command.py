# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""lm-eval / lmms-eval command builder for standard LLM evals."""

from __future__ import annotations

import atexit
import json
import logging
import os
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from utils.url_helpers import build_base_url
from workflow_module.engine_types import EvalLimitMode
from workflow_module.engine_types import WorkflowVenvType
from llm_module.request_overrides import resolve_request_body
from workflow_module.venv_provisioner import get_venv_provisioner
from workflows.eval_shard import (
    SHARD_RECORD_PREFIX,
    ShardError,
    parse_eval_shard,
    restrict_plan_to_doc_ids,
    shard_record,
    shard_samples,
)

if TYPE_CHECKING:
    from reference_config.evals.eval_config import EvalTask

logger = logging.getLogger(__name__)

SMOKE_TEST_EVAL_LIMIT = 3
# Client concurrency override (an int): replaces num_concurrent and its
# device_model_spec.max_concurrency clamp. See build_eval_command.
EVAL_CLIENT_CONCURRENCY_ENV = "TT_EVAL_CLIENT_CONCURRENCY"

# Per-request budget reserved for the prompt when clamping max_gen_toks against
# device_model_spec.max_context. A floor prevents pathological clamps on
# devices with very small max_context.
_MAX_GEN_TOKS_PROMPT_RESERVE = 1024
_MIN_OUTPUT_TOKENS = 256

# Headroom kept between the harness window and device_model_spec.max_context.
# lm-eval left-truncates to max_length - 1 - max_gen_toks, so a window equal to
# max_context lands every long prompt at exactly prompt + max_tokens ==
# max_context - 1; the gpt-oss aime25 note records empty responses on TT at that
# boundary. One paged-KV block (64 tokens) of slack keeps requests strictly
# inside the served context at negligible cost to a 32K-131K window.
_HARNESS_CONTEXT_RESERVE = 64


def _effective_model_kwargs(
    task: "EvalTask", device_max_context: Optional[int]
) -> dict:
    """Bind text-harness context to the selected deployment envelope."""
    model_kwargs = dict(task.model_kwargs or {})
    if task.workflow_venv_type in (
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ):
        return model_kwargs
    if not device_max_context:
        raise ValueError(
            f"{task.task_name} requires device_model_spec.max_context for text evals"
        )
    if isinstance(device_max_context, bool):
        raise ValueError(
            f"{task.task_name} requires positive integer max_context/max_length"
        )
    try:
        device_max_context = int(device_max_context)
        requested = int(model_kwargs.get("max_length", device_max_context))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{task.task_name} requires positive integer max_context/max_length"
        ) from exc
    if device_max_context <= 0 or requested <= 0:
        raise ValueError(
            f"{task.task_name} requires positive integer max_context/max_length"
        )
    # A shared task config may declare a larger harness window than this
    # deployment serves (e.g. 65536 on a 40960-token device). The harness can
    # only use what the device holds, so bind it to the device rather than
    # refusing a config that ran before this binding existed.
    ceiling = max(device_max_context - _HARNESS_CONTEXT_RESERVE, 1)
    requested = min(requested, ceiling)
    minimum = getattr(task, "min_context_required", None)
    if minimum is not None and device_max_context < minimum:
        raise ValueError(
            f"{task.task_name} requires device context >= {minimum}, but "
            f"device max_context={device_max_context}"
        )
    model_kwargs["max_length"] = requested
    return model_kwargs


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


def _slice_doc_ids_for_limit(task: "EvalTask", runtime_config) -> list:
    """Apply the active limit mode to ``task.doc_ids`` (lm-eval rejects
    ``--limit`` together with ``--samples``): an int keeps the first N ids, a
    fraction keeps that share (at least one), smoke-test keeps the first
    SMOKE_TEST_EVAL_LIMIT."""
    ids = list(task.doc_ids)
    limit = _resolve_eval_limit(task, runtime_config)
    if limit is None:
        return ids
    if isinstance(limit, float) and limit < 1:
        return ids[: max(1, int(len(ids) * limit))]
    return ids[: int(limit)]


def _resolve_eval_samples(task: "EvalTask", runtime_config) -> Optional[str]:
    """Resolve --eval-samples to a per-task JSON string for lm-eval's --samples.

    Returns ``None`` when eval_samples is unset, when the current task has no
    entry in the user-supplied mapping, or when the task uses a vision/audio
    (lmms-eval) backend that does not support the --samples flag.
    """
    eval_samples = getattr(runtime_config, "eval_samples", None)
    if not eval_samples and getattr(task, "doc_ids", None) is None:
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
    mapping = _parse_eval_samples_mapping(eval_samples) if eval_samples else None
    indices = mapping.get(task.task_name) if mapping else None
    if indices is None:
        if task.doc_ids is None:
            if mapping:
                logger.info(
                    "--eval-samples has no entry for task %s; skipping --samples for this task",
                    task.task_name,
                )
            return None
        indices = _slice_doc_ids_for_limit(task, runtime_config)
        logger.info(
            "Task %s uses its configured doc_ids subset (%d of %d) via --samples",
            task.task_name,
            len(indices),
            len(task.doc_ids),
        )
    if not isinstance(indices, (list, tuple)) or not all(
        isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in indices
    ):
        raise ValueError(
            f"--eval-samples entry for task '{task.task_name}' must be a list of "
            f"non-negative integers; got {indices!r}"
        )
    unique = list(dict.fromkeys(indices))
    if len(unique) != len(indices):
        # lm-eval selects documents by membership, so a repeated id runs once;
        # count it once too, or the noise band and timing would be graded on
        # an inflated n.
        logger.warning(
            "Task %s: %d duplicate doc_id(s) in the --samples list were dropped",
            task.task_name,
            len(indices) - len(unique),
        )
    logger.info(
        "Filtering task %s to %d doc_id(s) via --samples",
        task.task_name,
        len(unique),
    )
    return json.dumps({task.task_name: unique})


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARD_PLAN_SCRIPT = Path(__file__).with_name("eval_shard_plan.py")
# eval_shard_plan.UNSUPPORTED_HARNESS_RC: the venv's harness has no --samples.
_SHARD_PLAN_UNSUPPORTED_RC = 3


def _ttis_ref() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _known_issues(model_spec) -> list:
    issues = getattr(
        getattr(model_spec, "device_model_spec", None), "known_issues", None
    )
    out = []
    for issue in issues or []:
        get = (
            issue.get
            if isinstance(issue, dict)
            else lambda k, i=issue: getattr(i, k, None)
        )
        workflow = get("workflow_type")
        out.append(
            {
                "workflow_type": str(getattr(workflow, "name", workflow)),
                "task_name": get("task_name"),
                "reason": get("reason"),
            }
        )
    return out


def _shard_provenance(model_spec, device, cmd: List[str], output_dir_path) -> dict:
    """What every shard of one merge must share: package, image, ref, command."""
    device_spec = getattr(model_spec, "device_model_spec", None)
    env_vars = getattr(device_spec, "env_vars", None) or {}
    # The command minus its per-run output path: same task, same settings.
    stable = [str(c) for c in cmd if str(c) != str(output_dir_path)]
    return {
        "hf_model_repo": getattr(model_spec, "hf_model_repo", None),
        "package_manifest_sha256": env_vars.get("QUETZAL_BUNDLE_MANIFEST_SHA256"),
        "docker_image": getattr(model_spec, "docker_image", None),
        "ttis_ref": _ttis_ref(),
        "eval_command_sha": hashlib.sha256(json.dumps(stable).encode()).hexdigest(),
        "device": getattr(device, "name", str(device)),
        "model_status": str(
            getattr(
                getattr(model_spec, "status", None),
                "name",
                getattr(model_spec, "status", None),
            )
        ),
        "known_issues": _known_issues(model_spec),
    }


def _resolve_eval_shard(
    task: "EvalTask",
    runtime_config,
    *,
    task_venv_path: Path,
    model_spec,
    device,
    cmd: List[str],
    lm_eval_args: List[str],
    output_path,
    output_dir_path,
) -> Optional[str]:
    """--eval-shard i/N -> this shard's lm-eval --samples JSON, or None.

    Plans the task in the eval venv (leaf doc counts, few-shot sampler, metric
    aggregations), restricts it to the task's configured ``doc_ids`` when it
    has them, refuses a task that cannot shard exactly, and writes the shard
    record next to the lm-eval output for the merge to verify.

    ``lm_eval_args`` are the eval's own lm_eval arguments (everything after the
    executable or wrapper): the planner loads the tasks through the harness's
    parser from exactly these, so ``--metadata`` (merged with ``--model_args``),
    ``--include_path`` and ``--num_fewshot`` select the same doc set as the run.
    Whether the venv's harness can select docs at all (``--samples``) is checked
    by the planner from the harness itself, not from the venv's name.
    """
    value = getattr(runtime_config, "eval_shard", None)
    if not value:
        return None
    index, count = parse_eval_shard(value)
    if getattr(runtime_config, "eval_samples", None):
        raise ShardError("--eval-shard and --eval-samples are mutually exclusive")
    plan_path = Path(output_path) / f"{SHARD_RECORD_PREFIX}{task.task_name}.plan.json"
    plan_cmd = [
        str(task_venv_path / "bin" / "python"),
        str(_SHARD_PLAN_SCRIPT),
        "--out",
        str(plan_path),
        "--",
        *[str(a) for a in lm_eval_args],
    ]
    logger.info("Planning %s shard %s: %s", task.task_name, value, plan_cmd)
    planned = subprocess.run(plan_cmd, stderr=subprocess.PIPE, text=True)
    if planned.returncode == _SHARD_PLAN_UNSUPPORTED_RC:
        raise ShardError(
            f"--eval-shard refused for {task.task_name} on "
            f"{task.workflow_venv_type.name}: {(planned.stderr or '').strip()}"
        )
    if planned.returncode != 0:
        raise ShardError(
            f"shard planning failed for {task.task_name} (rc={planned.returncode}): "
            f"{(planned.stderr or '').strip()[-2000:]}"
        )
    plan = json.loads(plan_path.read_text())
    if task.doc_ids is not None:
        # The unsharded run evaluates only the configured doc set (sent as
        # --samples by _resolve_eval_samples), so the shards partition that
        # set, and the plan -- hence every shard record -- carries it.
        plan = restrict_plan_to_doc_ids(plan, task.task_name, task.doc_ids)
    samples = shard_samples(plan, index, count)  # refuses inexact plans
    record = shard_record(
        task.task_name,
        plan,
        index,
        count,
        _shard_provenance(model_spec, device, cmd, output_dir_path),
        run_id=getattr(runtime_config, "run_id", None),
    )
    record_path = Path(output_path) / f"{SHARD_RECORD_PREFIX}{task.task_name}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    logger.info(
        "Shard %s of %s: %s",
        value,
        task.task_name,
        {leaf: len(ix) for leaf, ix in samples.items()},
    )
    return json.dumps(samples, separators=(",", ":"))


def _check_request_overrides_supported(task, request_body: dict, model_spec) -> None:
    """request_body / thinking run through the request-overrides wrapper: the
    lm-eval adapters it patches, or an lmms-eval model that sends through the
    openai SDK's chat client (the body rides in the SDK's extra_body)."""
    from llm_module.lm_eval_request_overrides import (
        SUPPORTED_EVAL_CLASSES,
        SUPPORTED_LMMS_EVAL_CLASSES,
    )

    if task.workflow_venv_type in [
        WorkflowVenvType.EVALS_VISION,
        WorkflowVenvType.EVALS_AUDIO,
    ]:
        if task.eval_class not in SUPPORTED_LMMS_EVAL_CLASSES:
            raise ValueError(
                f"request_body/thinking on {task.task_name}: lmms-eval model "
                f"{task.eval_class!r} is not one of the openai-SDK clients the "
                "request-overrides wrapper patches "
                f"({', '.join(SUPPORTED_LMMS_EVAL_CLASSES)})"
            )
    elif task.eval_class not in SUPPORTED_EVAL_CLASSES:
        raise ValueError(
            f"request_body/thinking on {task.task_name}: eval_class "
            f"{task.eval_class!r} is not one of the lm-eval adapters the "
            f"request-overrides wrapper patches ({', '.join(SUPPORTED_EVAL_CLASSES)})"
        )
    elif "chat_template_kwargs" in request_body and not task.use_chat_api:
        raise ValueError(
            f"{task.task_name}: chat_template_kwargs (thinking={task.thinking!r}) "
            "only reach the server on the chat API; the completions API renders "
            "the chat template client-side. Set use_chat_api=True."
        )
    thinking = getattr(task, "thinking", None)
    if thinking is None:
        return
    # Warn when the serving entry's own switch is spelled differently: the
    # override would then not flip what the server flips by default.
    vllm_args = (
        getattr(getattr(model_spec, "device_model_spec", None), "vllm_args", None) or {}
    )
    raw_defaults = vllm_args.get("default-chat-template-kwargs")
    if not raw_defaults:
        return
    try:
        defaults = (
            json.loads(raw_defaults)
            if isinstance(raw_defaults, str)
            else dict(raw_defaults)
        )
    except (TypeError, ValueError):
        return
    if defaults and task.thinking_kwarg not in defaults:
        logger.warning(
            "%s: thinking=%s sets chat_template_kwargs[%r], but the serving entry's "
            "default-chat-template-kwargs uses %s; set thinking_kwarg to match",
            task.task_name,
            thinking,
            task.thinking_kwarg,
            sorted(defaults),
        )


def build_eval_command(
    task: "EvalTask",
    model_spec,
    device,
    output_path,
    service_port,
    runtime_config=None,
    deploy_url: str = "http://127.0.0.1",
    tokenizer_path: Optional[str] = None,
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
    task_venv_path = get_venv_provisioner().venv_path(task.workflow_venv_type)
    if task.use_chat_api:
        api_url = f"{base_url}/chat/completions"
    else:
        api_url = f"{base_url}/completions"

    # Clamp client concurrency/batch_size to the server's device_model_spec.max_concurrency.
    device_max_concurrency = getattr(
        getattr(model_spec, "device_model_spec", None), "max_concurrency", None
    )

    effective_max_concurrent = task.max_concurrent
    # Operator override for bring-your-own-server GPU runs, where the server's
    # max_num_seqs is set by the operator, not by device_model_spec (the
    # Colab GPU-reference runner sets both together). It changes only how
    # many requests are in flight; prompts, sampling and scoring are the same.
    client_override = os.environ.get(EVAL_CLIENT_CONCURRENCY_ENV)
    if client_override and effective_max_concurrent:
        logger.info(
            f"{EVAL_CLIENT_CONCURRENCY_ENV}={client_override}: {task.task_name} "
            f"num_concurrent {task.max_concurrent} -> {client_override}"
        )
        effective_max_concurrent = int(client_override)
    elif effective_max_concurrent and device_max_concurrency:
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
    effective_model_kwargs = _effective_model_kwargs(task, device_max_context)
    effective_gen_kwargs = _clamp_max_gen_toks(
        task.gen_kwargs,
        effective_model_kwargs.get("max_length"),
        task.task_name,
    )
    if getattr(task, "propagate_seed_to_gen_kwargs", True):
        effective_gen_kwargs = _inject_seed_into_gen_kwargs(
            effective_gen_kwargs, getattr(task, "seed", None)
        )
    # Quetzal long-generation tasks with no explicit timeout get one sized from
    # the device's KV capacity and decode rate instead of lm-eval's 1800 s
    # default, which cuts off full-length reasoning generations.
    from reference_config.evals.eval_config import resolve_capacity_request_timeout

    capacity_timeout = resolve_capacity_request_timeout(
        task,
        model_spec,
        gen_kwargs=effective_gen_kwargs,
        num_concurrent=effective_max_concurrent,
    )
    if capacity_timeout is not None:
        effective_model_kwargs["timeout"] = str(capacity_timeout)
        logger.info(
            f"{task.task_name}: lm-eval per-request timeout {capacity_timeout} s "
            "derived from device KV capacity and decode floor (no explicit timeout)"
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
    if getattr(task, "max_attempts", None) is not None:
        # max_attempts is a TOTAL attempt count including the first. lm-eval's
        # max_retries counts only the retries after it, so passing the total
        # straight through gives one attempt too many (max_attempts=1 -> 2).
        eval_max_retries = task.max_attempts - 1
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
        lm_eval_exec = task_venv_path / "bin" / "lmms-eval"
    else:
        lm_eval_exec = task_venv_path / "bin" / "lm_eval"

    lm_eval_prefix = [str(lm_eval_exec)]
    request_body = resolve_request_body(task)
    if request_body:
        _check_request_overrides_supported(task, request_body, model_spec)
        # One wrapper covers every payload patch for a task that opts into
        # request overrides (body, seed drop, reasoning capture). Tasks without
        # an override keep their existing launch path untouched, including the
        # capture_reasoning flag, which the plain lm_eval entry point does not
        # act on.
        lm_eval_prefix = [
            str(task_venv_path / "bin" / "python"),
            str(Path(__file__).with_name("lm_eval_request_overrides.py")),
            "--request-body",
            json.dumps(request_body, sort_keys=True),
        ]
        if not getattr(task, "propagate_seed_to_gen_kwargs", True):
            lm_eval_prefix.append("--drop-server-seed")
        if getattr(task, "capture_reasoning", False):
            lm_eval_prefix.append("--preserve-reasoning")
        if task.workflow_venv_type in [
            WorkflowVenvType.EVALS_VISION,
            WorkflowVenvType.EVALS_AUDIO,
        ]:
            lm_eval_prefix += ["--harness", "lmms-eval"]
        lm_eval_prefix.append("--")
    # TODO: remove this once diffusiongemma vLLM can ignore the seed gen kwarg
    # https://github.com/tenstorrent/tt-inference-server/issues/4993
    elif not getattr(task, "propagate_seed_to_gen_kwargs", True):
        if task.workflow_venv_type in [
            WorkflowVenvType.EVALS_VISION,
            WorkflowVenvType.EVALS_AUDIO,
        ]:
            raise ValueError(
                f"propagate_seed_to_gen_kwargs=False on {task.task_name} needs "
                "the lm-eval no-server-seed wrapper, which lmms-eval tasks "
                "cannot use"
            )
        # The pinned lm-eval adapters always copy their model seed into OpenAI
        # payloads, independently of --gen_kwargs. Use the scoped wrapper so
        # --seed still controls harness RNGs without reaching the server.
        lm_eval_prefix = [
            str(task_venv_path / "bin" / "python"),
            str(Path(__file__).with_name("lm_eval_no_server_seed.py")),
        ]

    if tokenizer_path is not None:
        effective_model_kwargs["tokenizer"] = tokenizer_path
    model_kwargs_list = [f"{k}={v}" for k, v in effective_model_kwargs.items()]
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
            *lm_eval_prefix,
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
            *lm_eval_prefix,
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
            *lm_eval_prefix,
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
                task_venv_path
                / "llama-cookbook/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval"
                / f"work_dir_{model_spec.model_name}"
            )
            staging_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"meta_eval_{model_spec.model_name}_",
                    dir=task_venv_path,
                )
            )
            atexit.register(shutil.rmtree, staging_dir, ignore_errors=True)
            staging_work_dir = staging_dir / "work_dir"
            os.symlink(meta_data_dir, staging_work_dir)
            cmd.append(staging_work_dir)
            os.chdir(staging_dir)
        else:
            cmd.append(task_venv_path / task.include_path)
            os.chdir(task_venv_path)
    if task.apply_chat_template:
        cmd.append("--apply_chat_template")  # Flag argument (no value)
    # Only when the task sets it: None leaves the harness default (on under
    # --apply_chat_template) and the command line unchanged.
    if getattr(task, "fewshot_as_multiturn", None) is not None:
        cmd.extend(
            ["--fewshot_as_multiturn", "true" if task.fewshot_as_multiturn else "false"]
        )

    # Add metadata parameter if specified (needed for tasks like RULER)
    if getattr(task, "custom_dataset_kwargs", None):
        cmd.append("--metadata")
        cmd.append(json.dumps(task.custom_dataset_kwargs))

    # Add safety flags for code evaluation tasks
    if task.workflow_venv_type == WorkflowVenvType.EVALS_COMMON:
        cmd.append("--trust_remote_code")
        cmd.append("--confirm_run_unsafe_code")

    samples_arg = _resolve_eval_samples(task, runtime_config)
    shard_arg = _resolve_eval_shard(
        task,
        runtime_config,
        task_venv_path=task_venv_path,
        model_spec=model_spec,
        device=device,
        cmd=[str(c) for c in cmd],
        lm_eval_args=[str(c) for c in cmd[len(lm_eval_prefix) :]],
        output_path=output_path,
        output_dir_path=output_dir_path,
    )
    if shard_arg is not None:
        samples_arg = shard_arg
    if samples_arg is not None:
        cmd.extend(["--samples", samples_arg])
    else:
        # lm-eval rejects --limit together with --samples; a doc_ids task
        # already folded its limit into the sample list.
        limit_arg = _resolve_eval_limit(task, runtime_config)
        if limit_arg is not None:
            cmd.extend(["--limit", str(limit_arg)])

    # force all cmd parts to be strs
    cmd = [str(c) for c in cmd]
    return cmd


__all__ = ["build_eval_command", "SMOKE_TEST_EVAL_LIMIT"]
