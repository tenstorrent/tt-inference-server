# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Build the LLM benchmark sweep from the model spec."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import LLMRunConfig, ServerConnection, StructuredOutputRunConfig
from .benchmark_protocol import fixed_workload_repetitions, uses_fixed_workload

logger = logging.getLogger(__name__)


def get_llm_configs(
    model_spec,
    device,
    *,
    limit_samples_mode: Optional[str] = None,
) -> List[LLMRunConfig]:
    """Return the text-benchmark sweep for ``model_spec`` on ``device``.

    ``device`` is a ``DeviceTypes`` value (``ctx.device``). Structured-output
    params (:func:`get_structured_output_configs`) and media (CNN/image/VLM)
    params are skipped — this sweep is text-only, so any param without both
    ``isl`` and ``osl`` is dropped.
    ``limit_samples_mode`` honours v1's smoke-test selection when set.
    """
    benchmark_config = _benchmark_config(model_spec, device, limit_samples_mode)

    configured_devices = {
        dev for task in benchmark_config.tasks for dev in task.param_map
    }
    if device not in configured_devices:
        available = sorted(getattr(dev, "name", str(dev)) for dev in configured_devices)
        raise ValueError(
            f"No benchmark params for device={getattr(device, 'name', device)!r} "
            f"for model_id={model_spec.model_id!r}. Configured devices: {available}."
        )

    text_params = [
        params
        for task in benchmark_config.tasks
        for params in task.param_map.get(device, [])
        if params.isl is not None
        and params.osl is not None
        and params.task_type == "text"
    ]

    metadata = getattr(model_spec, "metadata", None) or {}
    token_timing = uses_fixed_workload(metadata)
    repetitions = fixed_workload_repetitions(metadata) if token_timing else 1
    require_complete_metrics = bool(
        metadata.get("benchmark_require_complete_metrics", False)
    )

    def target_key(params):
        shape = (params.isl, params.osl, params.max_concurrency)
        return (*shape, params.num_prompts) if token_timing else shape

    targets_by_shape = {
        target_key(params): params.targets for params in text_params if params.targets
    }
    priority_by_shape = {
        (params.isl, params.osl, params.max_concurrency): params.priority
        for params in text_params
        if getattr(params, "priority", None)
    }
    target_priorities_by_shape = {
        (params.isl, params.osl, params.max_concurrency): params.target_priorities
        for params in text_params
        if getattr(params, "target_priorities", None)
    }
    goodput_by_shape = {
        (params.isl, params.osl, params.max_concurrency): params.goodput
        for params in text_params
        if getattr(params, "goodput", None)
    }

    output_block_size = int(metadata.get("output_block_size", 1) or 1)
    configs: List[LLMRunConfig] = []
    seen = set()
    for params in text_params:
        key = (params.isl, params.osl, params.max_concurrency, params.num_prompts)
        if key in seen:
            continue
        seen.add(key)
        shape = (params.isl, params.osl, params.max_concurrency)
        targets = dict(targets_by_shape.get(target_key(params), {}))
        has_targets = bool(targets)
        configs.append(
            LLMRunConfig(
                isl=params.isl,
                osl=params.osl,
                max_concurrency=params.max_concurrency,
                num_prompts=params.num_prompts,
                targets=targets,
                output_block_size=output_block_size,
                token_timing=token_timing,
                require_complete_metrics=require_complete_metrics,
                full_workload_warmup=token_timing and has_targets,
                repetitions=repetitions if token_timing and has_targets else 1,
                custom_dataset_path=(
                    Path(
                        f"speed_bench_prompts_isl-{params.isl}_n-{params.num_prompts}.jsonl"
                    )
                    if output_block_size > 1
                    else None
                ),
                priority=priority_by_shape.get(shape),
                target_priorities=target_priorities_by_shape.get(shape),
                goodput=goodput_by_shape.get(shape),
            )
        )

    if not configs:
        logger.warning(
            "No text benchmark params for model_id=%s on device=%s",
            model_spec.model_id,
            getattr(device, "name", device),
        )
    return configs


def _benchmark_config(model_spec, device, limit_samples_mode: Optional[str]):
    """The model's benchmark config from the target pack, cut down to the
    smoke-test selection when ``limit_samples_mode`` asks for it."""
    from workflow_module.engine_types import EvalLimitMode
    from workflow_module.target_pack import get_target_pack

    pack = get_target_pack()
    benchmark_config = pack.benchmark_config(model_spec)
    if (
        limit_samples_mode
        and EvalLimitMode.from_string(limit_samples_mode) == EvalLimitMode.SMOKE_TEST
    ):
        benchmark_config = pack.smoke_test_benchmark_config(benchmark_config, device)
    return benchmark_config


def structured_output_chat_template_kwargs(model_spec) -> Optional[Dict[str, Any]]:
    """Chat-template arguments for the structured-output benchmark's prompts.

    With a reasoning parser on the server, vLLM constrains the output only
    once the reasoning has ended, judged from the prompt. vLLM's benchmark
    sends raw-text prompts (and chat-templated ones with the template's default
    thinking), so a reasoning model would think through the whole output and
    the grammar would never apply. For such models the prompts are rendered as
    chat turns with thinking turned off, over the serving entry's
    ``default_chat_template_kwargs`` (the client renders the template, so the
    server default would not apply). Models without a reasoning parser keep
    the benchmark's prompts unchanged (``None``). The serving entry's
    ``vllm_args`` decide (either spelling of the flag); the metadata's
    ``reasoning_parser_name`` does not start the server with a parser.
    """
    device_model_spec = getattr(model_spec, "device_model_spec", None)
    vllm_args = getattr(device_model_spec, "vllm_args", None) or {}

    def arg(name: str) -> Any:
        return vllm_args.get(name) or vllm_args.get(name.replace("_", "-"))

    if not arg("reasoning_parser"):
        return None
    defaults = arg("default_chat_template_kwargs") or {}
    if isinstance(defaults, str):
        defaults = json.loads(defaults)
    return {**defaults, "enable_thinking": False}


def get_structured_output_configs(
    model_spec,
    device,
    *,
    limit_samples_mode: Optional[str] = None,
) -> List[StructuredOutputRunConfig]:
    """Return the structured-output benchmark runs for ``model_spec`` on
    ``device``: the ``structured_output`` params of its benchmark config
    (``STRUCTURED_OUTPUT_PAIRS``), with the chat-template arguments from
    :func:`structured_output_chat_template_kwargs`."""
    benchmark_config = _benchmark_config(model_spec, device, limit_samples_mode)
    chat_template_kwargs = structured_output_chat_template_kwargs(model_spec)
    return [
        StructuredOutputRunConfig(
            dataset=params.structured_dataset,
            structured_output_ratio=params.structured_output_ratio,
            osl=params.osl,
            max_concurrency=params.max_concurrency,
            num_prompts=params.num_prompts,
            chat_template_kwargs=chat_template_kwargs,
        )
        for task in benchmark_config.tasks
        for params in task.param_map.get(device, [])
        if params.task_type == "structured_output"
    ]


def ensure_custom_dataset(
    config: LLMRunConfig,
    server: ServerConnection,
    output_dir: Path,
) -> LLMRunConfig:
    """Materialize a configured custom dataset and resolve its path."""
    path = config.custom_dataset_path
    if path is None:
        return config
    resolved = path if path.is_absolute() else output_dir / path
    if not resolved.exists():
        from .speed_bench_prompts import write_speed_bench_prompt_file

        try:
            write_speed_bench_prompt_file(
                output_path=resolved,
                model=server.model,
                target_isl=config.isl,
                num_prompts=config.num_prompts,
                trust_remote_code=server.tokenizer_trust_remote_code,
            )
        except Exception as build_error:
            raise RuntimeError(
                "SPEED-Bench prompt construction failed for "
                f"{server.model} isl={config.isl}; refusing to run a mislabeled "
                "random-input benchmark"
            ) from build_error
    if resolved != path:
        return replace(config, custom_dataset_path=resolved)
    return config


__all__ = [
    "ensure_custom_dataset",
    "get_llm_configs",
    "get_structured_output_configs",
    "structured_output_chat_template_kwargs",
]
