# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Build the LLM benchmark sweep from the model spec."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .config import LLMRunConfig, ServerConnection

logger = logging.getLogger(__name__)


def get_llm_configs(
    model_spec,
    device,
    *,
    limit_samples_mode: Optional[str] = None,
) -> List[LLMRunConfig]:
    """Return the text-benchmark sweep for ``model_spec`` on ``device``.

    ``device`` is a ``DeviceTypes`` value (``ctx.device``). Structured-output
    and media (CNN/image/VLM) params are skipped — the LLM runner is
    text-only, so any param without both ``isl`` and ``osl`` is dropped.
    ``limit_samples_mode`` honours v1's smoke-test selection when set.
    """
    from workflow_module.engine_types import EvalLimitMode
    from workflow_module.target_pack import get_target_pack

    pack = get_target_pack()
    benchmark_config = pack.benchmark_config(model_spec)

    if (
        limit_samples_mode
        and EvalLimitMode.from_string(limit_samples_mode) == EvalLimitMode.SMOKE_TEST
    ):
        benchmark_config = pack.smoke_test_benchmark_config(benchmark_config, device)

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

    targets_by_shape = {
        (params.isl, params.osl, params.max_concurrency): params.targets
        for params in text_params
        if params.targets
    }

    metadata = getattr(model_spec, "metadata", None) or {}
    output_block_size = int(metadata.get("output_block_size", 1) or 1)
    configs: List[LLMRunConfig] = []
    seen = set()
    for params in text_params:
        key = (params.isl, params.osl, params.max_concurrency, params.num_prompts)
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            LLMRunConfig(
                isl=params.isl,
                osl=params.osl,
                max_concurrency=params.max_concurrency,
                num_prompts=params.num_prompts,
                targets=dict(
                    targets_by_shape.get(
                        (params.isl, params.osl, params.max_concurrency), {}
                    )
                ),
                output_block_size=output_block_size,
                custom_dataset_path=(
                    Path(
                        f"speed_bench_prompts_isl-{params.isl}_n-{params.num_prompts}.jsonl"
                    )
                    if output_block_size > 1
                    else None
                ),
            )
        )

    if not configs:
        logger.warning(
            "No text benchmark params for model_id=%s on device=%s",
            model_spec.model_id,
            getattr(device, "name", device),
        )
    return configs


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


__all__ = ["ensure_custom_dataset", "get_llm_configs"]
