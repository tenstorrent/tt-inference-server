# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Build the LLM benchmark sweep from the model spec."""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import LLMRunConfig

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
            )
        )

    if not configs:
        logger.warning(
            "No text benchmark params for model_id=%s on device=%s",
            model_spec.model_id,
            getattr(device, "name", device),
        )
    return configs


__all__ = ["get_llm_configs"]
