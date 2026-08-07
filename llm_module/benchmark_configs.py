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
    from reference_config.benchmarking.benchmark_config import (
        get_benchmark_config,
        select_smoke_test_benchmark_config,
    )
    from workflows.workflow_types import EvalLimitMode

    benchmark_config = get_benchmark_config(model_spec)

    if (
        limit_samples_mode
        and EvalLimitMode.from_string(limit_samples_mode) == EvalLimitMode.SMOKE_TEST
    ):
        benchmark_config = select_smoke_test_benchmark_config(benchmark_config, device)

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

    _enforce_scaling_quality_coverage(model_spec, device, configs)

    return configs


def _enforce_scaling_quality_coverage(model_spec, device, configs) -> None:
    """Fail fast when a scaling-quality-graded device's graded sweep cannot be fit.

    The scaling-quality rubric line fits time-to-first-token against input
    length separately at each graded concurrency level, so every graded
    concurrency level needs at least three distinct input lengths or the fit is
    meaningless (RFP Appendix B.1/B.2/F.1; readiness §5.7). We validate the
    *post-cap* graded set (the points that actually carry targets and will be
    graded) so context/token-budget capping that silently moves a point to a
    lower concurrency is caught here rather than producing an ungradeable run.
    """
    if not getattr(device, "grades_scaling_quality", None) or not device.grades_scaling_quality():
        return

    from reference_config.benchmarking.benchmark_config import (
        SCALING_QUALITY_MIN_INPUT_LENGTHS,
        max_gradeable_concurrency,
        min_token_pool_for_concurrency,
        scaling_quality_coverage_violations,
    )

    graded_points = [
        (c.isl, c.max_concurrency) for c in configs if c.targets
    ]
    violations = scaling_quality_coverage_violations(graded_points)
    if not violations:
        return

    dms = model_spec.device_model_spec
    max_context = getattr(dms, "max_context", None)
    max_tokens_all_users = getattr(dms, "max_tokens_all_users", None)
    model_max_concurrency = getattr(dms, "max_concurrency", None)

    detail = "; ".join(
        f"concurrency={c} has {len(isls)} input length(s) {isls}"
        for c, isls in sorted(violations.items())
    )
    hint = ""
    if None not in (max_context, max_tokens_all_users, model_max_concurrency):
        reachable_top = max_gradeable_concurrency(
            max_context=max_context,
            max_tokens_all_users=max_tokens_all_users,
            model_max_concurrency=model_max_concurrency,
        )
        try:
            pool_needed = min_token_pool_for_concurrency(
                model_max_concurrency, max_context=max_context
            )
        except ValueError:
            pool_needed = None
        hint = (
            f" With max_context={max_context}, max_tokens_all_users="
            f"{max_tokens_all_users}, max_concurrency={model_max_concurrency}, "
            f"only concurrency<={reachable_top} is reachable by "
            f"{SCALING_QUALITY_MIN_INPUT_LENGTHS} distinct input lengths."
        )
        if pool_needed is not None:
            hint += (
                f" To grade at concurrency {model_max_concurrency}, set "
                f"max_tokens_all_users_override>={pool_needed} on the "
                f"{getattr(device, 'name', device)} spec, or rescope the graded "
                f"concurrency levels."
            )

    raise ValueError(
        f"Scaling-quality coverage violation for model_id={model_spec.model_id!r} "
        f"on device={getattr(device, 'name', device)!r}: every graded concurrency "
        f"level must carry at least {SCALING_QUALITY_MIN_INPUT_LENGTHS} distinct "
        f"input lengths (RFP Appendix F.1; readiness §5.7), but {detail}.{hint}"
    )


__all__ = ["get_llm_configs"]
