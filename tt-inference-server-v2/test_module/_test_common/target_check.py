# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .report_types import ReportCheckTypes
from ..context import MediaContext

logger = logging.getLogger(__name__)


# v2-local performance targets JSON. Override at runtime by pointing
# OVERRIDE_BENCHMARK_TARGETS at a different file
_DEFAULT_TARGETS_PATH = (
    Path(__file__).parent / "targets" / "model_performance_reference.json"
)


# For latency-style metrics (lower is better) the threshold is target * multiplier;
# for throughput-style metrics (higher is better) it is target / multiplier.
TIER_MULTIPLIERS = {
    "functional": 10,
    "complete": 2,
    "target": 1,
}


@dataclass
class PerformanceTargets:
    """Parsed performance targets from model_performance_reference.json."""

    ttft_ms: Optional[float] = None
    ttft_streaming_ms: Optional[float] = None
    tput_user: Optional[float] = None
    tput_prefill: Optional[float] = None
    e2el_ms: Optional[float] = None
    tput: Optional[float] = None
    rtr: Optional[float] = None
    tolerance: float = 0.05
    max_concurrency: Optional[int] = None
    num_eval_runs: Optional[int] = None
    task_type: str = "text"

    @classmethod
    def from_device_config(cls, device_config: Dict[str, Any]) -> "PerformanceTargets":
        if not device_config:
            return cls()
        theoretical = device_config.get("targets", {}).get("theoretical", {})
        return cls(
            ttft_ms=theoretical.get("ttft_ms"),
            ttft_streaming_ms=theoretical.get("ttft_streaming_ms"),
            tput_user=theoretical.get("tput_user"),
            tput_prefill=theoretical.get("tput_prefill"),
            e2el_ms=theoretical.get("e2el_ms"),
            tput=theoretical.get("tput"),
            rtr=theoretical.get("rtr"),
            tolerance=theoretical.get("tolerance", 0.05),
            max_concurrency=device_config.get("max_concurrency"),
            num_eval_runs=device_config.get("num_eval_runs"),
            task_type=device_config.get("task_type", "text"),
        )


def _resolve_targets_path() -> Path:
    override = os.getenv("OVERRIDE_BENCHMARK_TARGETS")
    return Path(override) if override else _DEFAULT_TARGETS_PATH


_REFERENCE_CACHE: Optional[Dict[str, Any]] = None
_REFERENCE_CACHE_PATH: Optional[Path] = None


def _load_reference() -> Dict[str, Any]:
    """Read and cache the v2 model_performance_reference JSON."""
    global _REFERENCE_CACHE, _REFERENCE_CACHE_PATH
    path = _resolve_targets_path()
    if _REFERENCE_CACHE is not None and path == _REFERENCE_CACHE_PATH:
        return _REFERENCE_CACHE
    if not path.exists():
        raise FileNotFoundError(f"Performance reference file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        _REFERENCE_CACHE = json.load(f)
    _REFERENCE_CACHE_PATH = path
    logger.info(f"Loaded performance reference from {path}")
    return _REFERENCE_CACHE


def get_performance_targets(
    model_name: str, device_str: Optional[str]
) -> PerformanceTargets:
    """Look up performance targets for ``model_name`` on ``device_str``.

    Returns an empty ``PerformanceTargets`` (all ``None``) if no entry is
    found; callers handle that as ``ReportCheckTypes.NA``.
    """
    if not device_str:
        logger.warning(
            f"No device specified for model '{model_name}'; skipping target lookup"
        )
        return PerformanceTargets()
    device_key = device_str.lower()
    reference = _load_reference()
    model_data = reference.get(model_name, {})
    device_list = model_data.get(device_key, [])
    if device_list:
        logger.info(
            f"Found performance targets for model '{model_name}' on device '{device_key}'"
        )
        return PerformanceTargets.from_device_config(device_list[0])
    logger.warning(
        f"No performance targets found for model '{model_name}' on device '{device_key}'"
    )
    return PerformanceTargets()


@dataclass
class MetricSpec:
    """One actual-vs-target comparison.

    ``target_attr`` names a field on ``PerformanceTargets`` (e.g.
    ``"ttft_ms"``). ``lower_is_better=True`` for latency-style metrics,
    ``False`` for throughput-style. ``field_name`` is the key prefix used
    in the emitted ``target_checks`` dict.
    """

    name: str
    actual: Optional[float]
    target_attr: str
    lower_is_better: bool
    field_name: str


def load_targets(ctx: MediaContext) -> PerformanceTargets:
    device_str = ctx.model_spec.cli_args.get("device")
    targets = get_performance_targets(ctx.model_spec.model_name, device_str)
    logger.info(f"Performance targets: {targets}")
    return targets


def _tier_threshold(
    target_value: float, multiplier: int, lower_is_better: bool
) -> float:
    return target_value * multiplier if lower_is_better else target_value / multiplier


def _check_from_ratio(ratio: float, lower_is_better: bool) -> ReportCheckTypes:
    passed = ratio < 1.0 if lower_is_better else ratio > 1.0
    return ReportCheckTypes.PASS if passed else ReportCheckTypes.FAIL


def evaluate_tiered(
    targets: PerformanceTargets, specs: Sequence[MetricSpec]
) -> dict[str, dict]:
    """Build the 3-tier ``target_checks`` dict.

    For each tier (functional/complete/target) and each metric spec,
    emit ``<field>``, ``<field>_ratio``, and ``<field>_check`` keys.
    Missing target → ``"Undefined"`` for value/ratio and ``NA`` for check.
    Missing actual → ``0.0`` ratio and ``NA`` check.
    """
    result: dict[str, dict] = {}
    for tier_name, multiplier in TIER_MULTIPLIERS.items():
        tier_dict: dict = {}
        for spec in specs:
            field = spec.field_name
            target_value = getattr(targets, spec.target_attr, None)
            if target_value is None:
                tier_dict[field] = "Undefined"
                tier_dict[f"{field}_ratio"] = "Undefined"
                tier_dict[f"{field}_check"] = ReportCheckTypes.NA
                continue

            threshold = _tier_threshold(target_value, multiplier, spec.lower_is_better)
            tier_dict[field] = threshold

            if spec.actual is None or not threshold:
                tier_dict[f"{field}_ratio"] = 0.0
                tier_dict[f"{field}_check"] = ReportCheckTypes.NA
                continue

            ratio = spec.actual / threshold
            tier_dict[f"{field}_ratio"] = ratio
            tier_dict[f"{field}_check"] = _check_from_ratio(ratio, spec.lower_is_better)
        result[tier_name] = tier_dict
    return result


def summary_from_tiered(target_checks: dict[str, dict]) -> ReportCheckTypes:
    """Single-line verdict derived from the strictest ``target`` tier.

    PASS only if every real (non-NA) check in the target tier passed;
    FAIL if any failed; NA if no metric could be evaluated.
    """
    target_tier = target_checks.get("target", {})
    real_checks = [
        v
        for k, v in target_tier.items()
        if k.endswith("_check") and v != ReportCheckTypes.NA
    ]
    if not real_checks:
        return ReportCheckTypes.NA
    if any(c == ReportCheckTypes.FAIL for c in real_checks):
        return ReportCheckTypes.FAIL
    return ReportCheckTypes.PASS


def _log_tiered(target_checks: dict[str, dict]) -> None:
    for tier in ("functional", "complete", "target"):
        tier_dict = target_checks.get(tier, {})
        for field_check_key, check in tier_dict.items():
            if not field_check_key.endswith("_check"):
                continue
            field = field_check_key[: -len("_check")]
            ratio = tier_dict.get(f"{field}_ratio")
            threshold = tier_dict.get(field)
            if check == ReportCheckTypes.PASS:
                logger.info(f"✅ [{tier}] {field}: ratio={ratio} threshold={threshold}")
            elif check == ReportCheckTypes.FAIL:
                logger.warning(
                    f"❌ [{tier}] {field}: ratio={ratio} threshold={threshold}"
                )


def run_tiered_check(
    ctx: MediaContext, specs: Sequence[MetricSpec]
) -> tuple[dict[str, dict], ReportCheckTypes]:
    """Convenience wrapper: load targets, build 3-tier dict, log and summarise."""
    targets = load_targets(ctx)
    target_checks = evaluate_tiered(targets, specs)
    _log_tiered(target_checks)
    summary = summary_from_tiered(target_checks)
    logger.info(f"Overall accuracy_check (target tier): {summary.name}")
    return target_checks, summary


__all__ = [
    "MetricSpec",
    "PerformanceTargets",
    "TIER_MULTIPLIERS",
    "evaluate_tiered",
    "get_performance_targets",
    "load_targets",
    "run_tiered_check",
    "summary_from_tiered",
]
