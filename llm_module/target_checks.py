# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Grade one LLM benchmark block against its sweep point's perf targets.

The tiered targets come from the model spec's ``perf_reference`` entries
(``functional`` / ``complete`` / ``target``), carried onto the sweep point
by :func:`llm_module.benchmark_configs.get_llm_configs` and attached to the
parsed Block here, in the runner.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

from report_module.schema import Block
from workflows.workflow_types import ReportCheckTypes

logger = logging.getLogger(__name__)

# (field name in target_checks, PerformanceTarget attribute, lower_is_better)
_METRIC_SPECS: Tuple[Tuple[str, str, bool], ...] = (
    ("ttft", "ttft_ms", True),
    ("tput_user", "tput_user", False),
    ("tput", "tput", False),
)

TIER_ORDER: Tuple[str, ...] = ("functional", "complete", "target")


def graded_tier(
    target_checks: Mapping[str, Any],
) -> Optional[Tuple[str, Mapping[str, Any]]]:
    """The strictest tier present in ``target_checks``, as ``(name, checks)``.

    Strictest means last in :data:`TIER_ORDER`, so with the default
    functional/complete/target ladder this is ``target``.

    Callers must not look a tier up by name. Which tiers exist depends on the
    spec's ``perf_targets_map``, and Milestone-0 grades against a single tier
    named ``functional`` holding the published absolute value
    (``docs/rfp/m0-target-convention.md``). Hardcoding ``"target"`` silently finds
    nothing under that configuration — and "nothing" reads downstream as
    *ungradable*, not as *failed*, which is the safe-looking direction.
    """
    for tier_name in reversed(TIER_ORDER):
        tier = target_checks.get(tier_name)
        if isinstance(tier, Mapping) and tier:
            return tier_name, tier
    return None


def _measured(record: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Pull the three graded metrics out of a flat perf record.

    ``tput_user`` is per-user decode throughput: AIPerf and genai-perf
    report it directly, ``vllm bench serve`` does not, so derive it from
    mean TPOT the way v1's summary report did (``1000 / mean_tpot_ms``).
    """
    ttft = _as_float(record.get("mean_ttft_ms"))
    tput_user = _as_float(record.get("tput_user"))
    if tput_user is None:
        tpot = _as_float(record.get("mean_tpot_ms"))
        tput_user = 1000.0 / tpot if tpot else None
    return {
        "ttft": ttft,
        "tput_user": tput_user,
        "tput": _as_float(record.get("tps_decode_throughput")),
    }


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _check(ratio: float, tolerance: float, lower_is_better: bool) -> ReportCheckTypes:
    passed = ratio < (1 + tolerance) if lower_is_better else ratio > (1 - tolerance)
    return ReportCheckTypes.from_result(passed)


def build_target_checks(
    targets: Mapping[str, Any], record: Mapping[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], ReportCheckTypes]:
    """Build the tiered ``target_checks`` dict and its one-line verdict.

    ``targets`` maps a tier name to a ``workflows.utils_report.PerformanceTarget``.
    Each tier gets ``<field>`` (the target), ``<field>_ratio`` and
    ``<field>_check`` per metric; a metric the tier does not define, or one
    the tool did not measure, is ``NA`` rather than a failure.

    The verdict is the strictest tier that fully passes, reported as PASS
    only when the ``target`` tier passes at least one real check and none
    fail — matching how the acceptance check reads the same dict.
    """
    measured = _measured(record)
    target_checks: Dict[str, Dict[str, Any]] = {}
    for tier_name in TIER_ORDER:
        tier_target = targets.get(tier_name)
        if tier_target is None:
            continue
        tolerance = getattr(tier_target, "tolerance", 0.0) or 0.0
        tier: Dict[str, Any] = {}
        for field, target_attr, lower_is_better in _METRIC_SPECS:
            target_value = getattr(tier_target, target_attr, None)
            actual = measured.get(field)
            if not target_value or target_value <= 0:
                tier[f"{field}_check"] = ReportCheckTypes.NA
                continue
            tier[field] = target_value
            if actual is None:
                tier[f"{field}_ratio"] = 0.0
                tier[f"{field}_check"] = ReportCheckTypes.NA
                continue
            ratio = actual / target_value
            tier[f"{field}_ratio"] = ratio
            tier[f"{field}_check"] = _check(ratio, tolerance, lower_is_better)
        target_checks[tier_name] = tier
    return target_checks, _verdict(target_checks)


def _verdict(target_checks: Mapping[str, Mapping[str, Any]]) -> ReportCheckTypes:
    """PASS/FAIL/NA for the strictest tier this spec actually defines.

    Strictest means last in :data:`TIER_ORDER`, so with the default
    functional/complete/target ladder this is ``target``, unchanged.

    It used to read ``target`` by name. That silently mis-graded any spec with a
    custom ``perf_targets_map`` that does not define a tier of that name: the
    lookup returned ``{}``, the verdict came back ``NA``, and downstream
    acceptance then counted the point as *ungradable* rather than failed — so a
    point missing its target by any margin was accepted. Milestone-0 hits this
    directly, grading against a single tier named ``functional`` that holds the
    published absolute value (docs/rfp/m0-target-convention.md).
    """
    for tier_name in reversed(TIER_ORDER):
        tier = target_checks.get(tier_name)
        if not isinstance(tier, Mapping):
            continue
        real = [
            value
            for name, value in tier.items()
            if name.endswith("_check") and value != ReportCheckTypes.NA
        ]
        if not real:
            continue
        if any(value == ReportCheckTypes.FAIL for value in real):
            return ReportCheckTypes.FAIL
        return ReportCheckTypes.PASS
    return ReportCheckTypes.NA


__all__ = ["TIER_ORDER", "apply_target_checks", "build_target_checks", "graded_tier"]


def apply_target_checks(block: Block, config: Any) -> Block:
    """Return ``block`` with ``target_checks`` / ``target_check`` attached.

    Blocks that are not canonical benchmark blocks (prefix-cache,
    spec-decode, GuideLLM — each with its own kind and no sweep-point
    targets) pass through untouched, as do non-record payloads.

    A sweep point with no configured targets is marked ``status="na"``:
    acceptance then counts the block as ungradable instead of grading an
    all-NA ``target_checks`` dict as a pass, so "no targets defined for
    this config" cannot read as "benchmarks passed".
    """
    from .parsers.base import BENCHMARKS_KIND

    if block.kind != BENCHMARKS_KIND or not isinstance(block.data, Mapping):
        return block

    targets = getattr(config, "targets", None) or {}
    data = dict(block.data)
    if not targets:
        logger.warning(
            "No perf targets for sweep point isl=%s osl=%s max_concurrency=%s; "
            "benchmark block is reported as NA (ungraded).",
            getattr(config, "isl", "?"),
            getattr(config, "osl", "?"),
            getattr(config, "max_concurrency", "?"),
        )
        data["status"] = "na"
        data["target_check"] = ReportCheckTypes.NA
        return _replace_data(block, data)

    target_checks, verdict = build_target_checks(targets, block.data)
    if verdict == ReportCheckTypes.NA:
        data["status"] = "na"
    data["target_check"] = verdict
    data["target_checks"] = target_checks
    return _replace_data(block, data, title=_graded_title(block.title, config))


def _graded_title(title: Optional[str], config: Any) -> Optional[str]:
    """Name the sweep point in the title of a graded block.

    Ungraded sweep points keep the tool's shared title so the generator
    collapses them into one sweep table. A graded block cannot join that
    table — its tiered ``target_checks`` is a nested dict, which renders
    as a blob in a multi-row table — so it takes a per-config title and
    renders as its own section, which also keeps its acceptance blocker
    keys distinct from every other sweep point's.
    """
    isl = getattr(config, "isl", None)
    osl = getattr(config, "osl", None)
    concurrency = getattr(config, "max_concurrency", None)
    if isl is None or osl is None or concurrency is None:
        return title
    point = f"ISL {isl} / OSL {osl}, concurrency {concurrency}"
    return f"{title} Targets — {point}" if title else f"Benchmark Targets — {point}"


def _replace_data(
    block: Block, data: Dict[str, Any], title: Optional[str] = None
) -> Block:
    return Block(
        kind=block.kind,
        data=data,
        title=title if title is not None else block.title,
        task_type=block.task_type,
        id=block.id,
        targets=dict(block.targets),
    )


__all__ = ["TIER_ORDER", "apply_target_checks", "build_target_checks"]
