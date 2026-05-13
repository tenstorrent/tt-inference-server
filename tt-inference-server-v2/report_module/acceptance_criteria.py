# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .schema import Block, ReportSchema

BENCHMARK_KINDS = frozenset(
    {
        "image_benchmark",
        "cnn_benchmark",
        "audio_benchmark",
        "tts_benchmark",
        "video_benchmark",
        "embedding_benchmark",
    }
)
EVAL_KINDS = frozenset(
    {
        "image_eval",
        "cnn_eval",
        "audio_eval",
        "tts_eval",
        "video_eval",
        "embedding_eval",
    }
)
HEALTH_INFRA_KINDS = frozenset(
    {
        "device_liveness",
        "media_server_liveness",
        "logger_fork_safety",
    }
)
TARGET_LEVELS = ("functional", "complete", "target")
CHECK_SUFFIX = "_check"
FAIL_CHECK_INT = 3


def acceptance_criteria_check(
    schema: ReportSchema,
) -> Tuple[bool, Dict[str, str]]:
    blockers: Dict[str, str] = {}

    _check_benchmarks(schema, blockers)
    _check_evals(schema, blockers)
    _check_spec_tests(schema, blockers)

    return len(blockers) == 0, blockers


def format_acceptance_summary_markdown(
    accepted: bool, blockers: Mapping[str, str]
) -> str:
    lines = [
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: `{'PASS' if accepted else 'FAIL'}`",
    ]
    if accepted:
        lines.append("- All acceptance criteria passed.")
        return "\n".join(lines)
    for key, message in blockers.items():
        lines.append(f"- `{key}`: {message}")
    return "\n".join(lines)


def _check_benchmarks(schema: ReportSchema, blockers: Dict[str, str]) -> None:
    benchmark_blocks = [b for b in schema.sections if b.kind in BENCHMARK_KINDS]
    if not benchmark_blocks:
        return

    for block in benchmark_blocks:
        block_key = _block_key(block)
        target_checks = _resolve_nested(block.data, "target_checks")
        if not isinstance(target_checks, Mapping):
            blockers[f"{block_key}.target_checks"] = (
                "Missing target_checks in benchmark block."
            )
            continue

        if any(
            isinstance(target_checks.get(lvl), Mapping)
            and _level_passes(target_checks[lvl])
            for lvl in TARGET_LEVELS
        ):
            continue

        any_check_seen = False
        for lvl in TARGET_LEVELS:
            level_checks = target_checks.get(lvl)
            if not isinstance(level_checks, Mapping):
                continue
            for check_name, check_value in level_checks.items():
                if not check_name.endswith(CHECK_SUFFIX):
                    continue
                any_check_seen = True
                if not _passes_check(check_value):
                    metric = check_name[: -len(CHECK_SUFFIX)]
                    blockers[f"{block_key}.{lvl}.{check_name}"] = (
                        f"{lvl} {metric} failed (check={check_value!r})"
                    )
        if not any_check_seen:
            blockers[f"{block_key}.target_checks"] = (
                "No *_check fields found across any tier."
            )


def _check_evals(schema: ReportSchema, blockers: Dict[str, str]) -> None:
    eval_blocks = [b for b in schema.sections if b.kind in EVAL_KINDS]
    if not eval_blocks:
        return

    for block in eval_blocks:
        block_key = _block_key(block)
        check_value = _resolve_nested(block.data, "accuracy_check")
        if check_value is None:
            blockers[block_key] = "Missing accuracy_check on eval block."
            continue
        if not _passes_check(check_value):
            blockers[block_key] = f"Accuracy check failed (value={check_value!r})"


def _check_spec_tests(schema: ReportSchema, blockers: Dict[str, str]) -> None:
    excluded = BENCHMARK_KINDS | EVAL_KINDS | HEALTH_INFRA_KINDS
    for block in schema.sections:
        if block.kind in excluded:
            continue
        if not isinstance(block.data, Mapping):
            continue
        success = block.data.get("success")
        if success is False:
            blockers[f"spec.{_block_key(block)}"] = (
                f"{block.title or block.kind} reported success=False "
                f"(attempts={block.data.get('attempts', '?')})"
            )


def _block_key(block: Block) -> str:
    if block.title:
        return f"{block.kind}:{block.title}"
    return block.kind


def _level_passes(level_checks: Mapping[str, Any]) -> bool:
    check_values = [
        v for name, v in level_checks.items() if name.endswith(CHECK_SUFFIX)
    ]
    return bool(check_values) and all(_passes_check(v) for v in check_values)


def _resolve_nested(data: Any, key: str) -> Any:
    if not isinstance(data, Mapping):
        return None
    if key in data:
        return data[key]
    for value in data.values():
        if isinstance(value, Mapping) and key in value:
            return value[key]
    return None


def _passes_check(check_value: Any) -> bool:
    if check_value is None:
        return False
    try:
        return int(check_value) != FAIL_CHECK_INT
    except (TypeError, ValueError):
        return str(check_value).strip().upper() not in {"FAIL", "FAILED", "NA"}


__all__ = [
    "acceptance_criteria_check",
    "format_acceptance_summary_markdown",
    "BENCHMARK_KINDS",
    "EVAL_KINDS",
]
