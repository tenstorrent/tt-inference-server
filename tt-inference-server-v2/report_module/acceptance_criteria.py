# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

from .schema import Block, ReportSchema

# Three canonical kinds emitted by every runner. Acceptance routes by
# this field alone — no substring matching, no frozensets, no regex.
KIND_BENCHMARKS = "benchmarks"
KIND_EVALS = "evals"
KIND_SPEC_TESTS = "spec_tests"

TARGET_LEVELS = ("functional", "complete", "target")
CHECK_SUFFIX = "_check"
FAIL_CHECK_INT = 3

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NA = "NA"

CATEGORY_BENCHMARKS = "Benchmarks"
CATEGORY_EVALS = "Evals"
CATEGORY_SPEC_TESTS = "Spec Tests"

INFRA_TASK_TYPES = frozenset({"health", "infra", "unit", "stability", "integration"})


@dataclass(frozen=True)
class CategoryResult:
    name: str
    status: str
    total: int
    failed: int
    na: int = 0
    blockers: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return self.total - self.failed - self.na

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "na": self.na,
            "blockers": dict(self.blockers),
        }


def acceptance_criteria_check(
    schema: ReportSchema,
) -> Tuple[bool, Dict[str, str], List[CategoryResult]]:
    categories = [
        _check_benchmarks(schema),
        _check_evals(schema),
        _check_spec_tests(schema),
    ]
    blockers: Dict[str, str] = {}
    for category in categories:
        blockers.update(category.blockers)
    return len(blockers) == 0, blockers, categories


def format_acceptance_summary_markdown(
    accepted: bool,
    blockers: Mapping[str, str],
    categories: List[CategoryResult],
) -> str:
    lines = [
        "### Acceptance Criteria",
        "",
        f"- Overall status: `{STATUS_PASS if accepted else STATUS_FAIL}`",
    ]
    for category in categories:
        lines.append(f"- {category.name}: `{category.status}` ({_detail(category)})")

    if not blockers:
        lines.append("- All acceptance criteria passed.")
        return "\n".join(lines)

    lines.append("")
    lines.append("#### Blockers")
    lines.append("")
    for key, message in blockers.items():
        lines.append(f"- `{key}`: {message}")
    return "\n".join(lines)


def _detail(category: CategoryResult) -> str:
    if category.status == STATUS_NA:
        return "no blocks present"
    parts = [f"{category.passed}/{category.total} passed"]
    if category.failed:
        parts.append(f"{category.failed} failed")
    if category.na:
        parts.append(f"{category.na} NA")
    return ", ".join(parts)


def _check_benchmarks(schema: ReportSchema) -> CategoryResult:
    benchmark_blocks = [b for b in schema.sections if b.kind == KIND_BENCHMARKS]
    if not benchmark_blocks:
        return CategoryResult(CATEGORY_BENCHMARKS, STATUS_NA, 0, 0)

    blockers: Dict[str, str] = {}
    failed = 0
    for block in benchmark_blocks:
        block_key = _block_key(block)
        data = block.data if isinstance(block.data, Mapping) else None

        success_value = _resolve_nested(block.data, "success")
        attempts_value = _resolve_nested(block.data, "attempts")
        if success_value is False:
            blockers[block_key] = (
                f"{block.title or block.kind} reported success=False "
                f"(attempts={attempts_value if attempts_value is not None else '?'})"
            )
            failed += 1
            continue

        block_blockers: Dict[str, str] = {}
        target_checks = _resolve_nested(block.data, "target_checks")
        if not isinstance(target_checks, Mapping):
            block_blockers[f"{block_key}.target_checks"] = (
                "Missing target_checks in benchmark block."
            )
        elif any(
            isinstance(target_checks.get(lvl), Mapping)
            and _level_passes(target_checks[lvl])
            for lvl in TARGET_LEVELS
        ):
            pass
        else:
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
                        block_blockers[f"{block_key}.{lvl}.{check_name}"] = (
                            f"{lvl} {metric} failed (check={check_value!r})"
                        )
            if not any_check_seen:
                block_blockers[f"{block_key}.target_checks"] = (
                    "No *_check fields found across any tier."
                )

        if block_blockers:
            failed += 1
            blockers.update(block_blockers)

    status = STATUS_FAIL if failed else STATUS_PASS
    return CategoryResult(
        CATEGORY_BENCHMARKS,
        status,
        len(benchmark_blocks),
        failed,
        blockers=blockers,
    )


def _check_evals(schema: ReportSchema) -> CategoryResult:
    eval_blocks = [b for b in schema.sections if b.kind == KIND_EVALS]
    if not eval_blocks:
        return CategoryResult(CATEGORY_EVALS, STATUS_NA, 0, 0)

    blockers: Dict[str, str] = {}
    failed = 0
    na = 0
    for block in eval_blocks:
        block_key = _block_key(block)
        data = block.data if isinstance(block.data, Mapping) else None

        # success=False is decisive — a test that self-reported failure
        # is a blocker regardless of any accuracy_check value alongside it.
        if data is not None and data.get("success") is False:
            blockers[block_key] = (
                f"{block.title or block.kind} reported success=False "
                f"(attempts={data.get('attempts', '?')})"
            )
            failed += 1
            continue

        check_value = _resolve_nested(block.data, "accuracy_check")
        if check_value is None:
            blockers[block_key] = "Missing accuracy_check on eval block."
            failed += 1
            continue

        state = _check_state(check_value)
        if state == STATUS_FAIL:
            blockers[block_key] = f"Accuracy check failed (value={check_value!r})"
            failed += 1
        elif state == STATUS_NA:
            na += 1

    if failed:
        status = STATUS_FAIL
    elif na == len(eval_blocks):
        status = STATUS_NA
    else:
        status = STATUS_PASS
    return CategoryResult(
        CATEGORY_EVALS,
        status,
        len(eval_blocks),
        failed,
        na=na,
        blockers=blockers,
    )


def _check_spec_tests(schema: ReportSchema) -> CategoryResult:
    spec_blocks = [
        b
        for b in schema.sections
        if b.kind == KIND_SPEC_TESTS
        and (b.task_type or "") not in INFRA_TASK_TYPES
        and isinstance(b.data, Mapping)
    ]
    if not spec_blocks:
        return CategoryResult(CATEGORY_SPEC_TESTS, STATUS_NA, 0, 0)

    blockers: Dict[str, str] = {}
    failed = 0
    for block in spec_blocks:
        success = block.data.get("success")
        if success is False:
            blockers[f"spec.{_block_key(block)}"] = (
                f"{block.title or block.kind} reported success=False "
                f"(attempts={block.data.get('attempts', '?')})"
            )
            failed += 1

    status = STATUS_FAIL if failed else STATUS_PASS
    return CategoryResult(
        CATEGORY_SPEC_TESTS,
        status,
        len(spec_blocks),
        failed,
        blockers=blockers,
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
    if isinstance(check_value, bool):
        return check_value
    try:
        return int(check_value) != FAIL_CHECK_INT
    except (TypeError, ValueError):
        return str(check_value).strip().upper() not in {"FAIL", "FAILED", "NA"}


def _check_state(check_value: Any) -> str:
    """Map a raw accuracy_check value to one of STATUS_PASS / FAIL / NA.

    Honors the ReportCheckTypes IntEnum convention (NA=1, PASS=2, FAIL=3)
    and the string spellings used in legacy data.
    """
    if check_value is None:
        return STATUS_NA
    if isinstance(check_value, bool):
        return STATUS_PASS if check_value else STATUS_FAIL
    try:
        as_int = int(check_value)
    except (TypeError, ValueError):
        text = str(check_value).strip().upper()
        if text in {"FAIL", "FAILED"}:
            return STATUS_FAIL
        if text in {"NA", "N/A"}:
            return STATUS_NA
        return STATUS_PASS
    if as_int == FAIL_CHECK_INT:
        return STATUS_FAIL
    if as_int == 1:
        return STATUS_NA
    return STATUS_PASS


__all__ = [
    "acceptance_criteria_check",
    "format_acceptance_summary_markdown",
    "CategoryResult",
    "KIND_BENCHMARKS",
    "KIND_EVALS",
    "KIND_SPEC_TESTS",
    "STATUS_PASS",
    "STATUS_FAIL",
    "STATUS_NA",
    "CATEGORY_BENCHMARKS",
    "CATEGORY_EVALS",
    "CATEGORY_SPEC_TESTS",
    "INFRA_TASK_TYPES",
]
