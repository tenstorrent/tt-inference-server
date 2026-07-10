# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .schema import Block, ReportSchema
from .status import TestStatus, glyph_for_label

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


def _status_badge(status: str) -> str:
    """Render a status as ``<emoji> `STATUS```, or just the label if unknown."""
    emoji = glyph_for_label(status)
    return f"{emoji} `{status}`" if emoji else f"`{status}`"


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
    skipped: int = 0
    blockers: Dict[str, str] = field(default_factory=dict)
    # Would-be blockers dropped because a model_spec known_issues waiver matched.
    waived: Dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return self.total - self.failed - self.na - self.skipped - len(self.waived)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "na": self.na,
            "skipped": self.skipped,
            "blockers": dict(self.blockers),
            "waived": dict(self.waived),
        }


def acceptance_criteria_check(
    schema: ReportSchema,
    known_issues: Optional[Iterable[Any]] = None,
) -> Tuple[bool, Dict[str, str], List[CategoryResult]]:
    categories = [
        _check_benchmarks(schema),
        _check_evals(schema, known_issues),
        _check_spec_tests(schema),
    ]
    blockers: Dict[str, str] = {}
    for category in categories:
        blockers.update(category.blockers)
    return len(blockers) == 0, blockers, categories


def _find_waiver(
    known_issues: Optional[Iterable[Any]],
    workflow_name: str,
    task_name: Optional[str],
) -> Optional[str]:
    """Return the reason string of a matching known_issues waiver, else None.

    Accepts both ``workflows.model_spec.KnownIssue`` objects (workflow_type is a
    ``WorkflowType`` enum) and plain dicts (as they appear in the runtime spec
    JSON), so it works regardless of how the spec was deserialized. A waiver with
    ``task_name is None`` matches every task in its workflow.
    """
    if not known_issues:
        return None
    for issue in known_issues:
        wt = getattr(issue, "workflow_type", None)
        if wt is not None:
            wt_name = getattr(wt, "name", None) or str(wt)
            issue_task = getattr(issue, "task_name", None)
            reason = getattr(issue, "reason", "") or ""
        elif isinstance(issue, Mapping):
            wt_name = str(issue.get("workflow_type", ""))
            issue_task = issue.get("task_name")
            reason = issue.get("reason", "") or ""
        else:
            continue
        # Normalize "WorkflowType.EVALS" -> "EVALS".
        wt_name = wt_name.rsplit(".", 1)[-1]
        if wt_name.upper() != workflow_name.upper():
            continue
        if issue_task is None or issue_task == task_name:
            return reason or f"known issue ({workflow_name})"
    return None


ACCEPTANCE_EXPORT_KEYS = (
    "acceptance_criteria",
    "acceptance_blockers",
    "acceptance_criteria_metadata",
    "acceptance_summary_markdown",
)


def build_acceptance_export(
    accepted: bool,
    blockers: Mapping[str, str],
    categories: List[CategoryResult],
    model_status: str | None = None,
) -> Dict[str, Any]:
    return {
        "acceptance_criteria": accepted,
        "acceptance_blockers": dict(blockers),
        "acceptance_criteria_metadata": {
            "enforcement_result": STATUS_PASS if accepted else STATUS_FAIL,
            "model_status": model_status or "",
            "categories": [category.to_dict() for category in categories],
        },
        "acceptance_summary_markdown": format_acceptance_summary_markdown(
            accepted, blockers, categories, model_status
        ),
    }


def format_acceptance_summary_markdown(
    accepted: bool,
    blockers: Mapping[str, str],
    categories: List[CategoryResult],
    model_status: str | None = None,
) -> str:
    lines = [
        "### Acceptance Criteria",
        "",
        f"- Acceptance status: {_status_badge(STATUS_PASS if accepted else STATUS_FAIL)}",
    ]
    if model_status:
        lines.append(f"- Model status: `{model_status}`")
    for category in categories:
        lines.append(
            f"- {category.name}: {_status_badge(category.status)} ({_detail(category)})"
        )

    if not blockers:
        lines.append("- All acceptance criteria passed.")
        return "\n".join(lines)

    lines.append("")
    lines.append("#### Blockers")
    lines.append("")
    lines.extend(_format_blocker_lines(blockers))
    return "\n".join(lines)


def _format_blocker_lines(blockers: Mapping[str, str]) -> List[str]:
    """Render blockers, collapsing entries that share the same message."""
    grouped: Dict[str, List[str]] = {}
    for key, message in blockers.items():
        grouped.setdefault(message, []).append(key)

    lines: List[str] = []
    for message, keys in grouped.items():
        if len(keys) == 1:
            lines.append(f"- `{keys[0]}`: {message}")
            continue
        lines.append(f"- {message} ({len(keys)} blocks)")
        lines.extend(f"  - `{key}`" for key in keys)
    return lines


def _detail(category: CategoryResult) -> str:
    if category.total == 0:
        return "no blocks present"
    parts = [f"{category.passed}/{category.total} passed"]
    if category.failed:
        parts.append(f"{category.failed} failed")
    if category.waived:
        parts.append(f"{len(category.waived)} waived")
    if category.skipped:
        parts.append(f"{category.skipped} skipped")
    if category.na:
        parts.append(f"{category.na} NA")
    return ", ".join(parts)


def _check_benchmarks(schema: ReportSchema) -> CategoryResult:
    benchmark_blocks = [b for b in schema.sections if b.kind == KIND_BENCHMARKS]
    if not benchmark_blocks:
        return CategoryResult(CATEGORY_BENCHMARKS, STATUS_NA, 0, 0)

    blockers: Dict[str, str] = {}
    failed = 0
    skipped = 0
    na = 0
    for block in benchmark_blocks:
        block_key = _block_key(block)

        explicit = _explicit_status(block)
        if explicit is not None:
            if explicit.is_blocking:
                blockers[block_key] = (
                    f"{block.title or block.kind} reported status={explicit.value}"
                )
                failed += 1
            elif explicit is TestStatus.SKIP:
                skipped += 1
            elif explicit is TestStatus.NA:
                na += 1
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
                            _format_benchmark_failure(
                                lvl, check_name, metric, level_checks
                            )
                        )
            if not any_check_seen:
                block_blockers[f"{block_key}.target_checks"] = (
                    "No *_check fields found across any tier."
                )

        if block_blockers:
            failed += 1
            blockers.update(block_blockers)

    non_pass = failed + skipped + na
    if failed:
        status = STATUS_FAIL
    elif non_pass == len(benchmark_blocks) and (skipped or na):
        status = STATUS_NA
    else:
        status = STATUS_PASS
    return CategoryResult(
        CATEGORY_BENCHMARKS,
        status,
        len(benchmark_blocks),
        failed,
        na=na,
        skipped=skipped,
        blockers=blockers,
    )


def _check_evals(
    schema: ReportSchema,
    known_issues: Optional[Iterable[Any]] = None,
) -> CategoryResult:
    eval_blocks = [b for b in schema.sections if b.kind == KIND_EVALS]
    if not eval_blocks:
        return CategoryResult(CATEGORY_EVALS, STATUS_NA, 0, 0)

    blockers: Dict[str, str] = {}
    waived: Dict[str, str] = {}
    failed = 0
    na = 0
    skipped = 0
    for block in eval_blocks:
        block_key = _block_key(block)
        data = block.data if isinstance(block.data, Mapping) else None

        # Determine this block's blocker message (if any). success=False is
        # decisive — a test that self-reported failure is a blocker regardless
        # of any accuracy_check value alongside it.
        message: Optional[str] = None
        explicit = _explicit_status(block)
        if explicit is not None:
            if explicit.is_blocking:
                blockers[block_key] = (
                    f"{block.title or block.kind} reported status={explicit.value} "
                    f"(attempts={data.get('attempts', '?')})"
                )
                failed += 1
            elif explicit is TestStatus.SKIP:
                skipped += 1
            elif explicit is TestStatus.NA:
                na += 1
            continue

        # success=False is decisive — a test that self-reported failure
        # is a blocker regardless of any accuracy_check value alongside it.
        if data is not None and data.get("success") is False:
            message = (
                f"{block.title or block.kind} reported success=False "
                f"(attempts={data.get('attempts', '?')})"
            )
        else:
            check_value = _resolve_nested(block.data, "accuracy_check")
            if check_value is None:
                message = "Missing accuracy_check on eval block."
            else:
                state = _check_state(check_value)
                if state == STATUS_FAIL:
                    message = "Accuracy check failed."
                elif state == STATUS_NA:
                    na += 1

        if message is None:
            continue

        # A model_spec known_issues waiver (workflow_type EVALS, matching
        # task_name) demotes the blocker to a non-fatal waiver — mirroring the
        # v1 acceptance path, which the release workflow otherwise bypasses.
        task_name = data.get("task_name") if data is not None else None
        reason = _find_waiver(known_issues, "EVALS", task_name)
        if reason is not None:
            waived[block_key] = f"{message} (waived: {reason})"
            continue
        blockers[block_key] = message
        failed += 1

    non_pass = failed + na + skipped
    if failed:
        status = STATUS_FAIL
    elif na == len(eval_blocks) or (skipped and non_pass == len(eval_blocks)):
        status = STATUS_NA
    else:
        status = STATUS_PASS
    return CategoryResult(
        CATEGORY_EVALS,
        status,
        len(eval_blocks),
        failed,
        na=na,
        skipped=skipped,
        blockers=blockers,
        waived=waived,
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
    skipped = 0
    na = 0
    for block in spec_blocks:
        test_status = _block_test_status(block)
        if test_status.is_blocking:
            blockers[f"spec.{_block_key(block)}"] = (
                f"{block.title or block.kind} reported status={test_status.value} "
                f"(attempts={block.data.get('attempts', '?')})"
            )
            failed += 1
        elif test_status is TestStatus.SKIP:
            skipped += 1
        elif test_status is TestStatus.NA:
            na += 1

    status = STATUS_FAIL if failed else STATUS_PASS
    return CategoryResult(
        CATEGORY_SPEC_TESTS,
        status,
        len(spec_blocks),
        failed,
        na=na,
        skipped=skipped,
        blockers=blockers,
    )


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def _format_benchmark_failure(
    level_name: str,
    check_name: str,
    metric: str,
    level_checks: Mapping[str, Any],
) -> str:
    threshold = level_checks.get(metric)
    ratio = level_checks.get(f"{metric}_ratio")
    parts = [f"{level_name} {check_name} failed"]
    actual: Any = None
    if isinstance(threshold, (int, float)) and isinstance(ratio, (int, float)):
        actual = threshold * ratio
    if actual is not None:
        parts.append(f"actual {metric}={_format_metric_value(actual)}")
    if threshold is not None and not (
        isinstance(threshold, str) and threshold == "Undefined"
    ):
        parts.append(f"threshold {metric}={_format_metric_value(threshold)}")
    if ratio is not None and not (isinstance(ratio, str) and ratio == "Undefined"):
        parts.append(f"ratio={_format_metric_value(ratio)}")
    return "; ".join(parts) + "."


def _block_key(block: Block) -> str:
    if block.title:
        return f"{block.kind}:{block.title}"
    return block.kind


def _block_test_status(block: Block) -> TestStatus:
    """Resolve a block's :class:`TestStatus`, falling back to legacy fields."""
    data = block.data if isinstance(block.data, Mapping) else {}
    resolved = TestStatus.from_value(data.get("status"))
    if resolved is not None:
        return resolved
    return TestStatus.from_legacy(
        data.get("success"), skipped=bool(data.get("skipped"))
    )


def _explicit_status(block: Block):
    """Return the block's explicit :class:`TestStatus`, or None if unset.

    Unlike :func:`_block_test_status`, this does *not* infer a status from
    legacy fields — evals/benchmarks without a ``status`` keep their existing
    accuracy_check / target_checks grading.
    """
    data = block.data if isinstance(block.data, Mapping) else {}
    return TestStatus.from_value(data.get("status"))


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
    "build_acceptance_export",
    "format_acceptance_summary_markdown",
    "ACCEPTANCE_EXPORT_KEYS",
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
