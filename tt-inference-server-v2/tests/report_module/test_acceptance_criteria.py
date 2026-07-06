# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``report_module.acceptance_criteria`` routing + blocker logic.

Benchmark blocks keep the nested ``target_checks[level][*_check]`` layout the
runners emit; these tests assert against that shape directly.
"""

from __future__ import annotations

from report_module.acceptance_criteria import (
    CATEGORY_BENCHMARKS,
    CATEGORY_EVALS,
    CATEGORY_SPEC_TESTS,
    STATUS_FAIL,
    STATUS_NA,
    STATUS_PASS,
    CategoryResult,
    acceptance_criteria_check,
    build_acceptance_export,
    format_acceptance_summary_markdown,
)
from report_module.schema import Block, ReportSchema


def _schema(*blocks: Block) -> ReportSchema:
    return ReportSchema(metadata={"report_id": "r"}, sections=list(blocks))


def _categories_by_name(schema: ReportSchema):
    _, _, categories = acceptance_criteria_check(schema)
    return {c.name: c for c in categories}


# --- CategoryResult -------------------------------------------------------


def test_category_result_passed_and_to_dict():
    cat = CategoryResult("Benchmarks", STATUS_FAIL, total=5, failed=2, na=1, skipped=1)
    assert cat.passed == 1
    assert cat.to_dict() == {
        "name": "Benchmarks",
        "status": STATUS_FAIL,
        "total": 5,
        "passed": 1,
        "failed": 2,
        "na": 1,
        "skipped": 1,
        "blockers": {},
    }


# --- Benchmarks -----------------------------------------------------------


def _bench(target_checks) -> Block:
    return Block(kind="benchmarks", title="B", data={"target_checks": target_checks})


def test_benchmarks_absent_is_na():
    cat = _categories_by_name(_schema(Block(kind="evals", data={})))[
        CATEGORY_BENCHMARKS
    ]
    assert cat.status == STATUS_NA and cat.total == 0


def test_benchmark_passing_tier_accepts():
    schema = _schema(
        _bench({"target": {"ttft_check": 2, "ttft": 100, "ttft_ratio": 0.8}})
    )
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is True and blockers == {}


def test_benchmark_failing_check_blocks():
    schema = _schema(
        _bench({"target": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.2}})
    )
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is False
    assert "benchmarks:B.target.ttft_check" in blockers
    # The message surfaces the failed tier + metric.
    assert "ttft" in blockers["benchmarks:B.target.ttft_check"]


def test_benchmark_missing_target_checks_blocks():
    schema = _schema(Block(kind="benchmarks", title="B", data={}))
    _, blockers, _ = acceptance_criteria_check(schema)
    assert "benchmarks:B.target_checks" in blockers
    assert "Missing target_checks" in blockers["benchmarks:B.target_checks"]


def test_benchmark_no_check_fields_blocks():
    schema = _schema(_bench({"target": {"ttft": 100}}))
    _, blockers, _ = acceptance_criteria_check(schema)
    assert "No *_check fields" in blockers["benchmarks:B.target_checks"]


# --- Evals ----------------------------------------------------------------


def _eval(data) -> Block:
    return Block(kind="evals", title="E", data=data)


def test_eval_success_false_blocks_even_with_passing_accuracy():
    schema = _schema(_eval({"success": False, "attempts": 2, "accuracy_check": 2}))
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is False
    assert "attempts=2" in blockers["evals:E"]


def test_eval_missing_accuracy_check_blocks():
    _, blockers, _ = acceptance_criteria_check(_schema(_eval({"score": 0.9})))
    assert blockers["evals:E"] == "Missing accuracy_check on eval block."


def test_eval_accuracy_check_pass():
    accepted, blockers, _ = acceptance_criteria_check(
        _schema(_eval({"accuracy_check": 2}))
    )
    assert accepted is True and blockers == {}


def test_eval_accuracy_check_fail():
    _, blockers, _ = acceptance_criteria_check(_schema(_eval({"accuracy_check": 3})))
    assert "Accuracy check failed" in blockers["evals:E"]


def test_eval_all_na_is_na_status_not_failure():
    schema = _schema(_eval({"accuracy_check": 1}))  # 1 == NA tier
    accepted, blockers, cats = acceptance_criteria_check(schema)
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert by_name[CATEGORY_EVALS].status == STATUS_NA


# --- Spec tests -----------------------------------------------------------


def test_spec_tests_infra_task_types_excluded():
    schema = _schema(
        Block(kind="spec_tests", task_type="health", data={"success": False})
    )
    cat = _categories_by_name(schema)[CATEGORY_SPEC_TESTS]
    assert cat.status == STATUS_NA and cat.total == 0


def test_spec_tests_success_false_blocks():
    schema = _schema(
        Block(
            kind="spec_tests",
            title="T",
            task_type="functional",
            data={"success": False, "attempts": 3},
        )
    )
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is False
    assert "spec.spec_tests:T" in blockers


def test_spec_tests_success_true_passes():
    schema = _schema(
        Block(kind="spec_tests", task_type="functional", data={"success": True})
    )
    accepted, _, cats = acceptance_criteria_check(schema)
    by_name = {c.name: c for c in cats}
    assert accepted is True
    assert by_name[CATEGORY_SPEC_TESTS].status == STATUS_PASS


# --- Spec tests: status-aware (SKIP / ERROR / NA) -------------------------


def _spec(status_value: str, **extra) -> Block:
    data = {"success": status_value == "pass", "status": status_value, **extra}
    return Block(kind="spec_tests", title="T", task_type="functional", data=data)


def test_spec_skip_is_non_blocking():
    schema = _schema(_spec("skip", skipped=True, reason="no board"))
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_SPEC_TESTS]
    assert accepted is True and blockers == {}
    assert cat.skipped == 1 and cat.failed == 0 and cat.status == STATUS_PASS


def test_spec_na_is_non_blocking():
    schema = _schema(_spec("na"))
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_SPEC_TESTS]
    assert accepted is True and blockers == {}
    assert cat.na == 1 and cat.failed == 0


def test_spec_error_blocks():
    schema = _schema(_spec("error", error={"type": "AttributeError", "message": "x"}))
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_SPEC_TESTS]
    assert accepted is False
    assert "spec.spec_tests:T" in blockers
    assert "status=error" in blockers["spec.spec_tests:T"]
    assert cat.failed == 1


def test_spec_status_takes_precedence_over_success_flag():
    # success flag says failure, but explicit SKIP status must win (non-blocking).
    block = Block(
        kind="spec_tests",
        title="T",
        task_type="functional",
        data={"success": False, "status": "skip", "reason": "gated"},
    )
    accepted, blockers, _ = acceptance_criteria_check(_schema(block))
    assert accepted is True and blockers == {}


def test_mixed_spec_statuses_counts():
    schema = _schema(
        _spec("pass"),
        _spec("skip", skipped=True, reason="r"),
        _spec("na"),
    )
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_SPEC_TESTS]
    assert accepted is True
    assert cat.total == 3 and cat.passed == 1 and cat.skipped == 1 and cat.na == 1


# --- Evals: explicit status overrides accuracy heuristics -----------------


def test_eval_explicit_skip_is_non_blocking():
    schema = _schema(_eval({"success": False, "status": "skip", "reason": "gated"}))
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_EVALS]
    assert accepted is True and blockers == {}
    assert cat.skipped == 1


def test_eval_explicit_error_blocks():
    schema = _schema(_eval({"success": False, "status": "error"}))
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is False
    assert "status=error" in blockers["evals:E"]


# --- Benchmarks: status short-circuits target_checks ----------------------


def test_benchmark_skip_is_non_blocking_without_target_checks():
    # A skipped benchmark has no target_checks; it must NOT trip the
    # "Missing target_checks" blocker.
    schema = _schema(
        Block(kind="benchmarks", title="B", data={"status": "skip", "reason": "gated"})
    )
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_BENCHMARKS]
    assert accepted is True and blockers == {}
    assert cat.skipped == 1 and cat.status == STATUS_NA


def test_benchmark_error_blocks():
    schema = _schema(Block(kind="benchmarks", title="B", data={"status": "error"}))
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is False
    assert "status=error" in blockers["benchmarks:B"]


def test_benchmark_passing_target_checks_still_pass_with_status_absent():
    schema = _schema(
        _bench({"target": {"ttft_check": 2, "ttft": 100, "ttft_ratio": 0.8}})
    )
    accepted, blockers, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_BENCHMARKS]
    assert accepted is True and blockers == {}
    assert cat.status == STATUS_PASS and cat.skipped == 0


def test_benchmark_mixed_skip_and_pass_is_pass():
    schema = _schema(
        _bench({"target": {"ttft_check": 2, "ttft": 100, "ttft_ratio": 0.8}}),
        Block(kind="benchmarks", title="B2", data={"status": "skip", "reason": "x"}),
    )
    accepted, _, cats = acceptance_criteria_check(schema)
    cat = {c.name: c for c in cats}[CATEGORY_BENCHMARKS]
    assert accepted is True
    assert cat.total == 2 and cat.passed == 1 and cat.skipped == 1
    assert cat.status == STATUS_PASS


# --- Markdown summary -----------------------------------------------------


def test_summary_markdown_passing():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_PASS, total=2, failed=0)]
    md = format_acceptance_summary_markdown(True, {}, categories)
    assert "Acceptance status: `PASS`" in md
    assert "All acceptance criteria passed." in md
    assert "2/2 passed" in md


def test_summary_markdown_detail_shows_skipped():
    categories = [
        CategoryResult(CATEGORY_SPEC_TESTS, STATUS_PASS, total=3, failed=0, skipped=1)
    ]
    md = format_acceptance_summary_markdown(True, {}, categories)
    assert "2/3 passed" in md
    assert "1 skipped" in md


def test_summary_markdown_includes_model_status():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_PASS, total=1, failed=0)]
    md = format_acceptance_summary_markdown(True, {}, categories, "COMPLETE")
    assert "Acceptance status: `PASS`" in md
    assert "Model status: `COMPLETE`" in md


def test_summary_markdown_lists_blockers():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_FAIL, total=1, failed=1)]
    md = format_acceptance_summary_markdown(
        False, {"benchmarks:B.target.ttft_check": "ttft too slow"}, categories
    )
    assert "Acceptance status: `FAIL`" in md
    assert "#### Blockers" in md
    assert "`benchmarks:B.target.ttft_check`: ttft too slow" in md


def test_build_acceptance_export_shape():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_PASS, total=1, failed=0)]
    export = build_acceptance_export(True, {}, categories, "COMPLETE")
    assert export["acceptance_criteria"] is True
    assert export["acceptance_blockers"] == {}
    metadata = export["acceptance_criteria_metadata"]
    assert metadata["enforcement_result"] == "PASS"
    assert metadata["model_status"] == "COMPLETE"
    assert metadata["categories"][0]["name"] == CATEGORY_BENCHMARKS
    assert "Acceptance status: `PASS`" in export["acceptance_summary_markdown"]


def test_build_acceptance_export_failure_defaults_model_status():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_FAIL, total=1, failed=1)]
    export = build_acceptance_export(False, {"benchmarks:B": "bad"}, categories)
    assert export["acceptance_criteria"] is False
    assert export["acceptance_blockers"] == {"benchmarks:B": "bad"}
    assert export["acceptance_criteria_metadata"]["enforcement_result"] == "FAIL"
    assert export["acceptance_criteria_metadata"]["model_status"] == ""
