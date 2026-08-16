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
    fully_waived_task_types,
    build_acceptance_export,
    format_acceptance_summary_markdown,
    task_failure_blockers,
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
        "waived": {},
    }


# --- Task failure blockers ------------------------------------------------


def test_task_failure_blockers_ignores_successful_tasks():
    assert (
        task_failure_blockers([("evaluation", 0, True), ("benchmark", 0, True)]) == {}
    )


def test_task_failure_blockers_flags_crash_with_no_block():
    blockers = task_failure_blockers([("evaluation", 1, False)])
    assert set(blockers) == {"task:evaluation"}
    assert "produced no report block" in blockers["task:evaluation"]
    assert "exit=1" in blockers["task:evaluation"]


def test_task_failure_blockers_flags_failure_with_block():
    blockers = task_failure_blockers([("spec_tests", 1, True)])
    assert "after producing a report block" in blockers["task:spec_tests"]


def test_task_failure_blocker_fails_acceptance_when_category_is_na():
    schema = _schema(
        _bench({"functional": {"ttft_check": 2, "ttft": 100, "ttft_ratio": 0.8}})
    )
    accepted, blockers, _ = acceptance_criteria_check(schema)
    assert accepted is True and blockers == {}

    crash = task_failure_blockers([("evaluation", 1, False)])
    assert crash and (accepted and not crash) is False


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


def test_eval_known_issue_waives_blocker():
    # A failed eval whose task_name matches an EVALS known_issue is demoted to a
    # non-fatal waiver, so acceptance passes. Works with dict-shaped waivers.
    schema = _schema(_eval({"task_name": "longbench_code_e", "accuracy_check": 3}))
    known_issues = [
        {"workflow_type": "EVALS", "task_name": "longbench_code_e", "reason": "flaky"}
    ]
    accepted, blockers, cats = acceptance_criteria_check(schema, known_issues)
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert by_name[CATEGORY_EVALS].status == STATUS_PASS
    assert "evals:E" in by_name[CATEGORY_EVALS].waived


def test_eval_known_issue_wrong_task_still_blocks():
    # Waiver only matches its declared task_name; an unlisted failure blocks.
    schema = _schema(_eval({"task_name": "longbench_single_e", "accuracy_check": 3}))
    known_issues = [
        {"workflow_type": "EVALS", "task_name": "longbench_code_e", "reason": "flaky"}
    ]
    accepted, blockers, _ = acceptance_criteria_check(schema, known_issues)
    assert accepted is False and "evals:E" in blockers


def test_eval_known_issue_wrong_workflow_still_blocks():
    # A BENCHMARKS-scoped waiver must not mask an eval blocker.
    schema = _schema(_eval({"task_name": "longbench_code_e", "accuracy_check": 3}))
    known_issues = [
        {"workflow_type": "BENCHMARKS", "task_name": "longbench_code_e", "reason": "x"}
    ]
    accepted, _, _ = acceptance_criteria_check(schema, known_issues)
    assert accepted is False


def test_eval_all_na_is_na_status_not_failure():
    schema = _schema(_eval({"accuracy_check": 1}))  # 1 == NA tier
    accepted, blockers, cats = acceptance_criteria_check(schema)
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert by_name[CATEGORY_EVALS].status == STATUS_NA


def test_summary_detail_absent_vs_all_na_are_distinguished():
    # No eval block at all -> genuinely "no blocks present".
    absent = CategoryResult(CATEGORY_EVALS, STATUS_NA, total=0, failed=0)
    absent_md = format_acceptance_summary_markdown(True, {}, [absent])
    assert "no blocks present" in absent_md

    # One eval block that ran but self-reported NA accuracy -> NA status with
    # a block present. Must NOT be misreported as "no blocks present".
    present_all_na = CategoryResult(CATEGORY_EVALS, STATUS_NA, total=1, failed=0, na=1)
    present_md = format_acceptance_summary_markdown(True, {}, [present_all_na])
    assert "no blocks present" not in present_md
    assert "0/1 passed" in present_md
    assert "1 NA" in present_md


# --- Model-status-aware tier masking ---------------------------------------


def test_benchmark_failing_check_blocks_at_functional_status():
    # FUNCTIONAL requires the "functional" tier -- a failing functional check
    # still blocks, same as with no status at all.
    schema = _schema(
        _bench({"functional": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.2}})
    )
    accepted, blockers, _ = acceptance_criteria_check(schema, model_status="FUNCTIONAL")
    assert accepted is False
    assert "benchmarks:B.functional.ttft_check" in blockers


def test_benchmark_failing_check_informational_at_experimental_status():
    # EXPERIMENTAL requires no tiers -- the same failing check is masked to a
    # waiver instead of a blocker, and the run is accepted.
    schema = _schema(
        _bench({"target": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.2}})
    )
    accepted, blockers, cats = acceptance_criteria_check(
        schema, model_status="EXPERIMENTAL"
    )
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert "benchmarks:B" in by_name[CATEGORY_BENCHMARKS].waived


def test_benchmark_failing_complete_tier_still_blocks_at_complete_status():
    # COMPLETE requires functional + complete -- a failing "complete" tier
    # check still blocks. The also-failing "target" tier doesn't add a
    # blocker, but is still surfaced (as waived/informational) rather than
    # silently dropped just because the block already failed elsewhere.
    schema = _schema(
        _bench(
            {
                "complete": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.2},
                "target": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.5},
            }
        )
    )
    accepted, blockers, cats = acceptance_criteria_check(
        schema, model_status="COMPLETE"
    )
    by_name = {c.name: c for c in cats}
    assert accepted is False
    assert "benchmarks:B.complete.ttft_check" in blockers
    assert "benchmarks:B.target.ttft_check" not in blockers
    assert "benchmarks:B" in by_name[CATEGORY_BENCHMARKS].waived


def test_benchmark_unrecognized_status_falls_back_to_fully_enforced():
    # A missing/garbled status must never accidentally loosen acceptance.
    schema = _schema(
        _bench({"target": {"ttft_check": 3, "ttft": 100, "ttft_ratio": 1.2}})
    )
    accepted, blockers, _ = acceptance_criteria_check(
        schema, model_status="not_a_real_status"
    )
    assert accepted is False
    assert "benchmarks:B.target.ttft_check" in blockers


def test_eval_accuracy_check_fail_informational_at_experimental_status():
    schema = _schema(_eval({"accuracy_check": 3}))
    accepted, blockers, cats = acceptance_criteria_check(
        schema, model_status="EXPERIMENTAL"
    )
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert "evals:E" in by_name[CATEGORY_EVALS].waived


def test_eval_accuracy_check_fail_still_blocks_at_functional_status():
    schema = _schema(_eval({"accuracy_check": 3}))
    accepted, blockers, _ = acceptance_criteria_check(schema, model_status="FUNCTIONAL")
    assert accepted is False
    assert "Accuracy check failed" in blockers["evals:E"]


def test_eval_explicit_error_informational_at_experimental_status():
    schema = _schema(_eval({"status": "error", "attempts": 2}))
    accepted, blockers, cats = acceptance_criteria_check(
        schema, model_status="EXPERIMENTAL"
    )
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert "evals:E" in by_name[CATEGORY_EVALS].waived


def test_eval_known_issue_waiver_takes_precedence_over_status_message():
    # Both an EXPERIMENTAL status and a known_issues waiver would mask this --
    # the known_issue reason should still be the one recorded.
    schema = _schema(_eval({"task_name": "ifeval", "accuracy_check": 3}))
    known_issues = [
        {"workflow_type": "EVALS", "task_name": "ifeval", "reason": "tracked in #4733"}
    ]
    accepted, blockers, cats = acceptance_criteria_check(
        schema, known_issues=known_issues, model_status="EXPERIMENTAL"
    )
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert "waived: tracked in #4733" in by_name[CATEGORY_EVALS].waived["evals:E"]


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
    assert "Acceptance status: ✅ `PASS`" in md
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
    assert "Acceptance status: ✅ `PASS`" in md
    assert "Model status: `COMPLETE`" in md


def test_summary_markdown_lists_blockers():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_FAIL, total=1, failed=1)]
    md = format_acceptance_summary_markdown(
        False, {"benchmarks:B.target.ttft_check": "ttft too slow"}, categories
    )
    assert "Acceptance status: ❌ `FAIL`" in md
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
    assert "Acceptance status: ✅ `PASS`" in export["acceptance_summary_markdown"]


def test_build_acceptance_export_failure_defaults_model_status():
    categories = [CategoryResult(CATEGORY_BENCHMARKS, STATUS_FAIL, total=1, failed=1)]
    export = build_acceptance_export(False, {"benchmarks:B": "bad"}, categories)
    assert export["acceptance_criteria"] is False
    assert export["acceptance_blockers"] == {"benchmarks:B": "bad"}
    assert export["acceptance_criteria_metadata"]["enforcement_result"] == "FAIL"
    assert export["acceptance_criteria_metadata"]["model_status"] == ""


def _spec_suite(failing, passing=(), status="fail"):
    """A suite-style spec block with per-case verdicts (VLLMParamConformanceTest shape)."""
    summary = [{"test_case": c, "status": "❌ FAIL"} for c in failing]
    summary += [{"test_case": c, "status": "✅ PASS"} for c in passing]
    return _spec(
        status,
        attempts=1,
        test_name="VLLMParamConformanceTest",
        parameter_conformance_summary=summary,
    )


def _waiver(task_name, workflow="SPEC_TESTS"):
    return [{"workflow_type": workflow, "task_name": task_name, "reason": "known"}]


def test_spec_known_issue_waives_by_case_name():
    # Waiver naming an individual pytest function demotes the block, matching the
    # per-case granularity the v1 acceptance path had.
    schema = _schema(_spec_suite(["test_penalties"], passing=["test_stop"]))
    accepted, blockers, cats = acceptance_criteria_check(
        schema, _waiver("test_penalties")
    )
    by_name = {c.name: c for c in cats}
    assert accepted is True and blockers == {}
    assert by_name[CATEGORY_SPEC_TESTS].status == STATUS_PASS
    assert "spec.spec_tests:T" in by_name[CATEGORY_SPEC_TESTS].waived


def test_spec_known_issue_waives_by_suite_name():
    schema = _schema(_spec_suite(["test_n"]))
    accepted, _, _ = acceptance_criteria_check(
        schema, _waiver("VLLMParamConformanceTest")
    )
    assert accepted is True


def test_spec_known_issue_partial_coverage_still_blocks():
    # test_n is waived but test_logprobs is not, so the block must still fail —
    # a waiver must never mask an unlisted regression in the same suite.
    schema = _schema(_spec_suite(["test_n", "test_logprobs"]))
    accepted, blockers, _ = acceptance_criteria_check(schema, _waiver("test_n"))
    assert accepted is False and "spec.spec_tests:T" in blockers


def test_spec_known_issue_wrong_workflow_still_blocks():
    schema = _schema(_spec_suite(["test_n"]))
    accepted, _, _ = acceptance_criteria_check(
        schema, _waiver("test_n", workflow="EVALS")
    )
    assert accepted is False


def test_spec_known_issue_workflow_wide_waiver_applies():
    # task_name=None matches every task in the workflow.
    schema = _schema(_spec_suite(["test_n"]))
    accepted, _, _ = acceptance_criteria_check(schema, _waiver(None))
    assert accepted is True


def test_spec_block_without_case_summary_needs_suite_level_waiver():
    # No itemised cases (e.g. success:false blocks), so a case-level waiver can't
    # be evaluated and must not silently pass.
    schema = _schema(_spec("fail", attempts=1, test_name="OtherTest"))
    accepted, _, _ = acceptance_criteria_check(schema, _waiver("test_n"))
    assert accepted is False
    accepted, _, _ = acceptance_criteria_check(schema, _waiver("OtherTest"))
    assert accepted is True


# --- task-level exit-code blocker vs waivers ------------------------------


def test_waived_spec_task_does_not_block_on_exit_code():
    # A waived suite still exits non-zero -- that is exactly what the waiver
    # covers -- so the task-level blocker must not re-block the run.
    schema = _schema(_spec_suite(["test_penalties"]))
    waivers = _waiver("test_penalties")
    acceptance_criteria_check(schema, waivers)
    waived_types = fully_waived_task_types(schema, waivers)
    assert "spec_tests" in waived_types
    blockers = task_failure_blockers(
        [("spec_tests", 1, True)], waived_task_types=waived_types
    )
    assert blockers == {}


def test_unwaived_spec_task_still_blocks_on_exit_code():
    schema = _schema(_spec_suite(["test_n"]))
    blockers = task_failure_blockers(
        [("spec_tests", 1, True)],
        waived_task_types=fully_waived_task_types(schema, _waiver("test_penalties")),
    )
    assert "task:spec_tests" in blockers


def test_crash_without_block_blocks_even_when_waived():
    # No report block means the suite never ran to completion; a waiver must
    # not launder that into a PASS.
    schema = _schema(_spec_suite(["test_penalties"]))
    blockers = task_failure_blockers(
        [("spec_tests", 1, False)],
        waived_task_types=fully_waived_task_types(schema, _waiver("test_penalties")),
    )
    assert "task:spec_tests" in blockers
    assert "produced no report block" in blockers["task:spec_tests"]


def test_fully_waived_task_types_empty_when_category_has_blockers():
    # Partial coverage: one case waived, one not -> category still blocks, so
    # the task exit code must keep blocking too.
    schema = _schema(_spec_suite(["test_n", "test_logprobs"]))
    assert fully_waived_task_types(schema, _waiver("test_n")) == set()


def test_status_tier_masked_evals_do_not_excuse_a_crashed_eval_task():
    # #4830 masks eval failures for EXPERIMENTAL into the same `waived` bucket
    # an explicit waiver uses. An eval task only exits non-zero when its runner
    # raised, so that must still block -- masking a score is not a licence to
    # ignore a crash.
    schema = _schema(_eval({"task_name": "ifeval", "accuracy_check": 3}))
    accepted, _, _ = acceptance_criteria_check(schema, None, "EXPERIMENTAL")
    assert accepted is True
    assert fully_waived_task_types(schema, None) == set()
    blockers = task_failure_blockers(
        [("evaluation", 1, True)],
        waived_task_types=fully_waived_task_types(schema, None),
    )
    assert "task:evaluation" in blockers


def _infra_spec(test_name, task_type="unit", status="fail"):
    """A spec block whose task_type INFRA_TASK_TYPES hides from the category."""
    return Block(
        kind="spec_tests",
        title=test_name,
        task_type=task_type,
        data={
            "success": False,
            "status": status,
            "attempts": 1,
            "test_name": test_name,
        },
    )


def test_unwaived_infra_spec_failure_keeps_exit_code_blocker():
    # LoggerForkSafetyTest (TASK_TYPE="unit") is a prerequisite in every suite and
    # is invisible to the Spec Tests category, but its failure is in the task's
    # exit code. A waiver on a sibling functional case must not excuse it.
    schema = _schema(
        _spec_suite(["test_penalties"]), _infra_spec("LoggerForkSafetyTest")
    )
    waivers = _waiver("test_penalties")
    accepted, blockers, _ = acceptance_criteria_check(schema, waivers)
    assert accepted is True and blockers == {}  # category cannot see the infra block
    assert fully_waived_task_types(schema, waivers) == set()
    assert "task:spec_tests" in task_failure_blockers(
        [("spec_tests", 1, True)],
        waived_task_types=fully_waived_task_types(schema, waivers),
    )


def test_waived_infra_spec_failure_is_exempt():
    # An infra failure named by its own waiver is covered like any other.
    schema = _schema(
        _spec_suite(["test_penalties"]), _infra_spec("LoggerForkSafetyTest")
    )
    waivers = _waiver("test_penalties") + _waiver("LoggerForkSafetyTest")
    assert fully_waived_task_types(schema, waivers) == {"spec_tests"}


def test_passing_infra_spec_block_does_not_defeat_the_exemption():
    schema = _schema(
        _spec_suite(["test_penalties"]),
        _infra_spec("LoggerForkSafetyTest", status="pass"),
    )
    waivers = _waiver("test_penalties")
    assert fully_waived_task_types(schema, waivers) == {"spec_tests"}
