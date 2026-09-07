# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for ``workflow_module.exabox_merged_report`` filtering + merging."""

from __future__ import annotations

import json
from pathlib import Path

from workflow_module.exabox_merged_report import (
    SMOKE_FLAG,
    _strip_server_url,
    build_merged_schema,
    deduplicate,
    discover_test_reports,
    load_test_reports,
    merge_reports,
    mergeable,
    resolve_model_status,
)

SERVER = "https://gemma4-120c-ngrok.n.cloud.tenstorrent.com:443"


def _report(
    workflow="evals",
    *,
    task="r1_gpqa_diamond",
    accuracy_check=2,
    score=83.0,
    model_status="FUNCTIONAL",
    args="--dev-mode",
    server=SERVER,
    generated_at="2026-08-31 04:44:28",
):
    """A per-test report shaped like the ones exabox uploads.

    Acceptance fields sit at the TOP level, not under metadata — the generator
    hoists them out on the way to disk, so that is how they appear on a real
    report read back from an artifact.
    """
    return {
        "metadata": {
            "report_id": f"m_{workflow}",
            "model_name": "google/gemma-4-31B-it",
            "device": "SUPER_CLUSTER",
            "workflow": workflow,
            "generated_at": generated_at,
            "run_command": (
                f"python run.py --model gemma-4-31B-it --workflow {workflow} "
                f"--device super_cluster --server-url {server} "
                f"--skip-system-sw-validation {args}"
            ),
        },
        "sections": [
            {
                "kind": "evals",
                "title": f"LLM Eval — {task}",
                "task_type": "llm",
                "id": "gemma-4-31B-it_SUPER_CLUSTER",
                "targets": {"task_name": task},
                "data": {
                    "task_name": task,
                    "tolerance": 0.05,
                    "score": score,
                    "accuracy_check": accuracy_check,
                },
            }
        ],
        "acceptance_criteria": accuracy_check != 3,
        "acceptance_blockers": {},
        "acceptance_criteria_metadata": {
            "enforcement_result": "PASS" if accuracy_check != 3 else "FAIL",
            "model_status": model_status,
            "categories": [
                {
                    "name": "Evals",
                    "status": "PASS",
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "na": 0,
                    "skipped": 0,
                    "blockers": {},
                    "waived": {},
                }
            ],
        },
    }


def _write(container: Path, artifact: str, filename: str, payload) -> Path:
    path = container / artifact / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# --------------------------------------------------------------------------- #
# inclusion is by shape, never by test name
# --------------------------------------------------------------------------- #
def test_mergeable_accepts_a_normal_report():
    assert mergeable(_report()) == (True, "included")


def test_sanity_is_skipped_for_having_no_sections():
    sanity = {"report_type": "sanity", "total": 10, "passed": 10, "calls": []}
    keep, reason = mergeable(sanity)
    assert keep is False
    assert reason == "no report sections"


def test_evals_smoke_run_is_merged_like_any_other_report():
    """Smoke used to be dropped: its score comes from a handful of samples and
    it collides with the full run on every identity field, so a merged report
    carried two different scores for one task with nothing to tell them apart.
    That is now accepted deliberately — collecting every report matters more
    than avoiding the duplicate rows."""
    keep, reason = mergeable(_report(args=f"--dev-mode {SMOKE_FLAG}"))
    assert keep is True
    assert reason == "included"


def test_smoke_and_full_both_survive_deduplication(tmp_path):
    """They differ in run_command, so they are two identities, not one test run
    twice — de-duplication must not collapse them into one."""
    _write(
        tmp_path,
        "a",
        "report_evals_0_100.json",
        _report(args=f"--dev-mode {SMOKE_FLAG}", score=100.0),
    )
    _write(
        tmp_path, "b", "report_evals_0_200.json", _report(args="--dev-mode", score=83.0)
    )

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    kept, superseded = deduplicate(sources)
    assert len(kept) == 2
    assert superseded == []

    _, stats = merge_reports(tmp_path, tmp_path / "out")
    assert stats["merged"] == 2
    assert stats["sections"] == 2


def test_a_failed_test_is_still_merged():
    """Failing its acceptance check does not make a report less of a
    measurement — the filter is about shape, not outcome."""
    assert mergeable(_report(accuracy_check=3, score=40.0))[0] is True


def test_discovery_skips_workflow_logs(tmp_path):
    """Each artifact re-uploads run.py's own reports_output/ tree under
    workflow_logs/, holding the same results under a different name. Taking
    both would duplicate every section."""
    _write(tmp_path, "report_x-tests_m_t_1", "report_evals_0_1.json", _report())
    nested = (
        tmp_path
        / "report_x-tests_m_t_1"
        / "workflow_logs"
        / "reports_output"
        / "evals"
        / "data"
    )
    nested.mkdir(parents=True)
    (nested / "report_data_gemma_2026.json").write_text(json.dumps(_report()))

    found = discover_test_reports(tmp_path)
    assert [p.name for p in found] == ["report_evals_0_1.json"]


# --------------------------------------------------------------------------- #
# de-duplication across CI attempts
# --------------------------------------------------------------------------- #
def test_strip_server_url_normalises_both_flag_forms():
    assert _strip_server_url(
        "run.py --workflow evals --server-url http://a --dev-mode"
    ) == ("run.py --workflow evals --dev-mode")
    assert _strip_server_url("run.py --server-url=http://a --dev-mode") == (
        "run.py --dev-mode"
    )


def test_retried_test_keeps_only_the_newest_report(tmp_path):
    """A test that fails acceptance still writes a report, so retrying leaves
    two for one test. Merging both double-counts it and lets the superseded
    attempt's failure set the verdict."""
    _write(
        tmp_path, "a1", "report_evals_0_100.json", _report(accuracy_check=3, score=40.0)
    )
    _write(
        tmp_path, "a2", "report_evals_0_200.json", _report(accuracy_check=2, score=83.0)
    )

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    assert len(sources) == 2

    kept, superseded = deduplicate(sources)
    assert [s.job_id for s in kept] == [200]
    assert [(o.job_id, n.job_id) for o, n in superseded] == [(100, 200)]


def test_a_changed_server_url_does_not_split_one_test(tmp_path):
    """A deployment recreated between attempts hands out a new ngrok hostname;
    that must not read as two different tests."""
    _write(tmp_path, "a1", "report_evals_0_100.json", _report(server="https://old:443"))
    _write(tmp_path, "a2", "report_evals_0_200.json", _report(server="https://new:443"))

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    kept, superseded = deduplicate(sources)
    assert len(kept) == 1
    assert len(superseded) == 1


def test_different_tests_are_not_deduplicated(tmp_path):
    _write(tmp_path, "a", "report_evals_0_100.json", _report("evals"))
    _write(tmp_path, "b", "report_agentic_0_200.json", _report("agentic"))

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    kept, superseded = deduplicate(sources)
    assert len(kept) == 2
    assert superseded == []


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def test_merged_schema_concatenates_sections_and_claims_release(tmp_path):
    _write(tmp_path, "a", "report_evals_0_100.json", _report("evals"))
    _write(
        tmp_path, "b", "report_agentic_0_200.json", _report("agentic", task="swe_bench")
    )

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    schema = build_merged_schema(
        sources, model="google/gemma-4-31B-it", target="SC16:120-A"
    )

    assert len(schema.sections) == 2
    # Presents as a release so existing consumers treat it like a single-host report.
    assert schema.metadata["workflow"] == "release"
    assert schema.metadata["model_name"] == "google/gemma-4-31B-it"
    # No single run.py invocation stands behind a merged report.
    assert "merged from 2" in schema.metadata["run_command"]
    assert schema.metadata["report_id"]


def test_model_status_comes_from_the_sources(tmp_path):
    _write(
        tmp_path, "a", "report_evals_0_100.json", _report(model_status="EXPERIMENTAL")
    )
    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    assert resolve_model_status(sources) == "EXPERIMENTAL"


def test_enforcement_follows_the_model_status(tmp_path):
    """The same accuracy_check counts as a failure for FUNCTIONAL and as
    informational for EXPERIMENTAL. Delegating to acceptance_criteria_check
    inherits that gating instead of reimplementing it."""
    experimental = tmp_path / "experimental"
    _write(
        experimental,
        "a",
        "report_evals_0_100.json",
        _report(accuracy_check=3, score=40.0, model_status="EXPERIMENTAL"),
    )
    _, stats = merge_reports(experimental, tmp_path / "out-exp")
    assert stats["accepted"] is True

    functional = tmp_path / "functional"
    _write(
        functional,
        "a",
        "report_evals_0_100.json",
        _report(accuracy_check=3, score=40.0, model_status="FUNCTIONAL"),
    )
    _, stats = merge_reports(functional, tmp_path / "out-fun")
    assert stats["accepted"] is False


def test_disagreeing_model_status_enforces_everything(tmp_path):
    """A model spec change between attempts can leave sources disagreeing.
    Trusting either could let the laxer one relax enforcement for the whole
    report, so resolution yields "" and upstream then enforces every check —
    the same principle acceptance_criteria.py applies to a missing status."""
    _write(
        tmp_path, "a", "report_evals_0_100.json", _report(model_status="EXPERIMENTAL")
    )
    _write(
        tmp_path,
        "b",
        "report_agentic_0_200.json",
        _report("agentic", model_status="FUNCTIONAL"),
    )

    sources, _ = load_test_reports(discover_test_reports(tmp_path))
    assert resolve_model_status(sources) == ""


def test_disagreement_does_not_hide_a_failure(tmp_path):
    """The consequence of the rule above: an EXPERIMENTAL report alongside a
    failing FUNCTIONAL one must not launder the failure into a PASS."""
    _write(
        tmp_path, "a", "report_evals_0_100.json", _report(model_status="EXPERIMENTAL")
    )
    _write(
        tmp_path,
        "b",
        "report_agentic_0_200.json",
        _report("agentic", accuracy_check=3, score=40.0, model_status="FUNCTIONAL"),
    )

    _, stats = merge_reports(tmp_path, tmp_path / "out")
    assert stats["accepted"] is False


def test_a_requested_test_that_never_reported_blocks_acceptance(tmp_path):
    """Tests run sequentially with fail-fast, so later ones may never start.
    Without a blocker the merged report would read exactly like a clean run."""
    _write(tmp_path, "a", "report_evals_0_100.json", _report())

    _, stats = merge_reports(
        tmp_path,
        tmp_path / "out",
        missing_tests=["inference-workflow-benchmarks --dev-mode"],
    )
    assert stats["accepted"] is False
    assert stats["blockers"] >= 1


def test_nothing_to_merge_still_writes_a_report(tmp_path):
    """Otherwise "every test failed" is indistinguishable from "the job is not
    deployed yet"."""
    result, stats = merge_reports(tmp_path, tmp_path / "out")
    assert result is not None
    assert result.markdown_path.exists()
    assert stats["merged"] == 0
    assert stats["accepted"] is False


def test_merge_renders_the_release_shaped_json(tmp_path):
    _write(tmp_path, "a", "report_evals_0_100.json", _report())
    result, _ = merge_reports(tmp_path, tmp_path / "out", job_id="12345")

    payload = json.loads(result.json_path.read_text())
    assert set(payload) >= {
        "metadata",
        "sections",
        "acceptance_criteria",
        "acceptance_blockers",
        "acceptance_criteria_metadata",
        "acceptance_summary_markdown",
    }
    assert payload["metadata"]["workflow"] == "release"
    assert [
        c["name"] for c in payload["acceptance_criteria_metadata"]["categories"]
    ] == [
        "Benchmarks",
        "Evals",
        "Spec Tests",
    ]
