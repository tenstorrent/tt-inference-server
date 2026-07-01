"""
Tests for issue #41: orchestrate() must include 'Fixes #N' in every PR body
when the task string contains an issue number, so GitHub auto-closes the issue
on merge.
"""

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# _parse_issue_number (unit tests for the private helper)
# ---------------------------------------------------------------------------

class TestParseIssueNumber:
    def _fn(self):
        from orchestrator.orchestrator import _parse_issue_number
        return _parse_issue_number

    def test_standard_format(self):
        assert self._fn()("Fix issue #41: closing reference") == 41

    def test_single_digit(self):
        assert self._fn()("Fix issue #3: typo") == 3

    def test_large_number(self):
        assert self._fn()("Fix issue #1234: refactor") == 1234

    def test_first_match_wins(self):
        assert self._fn()("Fix #10 and also #20") == 10

    def test_no_issue_returns_none(self):
        assert self._fn()("triage open issues") is None

    def test_empty_string_returns_none(self):
        assert self._fn()("") is None

    def test_bare_number_no_hash_returns_none(self):
        assert self._fn()("Fix issue 18 in the codebase") is None

    def test_returns_int(self):
        result = self._fn()("Fix #7: something")
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# orchestrate() PR body — Fixes #N injection
# ---------------------------------------------------------------------------

class TestFixesClosingReference:
    def _run_orchestrate(self, task: str) -> str:
        """Run orchestrate() end-to-end (all I/O mocked) and return the PR body."""
        import orchestrator.orchestrator as orch_mod

        captured = {}

        def fake_create_pr(title, body, branch, cwd=None):
            captured["body"] = body
            return "https://github.com/org/repo/pull/99"

        def fake_agent_run(persona, messages, **kwargs):
            # Return APPROVED so we reach Phase 4 in one pass.
            return "APPROVED", [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]

        # create_pr is imported inside orchestrate() so we patch the source module.
        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
        ):
            orch_mod.orchestrate(task, repo_path="/fake/repo", verbose=False)

        return captured.get("body", "")

    def test_fixes_reference_present_for_numbered_task(self):
        body = self._run_orchestrate("Fix issue #41: add closing reference")
        assert "Fixes #41" in body

    def test_fixes_reference_correct_number(self):
        body = self._run_orchestrate("Fix issue #7: something small")
        assert "Fixes #7" in body

    def test_fixes_reference_large_number(self):
        body = self._run_orchestrate("Fix issue #1234: big task")
        assert "Fixes #1234" in body

    def test_no_fixes_reference_without_issue_number(self):
        body = self._run_orchestrate("triage open issues and apply labels")
        assert "Fixes #" not in body

    def test_fixes_reference_on_its_own_line(self):
        """GitHub requires 'Fixes #N' to be on its own line to trigger auto-close."""
        body = self._run_orchestrate("Fix issue #41: closing reference")
        lines = body.splitlines()
        assert any(line.strip() == "Fixes #41" for line in lines)

    def test_fixes_reference_at_end_of_body(self):
        # Fixes line must appear after the ## Testing section, not buried mid-body.
        body = self._run_orchestrate("Fix issue #41: closing reference")
        fixes_pos = body.rfind("Fixes #41")
        testing_pos = body.find("## Testing")
        assert fixes_pos > testing_pos

    def test_first_issue_number_used_when_multiple_hashes(self):
        """Only the first #N in the task is used."""
        body = self._run_orchestrate("Fix #10 which also relates to #20")
        assert "Fixes #10" in body
        assert "Fixes #20" not in body


# ---------------------------------------------------------------------------
# PR body structure — sections matching the PR template (issue #53)
# ---------------------------------------------------------------------------

class TestPRBodySections:
    def _run_orchestrate(self, task: str) -> str:
        import orchestrator.orchestrator as orch_mod

        captured = {}

        def fake_create_pr(title, body, branch, cwd=None):
            captured["body"] = body
            return "https://github.com/org/repo/pull/99"

        def fake_agent_run(persona, messages, **kwargs):
            return "APPROVED", [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
        ):
            orch_mod.orchestrate(task, repo_path="/fake/repo", verbose=False)

        return captured.get("body", "")

    def test_summary_section_present(self):
        body = self._run_orchestrate("Fix issue #53: add PR template")
        assert "## Summary" in body

    def test_changes_section_present(self):
        body = self._run_orchestrate("Fix issue #53: add PR template")
        assert "## Changes" in body

    def test_testing_section_present(self):
        body = self._run_orchestrate("Fix issue #53: add PR template")
        assert "## Testing" in body

    def test_fixes_section_present(self):
        body = self._run_orchestrate("Fix issue #53: add PR template")
        assert "## Fixes" in body

    def test_section_order(self):
        body = self._run_orchestrate("Fix issue #53: add PR template")
        positions = [body.find(s) for s in ["## Summary", "## Changes", "## Testing", "## Fixes"]]
        assert positions == sorted(positions), "Sections are out of order"

    def test_fixes_section_na_when_no_issue(self):
        body = self._run_orchestrate("triage open issues and apply labels")
        assert "## Fixes" in body
        fixes_idx = body.find("## Fixes")
        fixes_content = body[fixes_idx:]
        assert "N/A" in fixes_content
