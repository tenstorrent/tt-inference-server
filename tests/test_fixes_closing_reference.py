"""
Tests for issue #41: orchestrate() must include 'Fixes #N' in every PR body
when the task string contains an issue number, so GitHub auto-closes the issue
on merge.

Updated for issue #154: PR body now comes from a one-shot generate_pr_body()
call, not from the implementer's final IMPLEMENTATION_COMPLETE message.
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
# _ensure_fixes_line (unit tests for the normalisation helper)
# ---------------------------------------------------------------------------

class TestEnsureFixesLine:
    def _fn(self):
        from orchestrator.orchestrator import _ensure_fixes_line
        return _ensure_fixes_line

    def test_appends_fixes_line_when_absent(self):
        body = "## Summary\nfoo\n\n## Changes\nbar\n\n## Testing\nbaz\n\n## Fixes"
        result = self._fn()(body, 41)
        assert "Fixes #41" in result

    def test_fixes_line_at_column_0(self):
        body = "## Summary\nfoo\n\n## Fixes"
        result = self._fn()(body, 41)
        lines = result.splitlines()
        fixes_lines = [l for l in lines if "Fixes #41" in l]
        assert fixes_lines, "Fixes #41 not found"
        assert all(l == l.lstrip() for l in fixes_lines), "Fixes line has leading whitespace"

    def test_replaces_wrong_issue_number(self):
        body = "## Summary\nfoo\n\n## Fixes\nFixes #99"
        result = self._fn()(body, 41)
        assert "Fixes #41" in result
        assert "Fixes #99" not in result

    def test_na_when_no_issue_number(self):
        body = "## Summary\nfoo\n\n## Fixes"
        result = self._fn()(body, None)
        assert "N/A" in result
        assert "Fixes #" not in result

    def test_adds_fixes_section_if_missing(self):
        body = "## Summary\nfoo\n\n## Changes\nbar\n\n## Testing\nbaz"
        result = self._fn()(body, 7)
        assert "## Fixes" in result
        assert "Fixes #7" in result

    def test_fixes_line_after_fixes_section(self):
        body = "## Summary\nfoo\n\n## Fixes"
        result = self._fn()(body, 41)
        fixes_section_pos = result.find("## Fixes")
        fixes_line_pos = result.find("Fixes #41")
        assert fixes_line_pos > fixes_section_pos


# ---------------------------------------------------------------------------
# orchestrate() PR body — Fixes #N injection
# ---------------------------------------------------------------------------

_FAKE_PR_BODY_TEMPLATE = """\
## Summary
{task}

## Changes
- orchestrator/orchestrator.py: updated something

## Testing
Tests pass.

## Fixes
Fixes #{issue}"""

_FAKE_PR_BODY_NO_ISSUE = """\
## Summary
Triage open issues.

## Changes
- No code changes.

## Testing
N/A

## Fixes
N/A"""


class TestFixesClosingReference:
    def _run_orchestrate(self, task: str) -> str:
        """Run orchestrate() end-to-end (all I/O mocked) and return the PR body."""
        import orchestrator.orchestrator as orch_mod
        import re

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

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            m = re.search(r"#(\d+)", task)
            if m:
                return _FAKE_PR_BODY_TEMPLATE.format(task=task, issue=m.group(1))
            return _FAKE_PR_BODY_NO_ISSUE

        # create_pr is imported inside orchestrate() so we patch the source module.
        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
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
        import re

        captured = {}

        def fake_create_pr(title, body, branch, cwd=None):
            captured["body"] = body
            return "https://github.com/org/repo/pull/99"

        def fake_agent_run(persona, messages, **kwargs):
            return "APPROVED", [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            m = re.search(r"#(\d+)", task)
            if m:
                return _FAKE_PR_BODY_TEMPLATE.format(task=task, issue=m.group(1))
            return _FAKE_PR_BODY_NO_ISSUE

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
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


# ---------------------------------------------------------------------------
# issue #81 / #154: PR body normalisation (column-0 Fixes line)
#
# With #154, the PR body now comes from generate_pr_body() (a one-shot LLM
# call), not from the implementer's IMPLEMENTATION_COMPLETE message.
# _ensure_fixes_line() post-processes the LLM output to guarantee the correct
# Fixes #N line appears at column 0.  The sentinel-stripping tests are
# obsolete (IMPLEMENTATION_COMPLETE never appears in the new flow).
# ---------------------------------------------------------------------------

class TestPRBodyNormalisation:
    """Verifies _ensure_fixes_line post-processing of the LLM-generated body."""

    def _run_orchestrate(self, task: str, llm_body: str) -> str:
        """Run orchestrate() with a controlled generate_pr_body return value."""
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

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            return llm_body

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
        ):
            orch_mod.orchestrate(task, repo_path="/fake/repo", verbose=False)

        return captured.get("body", "")

    def test_fixes_line_at_column_0(self):
        # _ensure_fixes_line must never indent the Fixes line.
        llm_body = "## Summary\nDid work.\n\n## Changes\n- file.py\n\n## Testing\nOK\n\n## Fixes\nFixes #81"
        body = self._run_orchestrate("Fix issue #81: column-0 check", llm_body)
        for line in body.splitlines():
            if "Fixes #" in line:
                assert line == line.lstrip(), (
                    f"'Fixes #N' line has leading whitespace: {line!r}"
                )

    def test_fixes_line_not_indented(self):
        llm_body = "## Summary\nSome impl.\n\n## Changes\n- f.py\n\n## Testing\nPass\n\n## Fixes\n    Fixes #81"
        body = self._run_orchestrate("Fix issue #81: indentation check", llm_body)
        fixes_line = next(
            (l for l in body.splitlines() if l.strip().startswith("Fixes #")), None
        )
        assert fixes_line is not None, "Fixes #N line missing from PR body"
        assert not fixes_line.startswith(" "), (
            f"Fixes line is indented: {fixes_line!r}"
        )

    def test_wrong_issue_number_in_llm_body_is_replaced(self):
        # Even if the LLM wrote the wrong number, _ensure_fixes_line corrects it.
        llm_body = "## Summary\nfoo\n\n## Fixes\nFixes #999"
        body = self._run_orchestrate("Fix issue #81: wrong number", llm_body)
        assert "Fixes #81" in body
        assert "Fixes #999" not in body

    def test_sentinel_never_appears_in_pr_body(self):
        # IMPLEMENTATION_COMPLETE is no longer the source of the PR body.
        llm_body = "## Summary\nDone.\n\n## Changes\n- x.py\n\n## Testing\nOK\n\n## Fixes\nFixes #81"
        body = self._run_orchestrate("Fix issue #81: sentinel absence", llm_body)
        assert "IMPLEMENTATION_COMPLETE" not in body

    def test_fixes_line_present_even_when_llm_omits_fixes_section(self):
        # _ensure_fixes_line adds ## Fixes if the model forgot it entirely.
        llm_body = "## Summary\nDone.\n\n## Changes\n- x.py\n\n## Testing\nOK"
        body = self._run_orchestrate("Fix issue #81: missing section", llm_body)
        assert "Fixes #81" in body
        assert "## Fixes" in body
