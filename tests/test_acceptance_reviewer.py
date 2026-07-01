"""
Tests for issue #28: acceptance-criteria reviewer persona.

Covers:
  - ACCEPTANCE_REVIEWER is defined in personas.py and included in REVIEWERS
  - orchestrate() passes the original task prompt as context to the acceptance reviewer
  - orchestrate() returns False when the acceptance reviewer rejects (unmet criteria)
  - orchestrate() returns False when the acceptance reviewer rejects due to partial
    work (implementer hit max_tool_rounds in a rebuttal and the acceptance reviewer
    detects incomplete work)
  - All three reviewers must approve for consensus to be reached
  - _build_reviewer_messages() injects task context only for the acceptance reviewer

All external I/O is mocked; no network calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SYS_MSG = [{"role": "system", "content": "s"}]
TASK = "Add a widget: implement Widget class, add tests, update README."


def _make_fake_run(responses: dict):
    """Return a fake A.run() that maps persona name -> return value.

    ``responses`` maps persona_name -> either:
      - a string (returned as (string, _SYS_MSG))
      - a callable(persona, messages, **kwargs) -> (str, list)
    """
    def fake_run(persona, messages, **kwargs):
        name = persona["name"]
        val = responses.get(name, "APPROVED")
        if callable(val):
            return val(persona, messages, **kwargs)
        return val, _SYS_MSG[:]
    return fake_run


def _stub_create_pr(monkeypatch):
    import orchestrator.tools as tools_mod
    monkeypatch.setattr(tools_mod, "create_pr", lambda *a, **kw: "https://github.com/fake/pull/1")


# ---------------------------------------------------------------------------
# Persona definition tests
# ---------------------------------------------------------------------------

class TestAcceptanceReviewerPersona:
    def test_acceptance_reviewer_defined(self):
        """ACCEPTANCE_REVIEWER must be importable from orchestrator.personas."""
        from orchestrator.personas import ACCEPTANCE_REVIEWER
        assert ACCEPTANCE_REVIEWER["name"] == "acceptance_reviewer"

    def test_acceptance_reviewer_has_model_and_system(self):
        from orchestrator.personas import ACCEPTANCE_REVIEWER
        assert "model" in ACCEPTANCE_REVIEWER
        assert "system" in ACCEPTANCE_REVIEWER
        assert len(ACCEPTANCE_REVIEWER["system"]) > 50

    def test_acceptance_reviewer_in_reviewers_list(self):
        """ACCEPTANCE_REVIEWER must be included in the REVIEWERS list."""
        from orchestrator.personas import REVIEWERS, ACCEPTANCE_REVIEWER
        names = [r["name"] for r in REVIEWERS]
        assert ACCEPTANCE_REVIEWER["name"] in names

    def test_reviewers_has_three_members(self):
        """REVIEWERS must contain exactly three personas after adding ACCEPTANCE_REVIEWER."""
        from orchestrator.personas import REVIEWERS
        assert len(REVIEWERS) == 3

    def test_acceptance_reviewer_system_mentions_partial_work(self):
        """The acceptance reviewer system prompt must address partial/incomplete work."""
        from orchestrator.personas import ACCEPTANCE_REVIEWER
        system = ACCEPTANCE_REVIEWER["system"].lower()
        assert "partial" in system or "max_tool_rounds" in system or "incomplete" in system

    def test_acceptance_reviewer_system_mentions_acceptance_criteria(self):
        """The acceptance reviewer system prompt must reference acceptance criteria."""
        from orchestrator.personas import ACCEPTANCE_REVIEWER
        system = ACCEPTANCE_REVIEWER["system"].lower()
        assert "acceptance" in system or "criterion" in system or "criteria" in system


# ---------------------------------------------------------------------------
# _build_reviewer_messages tests
# ---------------------------------------------------------------------------

class TestBuildReviewerMessages:
    def _shared_history(self):
        return [{"role": "user", "content": "impl done"}, {"role": "assistant", "content": "ok"}]

    def test_acceptance_reviewer_gets_task_context(self):
        """Acceptance reviewer messages must include the original task prompt."""
        from orchestrator.orchestrator import _build_reviewer_messages
        from orchestrator.personas import ACCEPTANCE_REVIEWER

        shared = self._shared_history()
        msgs = _build_reviewer_messages(ACCEPTANCE_REVIEWER, shared, TASK, "give verdict")

        full_text = " ".join(m["content"] for m in msgs if m.get("content"))
        assert TASK in full_text

    def test_acceptance_reviewer_task_context_is_first_message(self):
        """The task context message must appear before the shared history."""
        from orchestrator.orchestrator import _build_reviewer_messages
        from orchestrator.personas import ACCEPTANCE_REVIEWER

        shared = self._shared_history()
        msgs = _build_reviewer_messages(ACCEPTANCE_REVIEWER, shared, TASK, "give verdict")

        # First message should contain the task
        assert TASK in msgs[0]["content"]

    def test_security_reviewer_does_not_get_task_prefix(self):
        """Security reviewer must NOT receive the injected task-context prefix."""
        from orchestrator.orchestrator import _build_reviewer_messages
        from orchestrator.personas import SECURITY_REVIEWER

        shared = self._shared_history()
        msgs = _build_reviewer_messages(SECURITY_REVIEWER, shared, TASK, "give verdict")

        # Should be shared_history + prompt only (no extra task prefix)
        assert msgs == shared + [{"role": "user", "content": "give verdict"}]

    def test_correctness_reviewer_does_not_get_task_prefix(self):
        """Correctness reviewer must NOT receive the injected task-context prefix."""
        from orchestrator.orchestrator import _build_reviewer_messages
        from orchestrator.personas import CORRECTNESS_REVIEWER

        shared = self._shared_history()
        msgs = _build_reviewer_messages(CORRECTNESS_REVIEWER, shared, TASK, "give verdict")

        assert msgs == shared + [{"role": "user", "content": "give verdict"}]

    def test_acceptance_reviewer_prompt_is_last_message(self):
        """The review prompt must be the final message in the acceptance reviewer call."""
        from orchestrator.orchestrator import _build_reviewer_messages
        from orchestrator.personas import ACCEPTANCE_REVIEWER

        shared = self._shared_history()
        prompt = "Please review the implementation and give your verdict."
        msgs = _build_reviewer_messages(ACCEPTANCE_REVIEWER, shared, TASK, prompt)

        assert msgs[-1]["content"] == prompt


# ---------------------------------------------------------------------------
# orchestrate() reject path: acceptance reviewer rejects
# ---------------------------------------------------------------------------

class TestOrchestrateAcceptanceReviewerRejectPath:

    def _patch_all(self, monkeypatch):
        import orchestrator.orchestrator as orch
        from orchestrator.personas import IMPLEMENTER, SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER
        monkeypatch.setattr(orch, "IMPLEMENTER", IMPLEMENTER)
        monkeypatch.setattr(orch, "REVIEWERS", [SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER])

    def test_returns_false_when_acceptance_reviewer_rejects(self, monkeypatch):
        """orchestrate() must return False when the acceptance reviewer rejects."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)

        responses = {
            "implementer": "IMPLEMENTATION_COMPLETE",
            "security_reviewer": "APPROVED",
            "correctness_reviewer": "APPROVED",
            "acceptance_reviewer": "OBJECTION: tests for Widget class are missing",
        }
        monkeypatch.setattr(orch.A, "run", _make_fake_run(responses))

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)
        assert result is False

    def test_acceptance_reviewer_rejection_appears_in_objectors(self, monkeypatch):
        """When acceptance reviewer rejects, the objection must cause failure."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)

        objection_text = "OBJECTION: README not updated"
        responses = {
            "implementer": "IMPLEMENTATION_COMPLETE",
            "security_reviewer": "APPROVED",
            "correctness_reviewer": "APPROVED",
            "acceptance_reviewer": objection_text,
        }
        monkeypatch.setattr(orch.A, "run", _make_fake_run(responses))

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)
        assert result is False

    def test_all_three_must_approve_for_success(self, monkeypatch):
        """A PR must NOT be opened unless all three reviewers approve."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        # Only security and correctness approve; acceptance rejects
        responses = {
            "implementer": "IMPLEMENTATION_COMPLETE",
            "security_reviewer": "APPROVED",
            "correctness_reviewer": "APPROVED",
            "acceptance_reviewer": "OBJECTION: Widget class not implemented",
        }
        monkeypatch.setattr(orch.A, "run", _make_fake_run(responses))

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)
        assert result is False

    def test_success_when_all_three_approve(self, monkeypatch):
        """orchestrate() must succeed (open PR) when all three reviewers approve."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        responses = {
            "implementer": "IMPLEMENTATION_COMPLETE",
            "security_reviewer": "APPROVED",
            "correctness_reviewer": "APPROVED",
            "acceptance_reviewer": "APPROVED",
        }
        monkeypatch.setattr(orch.A, "run", _make_fake_run(responses))

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)
        assert result is True

    def test_acceptance_reviewer_receives_task_prompt_in_messages(self, monkeypatch):
        """The acceptance reviewer's A.run() call must include the original task text."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        captured_messages = {}

        def tracking_run(persona, messages, **kwargs):
            if persona["name"] == "acceptance_reviewer":
                captured_messages["messages"] = messages
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", tracking_run)

        orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)

        assert "messages" in captured_messages, "acceptance_reviewer was never called"
        full_text = " ".join(m.get("content", "") for m in captured_messages["messages"])
        assert TASK in full_text, "Task prompt not found in acceptance reviewer messages"

    def test_other_reviewers_do_not_receive_task_prefix(self, monkeypatch):
        """Security and correctness reviewers must NOT have the task prepended."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        captured = {}

        def tracking_run(persona, messages, **kwargs):
            if persona["name"] in ("security_reviewer", "correctness_reviewer"):
                captured[persona["name"]] = messages
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", tracking_run)

        orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)

        for name in ("security_reviewer", "correctness_reviewer"):
            assert name in captured
            # The first message should not be the task-context prefix
            first_content = captured[name][0].get("content", "")
            assert "ORIGINAL TASK" not in first_content


# ---------------------------------------------------------------------------
# Reject path: partial work (simulated max_tool_rounds hit scenario)
# ---------------------------------------------------------------------------

class TestAcceptanceReviewerRejectsPartialWork:
    """The acceptance reviewer must reject when the implementer's output signals
    partial/incomplete work — including the case where max_tool_rounds was hit
    in a debate rebuttal and the acceptance reviewer detects unfinished work."""

    def _patch_all(self, monkeypatch):
        import orchestrator.orchestrator as orch
        from orchestrator.personas import IMPLEMENTER, SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER
        monkeypatch.setattr(orch, "IMPLEMENTER", IMPLEMENTER)
        monkeypatch.setattr(orch, "REVIEWERS", [SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER])

    def test_acceptance_reviewer_rejects_partial_implementation(self, monkeypatch):
        """When the implementation is explicitly incomplete, acceptance reviewer
        must reject regardless of security/correctness approval."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)

        # Implementation text signals partial work
        partial_impl = "I started adding the Widget class but ran out of steps. Partial work only."

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                return partial_impl, _SYS_MSG[:]
            if persona["name"] == "security_reviewer":
                return "APPROVED", _SYS_MSG[:]
            if persona["name"] == "correctness_reviewer":
                return "APPROVED", _SYS_MSG[:]
            if persona["name"] == "acceptance_reviewer":
                # Acceptance reviewer sees the partial impl and rejects
                return "OBJECTION: Widget class incomplete — tests and README not added", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=0, verbose=False)
        assert result is False

    def test_debate_does_not_proceed_to_pr_while_acceptance_rejects(self, monkeypatch):
        """Even after debate rounds, no PR is opened while acceptance reviewer keeps rejecting."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)

        review_call_count = {"acceptance": 0}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                return "IMPLEMENTATION_COMPLETE", _SYS_MSG[:]
            if persona["name"] == "security_reviewer":
                return "APPROVED", _SYS_MSG[:]
            if persona["name"] == "correctness_reviewer":
                return "APPROVED", _SYS_MSG[:]
            if persona["name"] == "acceptance_reviewer":
                review_call_count["acceptance"] += 1
                # Always rejects — criteria never met
                return "OBJECTION: tests for Widget class still missing", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=2, verbose=False)
        assert result is False
        # Acceptance reviewer should have been called once per debate round
        assert review_call_count["acceptance"] >= 1

    def test_acceptance_reviewer_called_on_every_debate_round(self, monkeypatch):
        """Acceptance reviewer must be invoked on each debate round, not just the first."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        call_counts = {"impl": 0, "acceptance": 0}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                call_counts["impl"] += 1
                return "IMPLEMENTATION_COMPLETE", _SYS_MSG[:]
            if persona["name"] == "acceptance_reviewer":
                call_counts["acceptance"] += 1
                # Reject first time, approve second time (after rebuttal)
                if call_counts["acceptance"] == 1:
                    return "OBJECTION: tests missing", _SYS_MSG[:]
                return "APPROVED", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        result = orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=2, verbose=False)
        assert result is True
        assert call_counts["acceptance"] == 2, (
            f"Expected acceptance_reviewer called twice (initial + after rebuttal), "
            f"got {call_counts['acceptance']}"
        )

    def test_acceptance_reviewer_receives_task_on_rebuttal_round(self, monkeypatch):
        """Task prompt must be injected into acceptance reviewer messages on every round."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        call_counts = {"impl": 0, "acceptance": 0}
        task_found_on_round = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                call_counts["impl"] += 1
                return "IMPLEMENTATION_COMPLETE", _SYS_MSG[:]
            if persona["name"] == "acceptance_reviewer":
                call_counts["acceptance"] += 1
                rnd = call_counts["acceptance"]
                full_text = " ".join(m.get("content", "") for m in messages)
                task_found_on_round[rnd] = TASK in full_text
                if rnd == 1:
                    return "OBJECTION: README missing", _SYS_MSG[:]
                return "APPROVED", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate(TASK, "/fake/repo", max_debate_rounds=2, verbose=False)

        assert task_found_on_round.get(1) is True, "Task not found in round 1 acceptance reviewer messages"
        assert task_found_on_round.get(2) is True, "Task not found in round 2 acceptance reviewer messages"


# ---------------------------------------------------------------------------
# max_tool_rounds forwarded to acceptance reviewer
# ---------------------------------------------------------------------------

class TestAcceptanceReviewerReceivesMaxToolRounds:

    def _patch_all(self, monkeypatch):
        import orchestrator.orchestrator as orch
        from orchestrator.personas import IMPLEMENTER, SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER
        monkeypatch.setattr(orch, "IMPLEMENTER", IMPLEMENTER)
        monkeypatch.setattr(orch, "REVIEWERS", [SECURITY_REVIEWER, CORRECTNESS_REVIEWER, ACCEPTANCE_REVIEWER])

    def test_max_tool_rounds_forwarded_to_acceptance_reviewer(self, monkeypatch):
        """orchestrate() must forward max_tool_rounds to the acceptance reviewer."""
        import orchestrator.orchestrator as orch
        self._patch_all(monkeypatch)
        _stub_create_pr(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "acceptance_reviewer":
                captured["max_tool_rounds"] = kwargs.get("max_tool_rounds")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate(TASK, "/fake/repo", max_tool_rounds=17, verbose=False)
        assert captured.get("max_tool_rounds") == 17


# ---------------------------------------------------------------------------
# _extract_verdict: finding forces reject (issue #34)
# ---------------------------------------------------------------------------

class TestExtractVerdictFindingForcesReject:
    """FINDING: is a third verdict branch in the bottom-up scan.

    Bottom-up means the latest verdict wins, preserving the invariant that a
    reviewer can quote an old FINDING: and then write APPROVED to lift it.
    A FINDING: only rejects when it is the last verdict token — i.e. it appears
    after (below) any APPROVED in the text.
    """

    def test_finding_as_final_verdict_is_rejected(self):
        # FINDING: is the last verdict line (starts the line) -> rejected
        from orchestrator.orchestrator import _extract_verdict
        text = "Looks mostly fine.\nAPPROVED\nFINDING: SQL injection at db.py:42"
        approved, objection = _extract_verdict(text)
        assert approved is False
        assert "FINDING" in objection.upper()

    def test_finding_only_no_approved_is_rejected(self):
        # No APPROVED at all; FINDING: alone -> rejected
        from orchestrator.orchestrator import _extract_verdict
        text = "FINDING: missing input validation (low severity)"
        approved, _ = _extract_verdict(text)
        assert approved is False

    def test_finding_returns_finding_line_as_objection(self):
        from orchestrator.orchestrator import _extract_verdict
        finding_line = "FINDING: missing auth check at api.py:10"
        text = f"Analysis complete.\n{finding_line}"
        _, objection = _extract_verdict(text)
        assert objection == finding_line

    def test_approved_after_finding_wins(self):
        # APPROVED appears after FINDING: in text -> bottom-up hits APPROVED first -> approve.
        # This is the re-review case: reviewer quotes an old finding then approves.
        from orchestrator.orchestrator import _extract_verdict
        text = "FINDING: null dereference (previous round)\nImplementer fixed it.\nAPPROVED"
        approved, _ = _extract_verdict(text)
        assert approved is True

    def test_no_finding_approved_still_passes(self):
        from orchestrator.orchestrator import _extract_verdict
        text = "Code looks correct. No issues found.\nAPPROVED"
        approved, _ = _extract_verdict(text)
        assert approved is True

    def test_objection_without_finding_still_rejected(self):
        from orchestrator.orchestrator import _extract_verdict
        text = "There is a null dereference risk.\nOBJECTION: null dereference at util.py:5"
        approved, objection = _extract_verdict(text)
        assert approved is False
        assert "OBJECTION" in objection.upper()

    def test_finding_case_insensitive(self):
        from orchestrator.orchestrator import _extract_verdict
        text = "finding: path traversal in file loader"
        approved, _ = _extract_verdict(text)
        assert approved is False

    def test_finding_after_approved_forces_reject(self):
        # Reviewer writes APPROVED then adds a finding below it -> finding wins
        from orchestrator.orchestrator import _extract_verdict
        text = (
            "The implementation looks good overall.\n"
            "APPROVED\n"
            "FINDING: unvalidated redirect at login.py:88 (low severity, noted for triage)"
        )
        approved, _ = _extract_verdict(text)
        assert approved is False

    def test_multiple_findings_returns_bottommost(self):
        # Bottom-up scan returns the last FINDING: in the text (closest to end)
        from orchestrator.orchestrator import _extract_verdict
        text = "FINDING: issue A\nFINDING: issue B"
        approved, objection = _extract_verdict(text)
        assert approved is False
        assert "issue B" in objection

    def test_approved_with_no_concerns_passes(self):
        from orchestrator.orchestrator import _extract_verdict
        for text in ["APPROVED", "Everything is correct.\nAPPROVED", "APPROVED\n"]:
            approved, _ = _extract_verdict(text)
            assert approved is True, f"Expected True for: {text!r}"


# ---------------------------------------------------------------------------
# Reviewer system prompts: findings must force OBJECTION (issue #34)
# ---------------------------------------------------------------------------

class TestReviewerPromptsRequireObjectionForFindings:

    def test_security_reviewer_prompt_forbids_approved_with_finding(self):
        from orchestrator.personas import SECURITY_REVIEWER
        system = SECURITY_REVIEWER["system"].lower()
        # Must tell the reviewer that any finding requires OBJECTION
        assert "objection" in system
        assert any(phrase in system for phrase in [
            "any finding", "regardless of severity", "zero unresolved",
        ])

    def test_correctness_reviewer_prompt_forbids_approved_with_finding(self):
        from orchestrator.personas import CORRECTNESS_REVIEWER
        system = CORRECTNESS_REVIEWER["system"].lower()
        assert "objection" in system
        assert any(phrase in system for phrase in [
            "any concern", "regardless of severity", "zero unresolved",
        ])

    def test_security_reviewer_prompt_mentions_severity_in_objection_text(self):
        from orchestrator.personas import SECURITY_REVIEWER
        system = SECURITY_REVIEWER["system"].lower()
        # Severity should be mentioned as belonging in OBJECTION text, not vote
        assert "severity" in system

    def test_correctness_reviewer_prompt_mentions_severity_in_objection_text(self):
        from orchestrator.personas import CORRECTNESS_REVIEWER
        system = CORRECTNESS_REVIEWER["system"].lower()
        assert "severity" in system

    def test_acceptance_reviewer_prompt_unchanged_wrt_approved_criteria(self):
        """Acceptance reviewer already requires zero unmet criteria for APPROVED."""
        from orchestrator.personas import ACCEPTANCE_REVIEWER
        system = ACCEPTANCE_REVIEWER["system"].lower()
        assert "approved" in system
        assert "objection" in system
