"""
Tests for issue #54: groomer epic assignment.

Covers:
  - GROOMER persona instructs listing open epics via list_issues(labels="epic")
  - GROOMER persona instructs checking whether an issue is already under an epic
  - GROOMER persona instructs skipping already-assigned issues with a comment
  - GROOMER persona instructs calling add_sub_issue to link matched issues
  - GROOMER persona instructs noting chosen epic and reason in the grooming comment
  - GROOMER persona instructs noting absence of fit in the grooming comment
  - GROOMER persona instructs graceful error handling when add_sub_issue fails
  - GROOMER persona instructs noting when no open epics exist
"""


class TestGroomerEpicAssignment:

    def _system(self):
        from orchestrator.personas import GROOMER
        return GROOMER["system"]

    def test_mentions_epic_label(self):
        assert "epic" in self._system()

    def test_instructs_list_open_epics(self):
        system = self._system()
        assert "list_issues" in system
        assert "epic" in system

    def test_instructs_list_issues_with_epic_label(self):
        # Exact call form: list_issues(state="open", labels="epic")
        system = self._system()
        assert "list_issues" in system
        assert '"epic"' in system or "'epic'" in system

    def test_instructs_skip_already_assigned(self):
        system = self._system().lower()
        assert "already" in system and "epic" in system

    def test_instructs_comment_on_already_assigned(self):
        system = self._system()
        assert "comment_issue" in system or "grooming comment" in system.lower()

    def test_instructs_add_sub_issue_for_match(self):
        system = self._system()
        assert "add_sub_issue" in system

    def test_instructs_note_chosen_epic_in_comment(self):
        system = self._system().lower()
        assert "assigned to epic" in system or ("chosen" in system and "epic" in system) or ("note" in system and "epic" in system)

    def test_instructs_note_no_fit_in_comment(self):
        system = self._system().lower()
        assert "no suitable epic" in system or "no fit" in system or "left unassigned" in system

    def test_instructs_note_no_epics_found(self):
        system = self._system().lower()
        assert "no open epic" in system or "no epics" in system

    def test_instructs_graceful_error_on_api_failure(self):
        system = self._system().lower()
        assert "error" in system and ("warn" in system or "warning" in system or "unavailable" in system)

    def test_epic_assignment_section_present(self):
        system = self._system()
        assert "Epic assignment" in system or "epic assignment" in system.lower()

    def test_add_sub_issue_called_with_epic_as_parent(self):
        system = self._system()
        assert "epic_number" in system or "parent_number=<epic_number>" in system

    def test_instructs_matching_based_on_title_and_description(self):
        system = self._system().lower()
        assert "title" in system and "description" in system
