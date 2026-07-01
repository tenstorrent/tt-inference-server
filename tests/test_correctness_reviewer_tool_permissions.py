"""
Tests for issue #79: correctness reviewer must check tool access permissions
when tools.py is modified.
"""

from orchestrator.personas import CORRECTNESS_REVIEWER


class TestCorrectnessReviewerToolPermissions:

    def test_checklist_mentions_tools_py(self):
        system = CORRECTNESS_REVIEWER["system"]
        assert "tools.py" in system

    def test_checklist_mentions_orchestrator_only(self):
        system = CORRECTNESS_REVIEWER["system"]
        assert "_ORCHESTRATOR_ONLY" in system

    def test_checklist_mentions_exclude_tools(self):
        system = CORRECTNESS_REVIEWER["system"]
        assert "exclude_tools" in system

    def test_checklist_frames_omission_as_correctness_defect(self):
        system = CORRECTNESS_REVIEWER["system"].lower()
        assert "correctness defect" in system

    def test_checklist_item_is_in_look_for_list(self):
        # The tool-permissions check must appear inside the "Look for:" bullet
        # list, not in a disconnected section.
        system = CORRECTNESS_REVIEWER["system"]
        look_for_start = system.index("Look for:")
        verdict_start = system.index("End your response")
        checklist_body = system[look_for_start:verdict_start]
        assert "_ORCHESTRATOR_ONLY" in checklist_body
        assert "exclude_tools" in checklist_body
