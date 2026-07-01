"""
Tests for issue #64: close_issue must be absent from the implementer's tool
schema and present in the groomer's.
"""

import orchestrator.agent as agent_mod
import orchestrator.tools as T
from orchestrator.personas import IMPLEMENTER, GROOMER
from orchestrator.orchestrator import _IMPLEMENTER_EXCLUDED_TOOLS


def _tool_names(exclude_tools=None):
    blocked = agent_mod._ORCHESTRATOR_ONLY | (exclude_tools or set())
    return {t["function"]["name"] for t in T.DEFS if t["function"]["name"] not in blocked}


class TestImplementerToolSchema:

    def test_close_issue_absent_for_implementer(self):
        names = _tool_names(exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS)
        assert "close_issue" not in names

    def test_create_pr_absent_for_implementer(self):
        # create_pr is orchestrator-only and must also be excluded
        names = _tool_names(exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS)
        assert "create_pr" not in names

    def test_other_tools_present_for_implementer(self):
        names = _tool_names(exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS)
        for tool in ("bash_exec", "read_file", "write_file", "git_status", "git_diff"):
            assert tool in names, f"{tool} missing from implementer schema"


class TestGroomerToolSchema:

    def test_close_issue_present_for_groomer(self):
        # Groomer passes no exclude_tools — close_issue must be available
        names = _tool_names(exclude_tools=None)
        assert "close_issue" in names

    def test_create_pr_absent_for_groomer(self):
        # create_pr is always orchestrator-only
        names = _tool_names(exclude_tools=None)
        assert "create_pr" not in names


class TestImplementerExcludedToolsConstant:

    def test_close_issue_in_excluded_set(self):
        assert "close_issue" in _IMPLEMENTER_EXCLUDED_TOOLS

    def test_excluded_set_does_not_contain_groomer_tools(self):
        groomer_tools = {"list_issues", "get_issue", "comment_issue", "label_issue",
                         "set_issue_field", "close_issue"}
        # Only close_issue should be excluded — all other issue tools stay available
        non_close = groomer_tools - {"close_issue"}
        assert not (non_close & _IMPLEMENTER_EXCLUDED_TOOLS)
