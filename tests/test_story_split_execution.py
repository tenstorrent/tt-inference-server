"""
Tests for issue #77: groomer story-split execution.

Covers:
  - create_issue: calls gh issue create with --title and --body-file
  - create_issue: optional labels are passed as --label tokens
  - create_issue: registered in IMPL and DEFS; required params correct
  - add_sub_issue: calls the GitHub sub-issues REST API via gh api
  - add_sub_issue: registered in IMPL and DEFS; required params correct
  - remove_label: calls gh issue edit --remove-label
  - remove_label: registered in IMPL and DEFS; required params correct
  - Access permissions: create_issue, add_sub_issue, remove_label excluded from implementer
  - GROOMER persona documents the full split-execution workflow
"""

from unittest.mock import patch, call


# ---------------------------------------------------------------------------
# create_issue
# ---------------------------------------------------------------------------

class TestCreateIssueFunction:

    def test_function_exists(self):
        from orchestrator.tools import create_issue
        assert callable(create_issue)

    def test_calls_gh_issue_create(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="https://github.com/org/repo/issues/99") as mock_gh:
            tools_mod.create_issue("My title", "My body", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert argv[0] == "issue"
        assert argv[1] == "create"

    def test_title_passed_as_argv_token(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.create_issue("My title", "body", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--title" in argv
        assert argv[argv.index("--title") + 1] == "My title"

    def test_body_passed_via_body_file(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.create_issue("title", "Some body content", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--body-file" in argv

    def test_labels_passed_as_label_tokens(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.create_issue("title", "body", labels="bug,enhancement", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--label" in argv
        label_indices = [i for i, a in enumerate(argv) if a == "--label"]
        label_values = [argv[i + 1] for i in label_indices]
        assert "bug" in label_values
        assert "enhancement" in label_values

    def test_no_label_flag_when_labels_empty(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.create_issue("title", "body", labels="", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--label" not in argv

    def test_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.create_issue("title", "body", cwd="/custom")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom"

    def test_returns_gh_output(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="https://github.com/org/repo/issues/42"):
            result = tools_mod.create_issue("title", "body")

        assert result == "https://github.com/org/repo/issues/42"


class TestCreateIssueDispatch:

    def test_execute_routes_to_create_issue(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "create_issue", return_value="ok") as mock_fn:
            result = tools_mod.execute("create_issue", {"title": "T", "body": "B"})

        mock_fn.assert_called_once_with("T", "B", "", None)
        assert result == "ok"

    def test_execute_passes_labels(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "create_issue", return_value="ok") as mock_fn:
            tools_mod.execute(
                "create_issue",
                {"title": "T", "body": "B", "labels": "bug"},
                cwd="/repo",
            )

        mock_fn.assert_called_once_with("T", "B", "bug", "/repo")


class TestCreateIssueSchema:

    def _get_def(self):
        from orchestrator.tools import DEFS
        matches = [d for d in DEFS if d["function"]["name"] == "create_issue"]
        assert matches, "create_issue not found in DEFS"
        return matches[0]

    def test_present_in_defs(self):
        self._get_def()

    def test_title_and_body_required(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "title" in required
        assert "body" in required

    def test_labels_optional(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "labels" not in required

    def test_title_and_body_are_strings(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["title"]["type"] == "string"
        assert props["body"]["type"] == "string"


# ---------------------------------------------------------------------------
# add_sub_issue
# ---------------------------------------------------------------------------

class TestAddSubIssueFunction:

    def test_function_exists(self):
        from orchestrator.tools import add_sub_issue
        assert callable(add_sub_issue)

    def test_calls_gh_api(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="{}") as mock_gh:
            tools_mod.add_sub_issue(10, 20, cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert argv[0] == "api"

    def test_targets_sub_issues_endpoint(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="{}") as mock_gh:
            tools_mod.add_sub_issue(10, 20, cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        endpoint = argv[1]
        assert "sub_issues" in endpoint
        assert "10" in endpoint

    def test_uses_post_method(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="{}") as mock_gh:
            tools_mod.add_sub_issue(10, 20, cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--method" in argv
        assert argv[argv.index("--method") + 1] == "POST"

    def test_passes_child_number(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="{}") as mock_gh:
            tools_mod.add_sub_issue(10, 20, cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        # child number must appear in the -f field argument
        field_args = [argv[i + 1] for i, a in enumerate(argv) if a == "-f"]
        assert any("20" in fa for fa in field_args)

    def test_numeric_injection_rejected(self):
        # int() coercion means a non-numeric string raises ValueError
        from orchestrator.tools import add_sub_issue
        import pytest
        with pytest.raises((ValueError, TypeError)):
            add_sub_issue("10; rm -rf /", 20)  # type: ignore[arg-type]

    def test_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="{}") as mock_gh:
            tools_mod.add_sub_issue(1, 2, cwd="/custom")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom"

    def test_returns_gh_output(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value='{"id": 1}'):
            result = tools_mod.add_sub_issue(1, 2)

        assert result == '{"id": 1}'


class TestAddSubIssueDispatch:

    def test_execute_routes_to_add_sub_issue(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "add_sub_issue", return_value="ok") as mock_fn:
            result = tools_mod.execute("add_sub_issue", {"parent_number": 10, "child_number": 20})

        mock_fn.assert_called_once_with(10, 20, None)
        assert result == "ok"

    def test_execute_forwards_cwd(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "add_sub_issue", return_value="ok") as mock_fn:
            tools_mod.execute("add_sub_issue", {"parent_number": 5, "child_number": 6}, cwd="/repo")

        mock_fn.assert_called_once_with(5, 6, "/repo")


class TestAddSubIssueSchema:

    def _get_def(self):
        from orchestrator.tools import DEFS
        matches = [d for d in DEFS if d["function"]["name"] == "add_sub_issue"]
        assert matches, "add_sub_issue not found in DEFS"
        return matches[0]

    def test_present_in_defs(self):
        self._get_def()

    def test_both_params_required(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "parent_number" in required
        assert "child_number" in required

    def test_params_are_integers(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["parent_number"]["type"] == "integer"
        assert props["child_number"]["type"] == "integer"


# ---------------------------------------------------------------------------
# remove_label
# ---------------------------------------------------------------------------

class TestRemoveLabelFunction:

    def test_function_exists(self):
        from orchestrator.tools import remove_label
        assert callable(remove_label)

    def test_calls_gh_issue_edit(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.remove_label(42, "needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert argv[0] == "issue"
        assert argv[1] == "edit"

    def test_issue_number_in_argv(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.remove_label(42, "needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "42" in argv

    def test_remove_label_flag_present(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.remove_label(42, "needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--remove-label" in argv

    def test_label_name_passed(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.remove_label(42, "needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        idx = argv.index("--remove-label")
        assert argv[idx + 1] == "needs-split"

    def test_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.remove_label(1, "needs-split", cwd="/custom")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom"

    def test_returns_gh_output(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="label removed"):
            result = tools_mod.remove_label(1, "needs-split")

        assert result == "label removed"


class TestRemoveLabelDispatch:

    def test_execute_routes_to_remove_label(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "remove_label", return_value="ok") as mock_fn:
            result = tools_mod.execute("remove_label", {"number": 42, "label": "needs-split"})

        mock_fn.assert_called_once_with(42, "needs-split", None)
        assert result == "ok"

    def test_execute_forwards_cwd(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "remove_label", return_value="ok") as mock_fn:
            tools_mod.execute("remove_label", {"number": 7, "label": "wip"}, cwd="/repo")

        mock_fn.assert_called_once_with(7, "wip", "/repo")


class TestRemoveLabelSchema:

    def _get_def(self):
        from orchestrator.tools import DEFS
        matches = [d for d in DEFS if d["function"]["name"] == "remove_label"]
        assert matches, "remove_label not found in DEFS"
        return matches[0]

    def test_present_in_defs(self):
        self._get_def()

    def test_number_and_label_required(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "number" in required
        assert "label" in required

    def test_number_is_integer(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["number"]["type"] == "integer"

    def test_label_is_string(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["label"]["type"] == "string"


# ---------------------------------------------------------------------------
# Access permissions — new tools excluded from implementer
# ---------------------------------------------------------------------------

class TestNewToolsAccessPermissions:

    def _implementer_tool_names(self):
        import orchestrator.agent as agent_mod
        import orchestrator.tools as T
        from orchestrator.orchestrator import _IMPLEMENTER_EXCLUDED_TOOLS
        blocked = agent_mod._ORCHESTRATOR_ONLY | _IMPLEMENTER_EXCLUDED_TOOLS
        return {t["function"]["name"] for t in T.DEFS if t["function"]["name"] not in blocked}

    def _all_tool_names(self):
        import orchestrator.agent as agent_mod
        import orchestrator.tools as T
        blocked = agent_mod._ORCHESTRATOR_ONLY
        return {t["function"]["name"] for t in T.DEFS if t["function"]["name"] not in blocked}

    def test_create_issue_excluded_from_implementer(self):
        assert "create_issue" not in self._implementer_tool_names()

    def test_add_sub_issue_excluded_from_implementer(self):
        assert "add_sub_issue" not in self._implementer_tool_names()

    def test_remove_label_excluded_from_implementer(self):
        assert "remove_label" not in self._implementer_tool_names()

    def test_create_issue_available_to_groomer(self):
        # Groomer passes no extra exclude_tools
        assert "create_issue" in self._all_tool_names()

    def test_add_sub_issue_available_to_groomer(self):
        assert "add_sub_issue" in self._all_tool_names()

    def test_remove_label_available_to_groomer(self):
        assert "remove_label" in self._all_tool_names()

    def test_create_issue_not_orchestrator_only(self):
        # create_issue must be callable by the groomer agent, not reserved for orchestrator
        import orchestrator.agent as agent_mod
        assert "create_issue" not in agent_mod._ORCHESTRATOR_ONLY

    def test_add_sub_issue_not_orchestrator_only(self):
        import orchestrator.agent as agent_mod
        assert "add_sub_issue" not in agent_mod._ORCHESTRATOR_ONLY

    def test_remove_label_not_orchestrator_only(self):
        import orchestrator.agent as agent_mod
        assert "remove_label" not in agent_mod._ORCHESTRATOR_ONLY

    def test_new_tools_in_implementer_excluded_set(self):
        from orchestrator.orchestrator import _IMPLEMENTER_EXCLUDED_TOOLS
        assert "create_issue" in _IMPLEMENTER_EXCLUDED_TOOLS
        assert "add_sub_issue" in _IMPLEMENTER_EXCLUDED_TOOLS
        assert "remove_label" in _IMPLEMENTER_EXCLUDED_TOOLS


# ---------------------------------------------------------------------------
# GROOMER persona documents story-split execution
# ---------------------------------------------------------------------------

class TestGroomerPersonaStorySplitExecution:

    def _system(self):
        from orchestrator.personas import GROOMER
        return GROOMER["system"]

    def test_mentions_create_issue(self):
        assert "create_issue" in self._system()

    def test_mentions_add_sub_issue(self):
        assert "add_sub_issue" in self._system()

    def test_mentions_remove_label(self):
        assert "remove_label" in self._system()

    def test_instructs_to_keep_parent_open(self):
        system = self._system().lower()
        assert "keep" in system and "open" in system

    def test_instructs_comment_listing_sub_issues(self):
        system = self._system()
        assert "comment_issue" in system

    def test_instructs_remove_needs_split_label_after_split(self):
        system = self._system()
        assert "remove_label" in system
        assert "needs-split" in system

    def test_instructs_not_fixed_count(self):
        system = self._system().lower()
        # Should mention that the count is not fixed
        assert "not a fixed" in system or "as many" in system or "appropriate number" in system

    def test_documents_sub_issue_api_link(self):
        system = self._system()
        # Must instruct linking via add_sub_issue
        assert "add_sub_issue" in system
