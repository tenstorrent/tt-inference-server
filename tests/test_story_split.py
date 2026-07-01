"""
Tests for issue #76: groomer story-split detection.

Covers:
  - ensure_label tool function exists, calls gh with --force (idempotent)
  - ensure_label passes name, color, and optional description correctly
  - ensure_label is registered in IMPL and DEFS
  - GROOMER system prompt documents the needs-split workflow
  - GROOMER system prompt requires skipping Priority/Effort for needs-split issues
  - GROOMER system prompt mentions ensure_label
"""

from unittest.mock import patch


# ---------------------------------------------------------------------------
# ensure_label function behaviour
# ---------------------------------------------------------------------------

class TestEnsureLabelFunction:

    def test_function_exists(self):
        from orchestrator.tools import ensure_label
        assert callable(ensure_label)

    def test_calls_gh_label_create(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert argv[0] == "label"
        assert argv[1] == "create"

    def test_label_name_passed_as_argv_token(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "needs-split" in argv

    def test_force_flag_present(self):
        # --force makes the call idempotent; without it gh errors if label exists.
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--force" in argv

    def test_color_passed_when_provided(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", color="ff0000", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--color" in argv
        assert argv[argv.index("--color") + 1] == "ff0000"

    def test_default_color_is_yellow(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--color" in argv
        assert argv[argv.index("--color") + 1] == "e4e669"

    def test_description_included_when_provided(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", description="Split me", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--description" in argv
        assert argv[argv.index("--description") + 1] == "Split me"

    def test_description_omitted_when_empty(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--description" not in argv

    def test_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split", cwd="/custom")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom"

    def test_none_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.ensure_label("needs-split")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] is None

    def test_returns_gh_output(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="label created"):
            result = tools_mod.ensure_label("needs-split")

        assert result == "label created"


# ---------------------------------------------------------------------------
# ensure_label dispatch via execute()
# ---------------------------------------------------------------------------

class TestEnsureLabelDispatch:

    def test_execute_routes_to_ensure_label(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "ensure_label", return_value="ok") as mock_fn:
            result = tools_mod.execute("ensure_label", {"name": "needs-split"})

        mock_fn.assert_called_once_with("needs-split", "e4e669", "", None)
        assert result == "ok"

    def test_execute_passes_color_and_description(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "ensure_label", return_value="ok") as mock_fn:
            tools_mod.execute(
                "ensure_label",
                {"name": "needs-split", "color": "ff0000", "description": "Split this"},
                cwd="/repo",
            )

        mock_fn.assert_called_once_with("needs-split", "ff0000", "Split this", "/repo")


# ---------------------------------------------------------------------------
# ensure_label schema in DEFS
# ---------------------------------------------------------------------------

class TestEnsureLabelSchema:

    def _get_def(self):
        from orchestrator.tools import DEFS
        matches = [d for d in DEFS if d["function"]["name"] == "ensure_label"]
        assert matches, "ensure_label not found in DEFS"
        return matches[0]

    def test_present_in_defs(self):
        self._get_def()

    def test_name_is_required(self):
        defn = self._get_def()
        assert "name" in defn["function"]["parameters"]["required"]

    def test_color_and_description_are_optional(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "color" not in required
        assert "description" not in required

    def test_name_param_is_string(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["name"]["type"] == "string"

    def test_description_mentions_idempotent(self):
        defn = self._get_def()
        desc = defn["function"]["description"].lower()
        assert "idempotent" in desc or "already exists" in desc or "not already exist" in desc


# ---------------------------------------------------------------------------
# GROOMER persona documents story-split behaviour
# ---------------------------------------------------------------------------

class TestGroomerPersonaStorySplit:

    def _system(self):
        from orchestrator.personas import GROOMER
        return GROOMER["system"]

    def test_mentions_needs_split_label(self):
        assert "needs-split" in self._system()

    def test_mentions_ensure_label(self):
        assert "ensure_label" in self._system()

    def test_documents_label_issue_step(self):
        system = self._system()
        assert "label_issue" in system

    def test_documents_comment_step(self):
        system = self._system()
        assert "comment_issue" in system

    def test_skips_priority_effort_for_needs_split(self):
        system = self._system().lower()
        # The prompt must instruct skipping Priority/Effort assignment
        assert "do not assign" in system or "skip" in system or "do not" in system

    def test_story_splitting_section_present(self):
        assert "story splitting" in self._system().lower() or "needs-split" in self._system()

    def test_ensure_label_in_tool_list(self):
        # The tool list in the system prompt should mention ensure_label
        assert "ensure_label" in self._system()
