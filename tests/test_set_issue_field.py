import json
import subprocess
from unittest.mock import patch, MagicMock


class TestSetIssueFieldFunction:

    def test_function_exists(self):
        from orchestrator.tools import set_issue_field
        assert callable(set_issue_field)

    def test_uses_patch_method(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field(42, 8891, "P1", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "--method" in argv
        assert argv[argv.index("--method") + 1] == "PATCH"

    def test_targets_issue_url_not_field_values_subpath(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field(7, 8891, "P2", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        url = argv[1]
        assert url.endswith("/issues/7")
        assert "issue-field-values" not in url

    def test_includes_api_version_header(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field(1, 8891, "P0", cwd="/repo")

        args, _ = mock_gh.call_args
        argv = args[0]
        assert "-H" in argv
        assert argv[argv.index("-H") + 1] == "X-GitHub-Api-Version: 2026-03-10"

    def test_request_body_contains_field_id_and_value(self):
        from orchestrator import tools as tools_mod

        captured_body = {}

        def fake_gh(argv, cwd=None):
            idx = argv.index("--input")
            tmp_path = argv[idx + 1]
            with open(tmp_path) as f:
                captured_body["data"] = json.load(f)
            return ""

        with patch.object(tools_mod, "_gh", side_effect=fake_gh):
            tools_mod.set_issue_field(10, 8894, "High", cwd="/repo")

        payload = captured_body["data"]
        assert "issue_field_values" in payload
        entry = payload["issue_field_values"][0]
        assert entry["field_id"] == 8894
        assert entry["value"] == "High"

    def test_number_cast_to_int_in_url(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field("99", 8891, "P3", cwd="/repo")  # type: ignore[arg-type]

        args, _ = mock_gh.call_args
        argv = args[0]
        assert argv[1].endswith("/issues/99")

    def test_field_id_cast_to_int_in_body(self):
        from orchestrator import tools as tools_mod

        captured_body = {}

        def fake_gh(argv, cwd=None):
            idx = argv.index("--input")
            with open(argv[idx + 1]) as f:
                captured_body["data"] = json.load(f)
            return ""

        with patch.object(tools_mod, "_gh", side_effect=fake_gh):
            tools_mod.set_issue_field(5, "8891", "P1", cwd="/repo")  # type: ignore[arg-type]

        entry = captured_body["data"]["issue_field_values"][0]
        assert entry["field_id"] == 8891
        assert isinstance(entry["field_id"], int)

    def test_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field(3, 8894, "Low", cwd="/custom/path")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom/path"

    def test_none_cwd_forwarded(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.set_issue_field(3, 8894, "Low")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] is None

    def test_returns_gh_output(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value='{"id": 42}'):
            result = tools_mod.set_issue_field(1, 8891, "P0")

        assert result == '{"id": 42}'

    def test_tempfile_cleaned_up_on_success(self):
        import os
        from orchestrator import tools as tools_mod

        seen_paths = []

        def fake_gh(argv, cwd=None):
            idx = argv.index("--input")
            seen_paths.append(argv[idx + 1])
            return ""

        with patch.object(tools_mod, "_gh", side_effect=fake_gh):
            tools_mod.set_issue_field(1, 8891, "P1")

        assert seen_paths, "fake_gh was never called"
        assert not os.path.exists(seen_paths[0])

    def test_tempfile_cleaned_up_on_gh_exception(self):
        import os
        from orchestrator import tools as tools_mod

        seen_paths = []

        def fake_gh(argv, cwd=None):
            idx = argv.index("--input")
            seen_paths.append(argv[idx + 1])
            raise RuntimeError("gh exploded")

        with patch.object(tools_mod, "_gh", side_effect=fake_gh):
            try:
                tools_mod.set_issue_field(1, 8891, "P1")
            except RuntimeError:
                pass

        assert seen_paths
        assert not os.path.exists(seen_paths[0])


class TestSetIssueFieldDispatch:

    def test_execute_routes_to_set_issue_field(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "set_issue_field", return_value="ok") as mock_fn:
            result = tools_mod.execute(
                "set_issue_field",
                {"number": 5, "field_id": 8891, "value": "P2"},
            )

        mock_fn.assert_called_once_with(5, 8891, "P2", None)
        assert result == "ok"

    def test_execute_passes_cwd(self):
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "set_issue_field", return_value="ok") as mock_fn:
            tools_mod.execute(
                "set_issue_field",
                {"number": 5, "field_id": 8894, "value": "High"},
                cwd="/repo",
            )

        mock_fn.assert_called_once_with(5, 8894, "High", "/repo")


class TestSetIssueFieldSchema:

    def _get_def(self):
        from orchestrator.tools import DEFS
        matches = [d for d in DEFS if d["function"]["name"] == "set_issue_field"]
        assert matches, "set_issue_field not found in DEFS"
        return matches[0]

    def test_present_in_defs(self):
        self._get_def()

    def test_required_params(self):
        defn = self._get_def()
        required = defn["function"]["parameters"]["required"]
        assert "number" in required
        assert "field_id" in required
        assert "value" in required

    def test_param_types(self):
        defn = self._get_def()
        props = defn["function"]["parameters"]["properties"]
        assert props["number"]["type"] == "integer"
        assert props["field_id"]["type"] == "integer"
        assert props["value"]["type"] == "string"

    def test_description_mentions_priority_and_effort(self):
        defn = self._get_def()
        desc = defn["function"]["description"]
        assert "8891" in desc
        assert "8894" in desc
        assert "Priority" in desc
        assert "Effort" in desc
