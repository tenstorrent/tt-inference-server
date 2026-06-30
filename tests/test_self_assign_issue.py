"""
Tests for issue #18: orchestrator self-assigns the GitHub issue it is fixing.

Covers:
  - parse_issue_number() extracts the correct issue number from various task strings
  - parse_issue_number() returns None when no issue number is present
  - assign_issue_if_present() calls gh with the correct argv when an issue is found
  - assign_issue_if_present() is a no-op when no issue number is in the task string
  - assign_issue_if_present() prints a warning but does NOT raise when gh fails
  - assign_issue_if_present() prints a warning but does NOT raise when gh is unavailable
  - assign_issue() in orchestrator/tools.py builds the correct gh argv (shell=False)
  - main() calls assign_issue_if_present() before orchestrate()

All external I/O (subprocess, orchestrate) is mocked so no network calls are made.
"""

import sys
import subprocess
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# parse_issue_number
# ---------------------------------------------------------------------------

class TestParseIssueNumber:
    """Unit tests for run.parse_issue_number()."""

    def _fn(self):
        from run import parse_issue_number
        return parse_issue_number

    def test_standard_format(self):
        """'Fix issue #18: ...' -> 18"""
        fn = self._fn()
        assert fn("Fix issue #18: orchestrator should self-assign") == 18

    def test_single_digit(self):
        """Issue numbers with a single digit are parsed correctly."""
        fn = self._fn()
        assert fn("Fix issue #3: typo in README") == 3

    def test_large_number(self):
        """Large issue numbers (e.g. #1234) are parsed correctly."""
        fn = self._fn()
        assert fn("Fix issue #1234: large refactor") == 1234

    def test_hash_at_start(self):
        """#N at the very start of the task string is matched."""
        fn = self._fn()
        assert fn("#42 fix the thing") == 42

    def test_hash_in_middle(self):
        """#N embedded in the middle of a sentence is matched."""
        fn = self._fn()
        assert fn("Resolve the problem described in #99 immediately") == 99

    def test_first_match_returned(self):
        """When multiple issue numbers appear, only the first is returned."""
        fn = self._fn()
        assert fn("Fix #10 and also addresses #20") == 10

    def test_no_issue_number_returns_none(self):
        """Task strings without #N return None."""
        fn = self._fn()
        assert fn("triage open issues") is None

    def test_empty_string_returns_none(self):
        fn = self._fn()
        assert fn("") is None

    def test_plain_number_no_hash_returns_none(self):
        """A bare number without the # prefix does not match."""
        fn = self._fn()
        assert fn("Fix issue 18 in the codebase") is None

    def test_returns_int_not_string(self):
        """The return type must be int, not str."""
        fn = self._fn()
        result = fn("Fix #7: something")
        assert isinstance(result, int)

    def test_groom_task_returns_none(self):
        """Typical groom-mode tasks have no issue number."""
        fn = self._fn()
        assert fn("triage open issues and apply labels") is None


# ---------------------------------------------------------------------------
# assign_issue_if_present
# ---------------------------------------------------------------------------

class TestAssignIssueIfPresent:
    """Unit tests for run.assign_issue_if_present()."""

    def _fn(self):
        from run import assign_issue_if_present
        return assign_issue_if_present

    def _make_completed_process(self, returncode=0, stdout="", stderr=""):
        cp = MagicMock(spec=subprocess.CompletedProcess)
        cp.returncode = returncode
        cp.stdout = stdout
        cp.stderr = stderr
        return cp

    # -- happy path -----------------------------------------------------------

    def test_calls_gh_with_correct_argv(self):
        """assign_issue_if_present() must invoke gh with shell=False and the
        correct argv when the task contains an issue number."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process()) as mock_run:
            fn("Fix issue #18: self-assign", "/fake/repo")

        mock_run.assert_called_once_with(
            ["gh", "issue", "edit", "18", "--add-assignee", "@me"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/fake/repo",
        )

    def test_issue_number_cast_to_str_in_argv(self):
        """The issue number in the argv list must be a string (not int)."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process()) as mock_run:
            fn("Fix issue #42: something", "/repo")

        argv = mock_run.call_args[0][0]
        # The number token in position 3 must be a str for subprocess
        assert argv[3] == "42"
        assert isinstance(argv[3], str)

    def test_shell_is_false(self):
        """shell=False must be enforced — no shell interpolation of the argv."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process()) as mock_run:
            fn("Fix #5: security fix", "/repo")

        kwargs = mock_run.call_args[1]
        assert kwargs["shell"] is False

    def test_cwd_is_repo_path(self):
        """The subprocess must be run inside repo_path so gh can infer the repo."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process()) as mock_run:
            fn("Fix #7: something", "/my/repo/path")

        assert mock_run.call_args[1]["cwd"] == "/my/repo/path"

    # -- no-op path -----------------------------------------------------------

    def test_no_call_when_no_issue_number(self):
        """When the task has no #N, subprocess.run must NOT be called."""
        fn = self._fn()

        with patch("subprocess.run") as mock_run:
            fn("triage open issues", "/repo")

        mock_run.assert_not_called()

    def test_no_call_for_empty_task(self):
        fn = self._fn()

        with patch("subprocess.run") as mock_run:
            fn("", "/repo")

        mock_run.assert_not_called()

    # -- failure resilience ---------------------------------------------------

    def test_does_not_raise_on_nonzero_exit(self, capsys):
        """A non-zero gh exit code must print a warning but NOT raise an exception."""
        fn = self._fn()

        with patch(
            "subprocess.run",
            return_value=self._make_completed_process(returncode=1, stderr="not authenticated"),
        ):
            fn("Fix #10: something", "/repo")  # must not raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "10" in captured.out

    def test_does_not_raise_on_subprocess_exception(self, capsys):
        """If subprocess.run itself raises (e.g. gh not installed), the function
        must print a warning and return without re-raising."""
        fn = self._fn()

        with patch("subprocess.run", side_effect=FileNotFoundError("gh not found")):
            fn("Fix #11: something", "/repo")  # must not raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_does_not_raise_on_timeout(self, capsys):
        """A subprocess.TimeoutExpired must be caught and warned, not re-raised."""
        fn = self._fn()

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["gh"], timeout=30),
        ):
            fn("Fix #12: something", "/repo")  # must not raise

        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    # -- output / logging -----------------------------------------------------

    def test_prints_issue_number_on_start(self, capsys):
        """A log line mentioning the issue number must be printed before the gh call."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process()):
            fn("Fix issue #99: something", "/repo")

        captured = capsys.readouterr()
        assert "99" in captured.out

    def test_prints_ok_on_success(self, capsys):
        """On a zero-exit-code result, a success line must be printed."""
        fn = self._fn()

        with patch("subprocess.run", return_value=self._make_completed_process(returncode=0, stdout="")):
            fn("Fix #7: something", "/repo")

        captured = capsys.readouterr()
        assert "7" in captured.out


# ---------------------------------------------------------------------------
# assign_issue in orchestrator/tools.py
# ---------------------------------------------------------------------------

class TestToolsAssignIssue:
    """Unit tests for orchestrator.tools.assign_issue()."""

    def test_function_exists(self):
        """assign_issue must be importable from orchestrator.tools."""
        from orchestrator.tools import assign_issue
        assert callable(assign_issue)

    def test_builds_correct_argv(self):
        """assign_issue() must call _gh with the correct positional argv."""
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.assign_issue(18, cwd="/repo")

        # _gh is always called as _gh(argv, cwd=<value>) — argv is positional,
        # cwd is a keyword argument.
        args, kwargs = mock_gh.call_args
        assert args[0] == ["issue", "edit", "18", "--add-assignee", "@me"]

    def test_number_is_cast_to_int(self):
        """assign_issue() must cast the number to int (prevents string injection)."""
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            # Passing a string "18" should still work (int("18") == 18)
            tools_mod.assign_issue("18", cwd="/repo")  # type: ignore[arg-type]

        args, kwargs = mock_gh.call_args
        assert args[0][2] == "18"  # str(int("18"))

    def test_returns_gh_output(self):
        """assign_issue() must return whatever _gh() returns."""
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="https://github.com/..."):
            result = tools_mod.assign_issue(5, cwd="/repo")

        assert result == "https://github.com/..."

    def test_cwd_forwarded_to_gh(self):
        """The cwd argument must be forwarded to _gh() as a keyword argument."""
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.assign_issue(3, cwd="/custom/path")

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] == "/custom/path"

    def test_none_cwd_forwarded(self):
        """cwd=None (the default) must be forwarded to _gh() unchanged."""
        from orchestrator import tools as tools_mod

        with patch.object(tools_mod, "_gh", return_value="") as mock_gh:
            tools_mod.assign_issue(3)

        _, kwargs = mock_gh.call_args
        assert kwargs["cwd"] is None


# ---------------------------------------------------------------------------
# main() integration: assign_issue_if_present called before orchestrate()
# ---------------------------------------------------------------------------

class TestMainCallsAssignBeforeOrchestrate:
    """Verify the ordering guarantee in run.main():
    assign_issue_if_present() is called after the git-repo check and before
    orchestrate() / orchestrate_groom()."""

    def _run_main(self, monkeypatch, task, mode="pr"):
        """Drive run.main() with mocked external calls.

        Returns the call-order log as a list of string tokens.
        """
        import run as run_mod
        import tempfile, os

        call_order = []

        def fake_assign(t, r):
            call_order.append("assign")

        def fake_orchestrate(*a, **kw):
            call_order.append("orchestrate")
            return True

        def fake_orchestrate_groom(*a, **kw):
            call_order.append("orchestrate_groom")
            return True

        monkeypatch.setattr(run_mod, "assign_issue_if_present", fake_assign)
        monkeypatch.setattr(run_mod, "orchestrate", fake_orchestrate)
        monkeypatch.setattr(run_mod, "orchestrate_groom", fake_orchestrate_groom)

        # Fake a valid git repo directory
        with tempfile.TemporaryDirectory() as tmpdir:
            git_dir = os.path.join(tmpdir, ".git")
            os.makedirs(git_dir)
            argv = ["run.py", tmpdir, task]
            if mode == "groom":
                argv = ["run.py", "--mode", "groom", tmpdir, task]
            monkeypatch.setattr(sys, "argv", argv)
            try:
                run_mod.main()
            except SystemExit:
                pass

        return call_order

    def test_assign_called_before_orchestrate(self, monkeypatch):
        """assign_issue_if_present() must appear before orchestrate() in call order."""
        call_order = self._run_main(monkeypatch, "Fix issue #18: self-assign")
        assert call_order.index("assign") < call_order.index("orchestrate")

    def test_assign_called_before_orchestrate_groom(self, monkeypatch):
        """assign_issue_if_present() must appear before orchestrate_groom() in
        groom mode as well."""
        call_order = self._run_main(monkeypatch, "Fix issue #18: groom task", mode="groom")
        assert call_order.index("assign") < call_order.index("orchestrate_groom")

    def test_assign_receives_task_string(self, monkeypatch):
        """assign_issue_if_present() must be called with the exact task string
        that was passed on the CLI."""
        import run as run_mod
        import tempfile, os

        received_task = []

        def fake_assign(t, r):
            received_task.append(t)

        monkeypatch.setattr(run_mod, "assign_issue_if_present", fake_assign)
        monkeypatch.setattr(run_mod, "orchestrate", lambda *a, **kw: True)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".git"))
            task = "Fix issue #77: test task"
            monkeypatch.setattr(sys, "argv", ["run.py", tmpdir, task])
            try:
                run_mod.main()
            except SystemExit:
                pass

        assert received_task == [task]

    def test_assign_receives_repo_path(self, monkeypatch):
        """assign_issue_if_present() must be called with the absolute repo path."""
        import run as run_mod
        import tempfile, os

        received_repo = []

        def fake_assign(t, r):
            received_repo.append(r)

        monkeypatch.setattr(run_mod, "assign_issue_if_present", fake_assign)
        monkeypatch.setattr(run_mod, "orchestrate", lambda *a, **kw: True)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, ".git"))
            monkeypatch.setattr(sys, "argv", ["run.py", tmpdir, "Fix #1: task"])
            try:
                run_mod.main()
            except SystemExit:
                pass

        assert len(received_repo) == 1
        assert os.path.isabs(received_repo[0])
