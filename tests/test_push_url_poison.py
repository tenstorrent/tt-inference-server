"""
Tests for issue #62: push-URL poisoning in run.py and restore in create_pr.

Covers:
  - run.py sets push URL to DISABLED after repo validation
  - create_pr() restores the real push URL before git push
  - create_pr() returns a formatted error string (not an exception) when
    git remote get-url fails (no origin, git not on PATH, etc.)
  - create_pr() returns a formatted error string when set-url fails
  - The push URL seen by the git push step is the real URL, not DISABLED
"""

import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/example/repo.git"],
        cwd=repo, check=True,
    )
    return str(repo)


def _get_push_url(repo_path):
    return subprocess.check_output(
        ["git", "remote", "get-url", "--push", "origin"],
        cwd=repo_path,
    ).decode().strip()


def _make_poisoned_repo(tmp_path):
    repo = _make_repo(tmp_path)
    subprocess.run(
        ["git", "remote", "set-url", "--push", "origin", "DISABLED"],
        cwd=repo, check=True,
    )
    (tmp_path / "repo" / "f.txt").write_text("hello")
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


# ---------------------------------------------------------------------------
# run.py: poison step
# ---------------------------------------------------------------------------

class TestRunPyPoisonsPushUrl:

    def test_push_url_set_to_disabled(self, tmp_path):
        """After main() validates the repo, the push URL must be DISABLED."""
        repo = _make_repo(tmp_path)

        with patch("run.assign_issue_if_present"), \
             patch("run.orchestrate", return_value=True), \
             patch("sys.argv", ["run.py", repo, "Fix #1: test"]):
            import run
            with pytest.raises(SystemExit):
                run.main()

        assert _get_push_url(repo) == "DISABLED"

    def test_poison_uses_shell_false(self, tmp_path):
        """The subprocess.run call that poisons the URL must use shell=False."""
        repo = _make_repo(tmp_path)

        poison_calls = []
        real_run = subprocess.run

        def capturing_run(argv, **kwargs):
            if argv == ["git", "remote", "set-url", "--push", "origin", "DISABLED"]:
                poison_calls.append(kwargs)
            return real_run(argv, **kwargs)

        with patch("subprocess.run", side_effect=capturing_run), \
             patch("run.assign_issue_if_present"), \
             patch("run.orchestrate", return_value=True), \
             patch("sys.argv", ["run.py", repo, "Fix #1: test"]):
            import run
            with pytest.raises(SystemExit):
                run.main()

        assert len(poison_calls) == 1
        assert poison_calls[0].get("shell", False) is False


# ---------------------------------------------------------------------------
# create_pr: restore step
# Patch orchestrator.tools.subprocess.run so the intercept reaches _run().
# ---------------------------------------------------------------------------

class TestCreatePrRestoresPushUrl:

    def test_push_url_restored_before_git_push(self, tmp_path):
        """create_pr must restore the push URL to the real origin URL before pushing."""
        repo = _make_poisoned_repo(tmp_path)
        assert _get_push_url(repo) == "DISABLED"

        push_url_at_push_time = []
        import orchestrator.tools as tools_mod
        real_run = tools_mod.subprocess.run

        def capturing_run(argv, **kwargs):
            if len(argv) >= 2 and argv[:2] == ["git", "push"]:
                push_url_at_push_time.append(_get_push_url(repo))
                r = MagicMock()
                r.returncode = 1
                r.stdout = ""
                r.stderr = "push blocked by test"
                return r
            return real_run(argv, **kwargs)

        from orchestrator.tools import create_pr
        with patch.object(tools_mod.subprocess, "run", side_effect=capturing_run):
            create_pr("test title", "body", "ai/test-branch", cwd=repo)

        assert len(push_url_at_push_time) == 1
        assert push_url_at_push_time[0] == "https://github.com/example/repo.git"

    def test_get_url_failure_returns_error_string_not_exception(self, tmp_path):
        """When git remote get-url fails, create_pr must return an error string."""
        repo = _make_poisoned_repo(tmp_path)

        import orchestrator.tools as tools_mod
        real_run = tools_mod.subprocess.run

        def failing_get_url(argv, **kwargs):
            if argv[:3] == ["git", "remote", "get-url"]:
                r = MagicMock()
                r.returncode = 128
                r.stdout = ""
                r.stderr = "fatal: No such remote 'origin'"
                return r
            return real_run(argv, **kwargs)

        from orchestrator.tools import create_pr
        with patch.object(tools_mod.subprocess, "run", side_effect=failing_get_url):
            result = create_pr("title", "body", "ai/branch", cwd=repo)

        assert isinstance(result, str)
        assert "exit code" in result
        assert "128" in result

    def test_set_url_failure_returns_error_string_not_exception(self, tmp_path):
        """When git remote set-url fails, create_pr must return an error string."""
        repo = _make_poisoned_repo(tmp_path)

        import orchestrator.tools as tools_mod
        real_run = tools_mod.subprocess.run

        def failing_set_url(argv, **kwargs):
            if argv[:4] == ["git", "remote", "set-url", "--push"]:
                r = MagicMock()
                r.returncode = 1
                r.stdout = ""
                r.stderr = "error: set-url denied"
                return r
            return real_run(argv, **kwargs)

        from orchestrator.tools import create_pr
        with patch.object(tools_mod.subprocess, "run", side_effect=failing_set_url):
            result = create_pr("title", "body", "ai/branch", cwd=repo)

        assert isinstance(result, str)
        assert "exit code" in result

    def test_get_url_failure_does_not_proceed_to_push(self, tmp_path):
        """When get-url fails, create_pr must not attempt git push."""
        repo = _make_poisoned_repo(tmp_path)

        push_attempted = []
        import orchestrator.tools as tools_mod
        real_run = tools_mod.subprocess.run

        def tracking_run(argv, **kwargs):
            if argv[:3] == ["git", "remote", "get-url"]:
                r = MagicMock()
                r.returncode = 128
                r.stdout = ""
                r.stderr = "fatal: No such remote"
                return r
            if len(argv) >= 2 and argv[:2] == ["git", "push"]:
                push_attempted.append(argv)
            return real_run(argv, **kwargs)

        from orchestrator.tools import create_pr
        with patch.object(tools_mod.subprocess, "run", side_effect=tracking_run):
            create_pr("title", "body", "ai/branch", cwd=repo)

        assert push_attempted == []
