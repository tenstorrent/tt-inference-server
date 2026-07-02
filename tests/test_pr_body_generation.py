"""
Tests for issue #154: PR body generation via one-shot LLM call.

Verifies that:
  - generate_pr_body() uses the implementer's model and provider (not hardcoded)
  - generate_pr_body() strips <think> blocks from the response
  - generate_pr_body() passes the task and diff to the model
  - _get_diff() returns the git diff or empty string on failure
  - orchestrate() calls generate_pr_body() with the implementer persona after consensus
  - orchestrate() calls generate_pr_body() with an overridden implementer persona
  - The IMPLEMENTATION_COMPLETE sentinel never leaks into the PR body
"""

import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


_FAKE_PR_BODY = (
    "## Summary\nDid the thing.\n\n"
    "## Changes\n- foo.py: changed bar\n\n"
    "## Testing\nRan tests.\n\n"
    "## Fixes\nFixes #154"
)

_SYS_MSG = [{"role": "system", "content": "s"}]


# ---------------------------------------------------------------------------
# generate_pr_body() — unit tests
# ---------------------------------------------------------------------------

class TestGeneratePrBody:

    @patch("orchestrator.agent._client")
    def test_uses_implementer_model(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        implementer = {"name": "implementer", "model": "my-special-model", "provider": "litellm", "system": "s"}
        generate_pr_body(implementer, "task", "diff")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-special-model"

    @patch("orchestrator.agent._client")
    def test_uses_implementer_provider(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        implementer = {"name": "implementer", "model": "m", "provider": "tt-console", "system": "s"}
        generate_pr_body(implementer, "task", "diff")

        # _client must have been called with the implementer's provider.
        mock_client_factory.assert_called_once_with("tt-console", None)

    @patch("orchestrator.agent._client")
    def test_passes_task_to_model(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                         "Fix issue #99: do the thing", "some diff")

        messages = client.chat.completions.create.call_args[1]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "Fix issue #99: do the thing" in user_content

    @patch("orchestrator.agent._client")
    def test_passes_diff_to_model(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                         "task", "diff: +new line\n-old line")

        messages = client.chat.completions.create.call_args[1]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        assert "diff: +new line" in user_content

    @patch("orchestrator.agent._client")
    def test_returns_stripped_text(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response("  " + _FAKE_PR_BODY + "  ")

        result = generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                                  "task", "diff")
        assert result == _FAKE_PR_BODY

    @patch("orchestrator.agent._client")
    def test_strips_think_blocks(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        # Simulate a reasoning model that leaks think-block content.
        client.chat.completions.create.return_value = _make_response(
            "<think>I see == 'ok'. There are two newlines before def...</think>" + _FAKE_PR_BODY
        )

        result = generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                                  "task", "diff")
        assert "<think>" not in result
        assert "I see ==" not in result
        assert "## Summary" in result

    @patch("orchestrator.agent._client")
    def test_does_not_send_tools(self, mock_client_factory):
        """generate_pr_body() must be a plain completion with no tool definitions."""
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                         "task", "diff")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert "tools" not in call_kwargs

    @patch("orchestrator.agent._client")
    def test_respects_max_tokens_in_persona(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        implementer = {"name": "i", "model": "m", "provider": "litellm", "system": "s", "max_tokens": 1234}
        generate_pr_body(implementer, "task", "diff")

        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs.get("max_tokens") == 1234

    @patch("orchestrator.agent._client")
    def test_truncates_large_diff(self, mock_client_factory):
        from orchestrator.agent import generate_pr_body

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(_FAKE_PR_BODY)

        huge_diff = "x" * 20_000
        generate_pr_body({"name": "i", "model": "m", "provider": "litellm", "system": "s"},
                         "task", huge_diff)

        messages = client.chat.completions.create.call_args[1]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        # Diff must be truncated; the total user message must be well under 20k + overhead.
        assert len(user_content) < 15_000
        assert "[diff truncated]" in user_content


# ---------------------------------------------------------------------------
# _get_diff() — unit tests
# ---------------------------------------------------------------------------

class TestGetDiff:
    def test_returns_stdout_on_success(self, tmp_path):
        from orchestrator.orchestrator import _get_diff
        import subprocess

        with patch("orchestrator.orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="diff content here")
            result = _get_diff(str(tmp_path))

        assert result == "diff content here"
        mock_run.assert_called_once_with(
            ["git", "diff", "main..HEAD"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(tmp_path),
        )

    def test_returns_empty_string_on_nonzero_exit(self, tmp_path):
        from orchestrator.orchestrator import _get_diff

        with patch("orchestrator.orchestrator.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _get_diff(str(tmp_path))

        assert result == ""

    def test_returns_empty_string_on_exception(self, tmp_path):
        from orchestrator.orchestrator import _get_diff

        with patch("orchestrator.orchestrator.subprocess.run", side_effect=Exception("no git")):
            result = _get_diff(str(tmp_path))

        assert result == ""


# ---------------------------------------------------------------------------
# orchestrate() integration: generate_pr_body is called with the right args
# ---------------------------------------------------------------------------

class TestOrchestrateCallsGeneratePrBody:

    def _run_orchestrate(self, task, captured_gpb, fake_implementer=None):
        import orchestrator.orchestrator as orch_mod

        def fake_agent_run(persona, messages, **kwargs):
            return "APPROVED", [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ]

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            captured_gpb.update({"implementer": implementer, "task": task, "diff": diff})
            return _FAKE_PR_BODY

        def fake_create_pr(title, body, branch, cwd=None):
            return "https://github.com/org/repo/pull/1"

        monkeypatch_kwargs = {}
        if fake_implementer:
            monkeypatch_kwargs["implementer_override"] = fake_implementer

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
            patch("orchestrator.orchestrator._get_diff", return_value="mocked diff"),
        ):
            orch_mod.orchestrate(
                task,
                repo_path="/fake/repo",
                verbose=False,
                **monkeypatch_kwargs,
            )

    def test_generate_pr_body_called_with_implementer_persona(self):
        captured = {}
        self._run_orchestrate("Fix issue #154: test", captured)
        assert captured.get("implementer") is not None
        assert captured["implementer"]["name"] == "implementer"

    def test_generate_pr_body_called_with_task(self):
        captured = {}
        self._run_orchestrate("Fix issue #154: test", captured)
        assert captured.get("task") == "Fix issue #154: test"

    def test_generate_pr_body_called_with_diff(self):
        captured = {}
        self._run_orchestrate("Fix issue #154: test", captured)
        assert captured.get("diff") == "mocked diff"

    def test_generate_pr_body_uses_overridden_implementer_model(self):
        """When implementer_override changes the model, generate_pr_body must
        receive the overridden persona (not the default IMPLEMENTER)."""
        import orchestrator.orchestrator as orch_mod

        captured = {}

        def fake_agent_run(persona, messages, **kwargs):
            return "APPROVED", _SYS_MSG[:]

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            captured["model"] = implementer["model"]
            return _FAKE_PR_BODY

        def fake_create_pr(title, body, branch, cwd=None):
            return "https://github.com/org/repo/pull/1"

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
            patch("orchestrator.orchestrator._get_diff", return_value="diff"),
        ):
            orch_mod.orchestrate(
                "Fix #154: test",
                repo_path="/fake/repo",
                verbose=False,
                implementer_override={"model": "custom-kimi-model"},
            )

        assert captured.get("model") == "custom-kimi-model"

    def test_implementation_complete_not_in_pr_body(self):
        """IMPLEMENTATION_COMPLETE must never appear in the PR body."""
        import orchestrator.orchestrator as orch_mod

        pr_body_seen = {}

        def fake_agent_run(persona, messages, **kwargs):
            return "IMPLEMENTATION_COMPLETE", [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "IMPLEMENTATION_COMPLETE"},
            ]

        def fake_generate_pr_body(implementer, task, diff, api_key=None):
            return _FAKE_PR_BODY

        def fake_create_pr(title, body, branch, cwd=None):
            pr_body_seen["body"] = body
            return "https://github.com/org/repo/pull/1"

        with (
            patch.object(orch_mod.A, "run", side_effect=fake_agent_run),
            patch.object(orch_mod.A, "generate_pr_body", side_effect=fake_generate_pr_body),
            patch("orchestrator.tools.create_pr", side_effect=fake_create_pr),
            patch("orchestrator.orchestrator._get_diff", return_value="diff"),
        ):
            orch_mod.orchestrate("Fix #154: test", repo_path="/fake/repo", verbose=False)

        assert "IMPLEMENTATION_COMPLETE" not in pr_body_seen.get("body", "")
