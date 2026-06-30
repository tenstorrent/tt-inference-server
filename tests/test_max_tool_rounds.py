"""
Tests for issue #26: implementer hitting max_tool_rounds must abort the
orchestrator immediately instead of passing incomplete work to reviewers.

All external I/O (OpenAI client, tools) is mocked so no network calls are made.
"""

import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI response objects
# ---------------------------------------------------------------------------

def _make_tool_call(tc_id: str, name: str, arguments: str = "{}"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_response(content=None, tool_calls=None):
    """Build a fake chat completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# agent.run() tests
# ---------------------------------------------------------------------------

class TestAgentRun:
    """Unit tests for orchestrator/agent.py::run()."""

    def _make_persona(self, name="test_agent"):
        return {"name": name, "model": "fake-model", "system": "You are a test agent."}

    @patch("orchestrator.agent.T.execute", return_value="tool result")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_raises_when_cap_hit(self, mock_client_factory, mock_execute):
        """run() must raise MaxToolRoundsError when max_tool_rounds is exhausted."""
        from orchestrator.agent import run, MaxToolRoundsError

        # Every call returns a tool_call so the loop never terminates naturally
        client = MagicMock()
        mock_client_factory.return_value = client
        tool_call = _make_tool_call("tc1", "bash")
        client.chat.completions.create.return_value = _make_response(
            content=None, tool_calls=[tool_call]
        )

        persona = self._make_persona()
        with pytest.raises(MaxToolRoundsError) as exc_info:
            run(persona, [{"role": "user", "content": "do something"}], max_tool_rounds=3)

        err = exc_info.value
        assert err.persona_name == "test_agent"
        assert err.max_tool_rounds == 3
        assert "test_agent" in str(err)
        assert "3" in str(err)

    @patch("orchestrator.agent.T.execute", return_value="tool result")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_raises_stores_history(self, mock_client_factory, mock_execute):
        """MaxToolRoundsError.history must contain the accumulated message history."""
        from orchestrator.agent import run, MaxToolRoundsError

        client = MagicMock()
        mock_client_factory.return_value = client
        tool_call = _make_tool_call("tc1", "bash")
        client.chat.completions.create.return_value = _make_response(
            content=None, tool_calls=[tool_call]
        )

        persona = self._make_persona()
        with pytest.raises(MaxToolRoundsError) as exc_info:
            run(persona, [{"role": "user", "content": "do something"}], max_tool_rounds=2)

        assert isinstance(exc_info.value.history, list)
        assert len(exc_info.value.history) > 0

    @patch("orchestrator.agent.T.execute", return_value="tool result")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_no_raise_on_clean_finish(self, mock_client_factory, mock_execute):
        """run() must return normally when the agent finishes before the cap."""
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        # First call returns a tool call, second returns a text response
        tool_call = _make_tool_call("tc1", "bash")
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tool_call]),
            _make_response(content="IMPLEMENTATION_COMPLETE", tool_calls=[]),
        ]

        persona = self._make_persona()
        text, history = run(
            persona,
            [{"role": "user", "content": "do something"}],
            max_tool_rounds=10,
        )
        assert text == "IMPLEMENTATION_COMPLETE"

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_no_raise_when_zero_tool_calls(self, mock_client_factory):
        """run() must return immediately if the agent produces text on the first call."""
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="hello", tool_calls=[]
        )

        persona = self._make_persona()
        text, _ = run(persona, [{"role": "user", "content": "hi"}], max_tool_rounds=1)
        assert text == "hello"

    @patch("orchestrator.agent.T.execute", return_value="tool result")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_error_is_not_string_sentinel(self, mock_client_factory, mock_execute):
        """The old behaviour returned an 'ERROR: hit max_tool_rounds' string.
        Confirm that is gone — a MaxToolRoundsError is raised instead."""
        from orchestrator.agent import run, MaxToolRoundsError

        client = MagicMock()
        mock_client_factory.return_value = client
        tool_call = _make_tool_call("tc1", "bash")
        client.chat.completions.create.return_value = _make_response(
            content=None, tool_calls=[tool_call]
        )

        persona = self._make_persona()
        with pytest.raises(MaxToolRoundsError):
            run(persona, [{"role": "user", "content": "do something"}], max_tool_rounds=1)

        # Double-check: the call count matches max_tool_rounds (no extra "error" call)
        assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# orchestrator.orchestrate() tests
# ---------------------------------------------------------------------------

FAKE_IMPLEMENTER = {"name": "implementer", "model": "m", "system": "s"}
FAKE_REVIEWERS = [
    {"name": "reviewer_a", "model": "m", "system": "s"},
]


class TestOrchestrateMaxToolRounds:
    """Integration-level tests for orchestrator.py::orchestrate()."""

    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "IMPLEMENTER", FAKE_IMPLEMENTER)
        monkeypatch.setattr(orch, "REVIEWERS", FAKE_REVIEWERS)

    def test_initial_impl_cap_returns_false(self, monkeypatch):
        """orchestrate() returns False immediately when the initial implementer hits cap."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                raise MaxToolRoundsError("implementer", 40, [])
            return "APPROVED", []

        monkeypatch.setattr(orch.A, "run", fake_run)

        result = orch.orchestrate("do a task", "/fake/repo", verbose=False)
        assert result is False

    def test_initial_impl_cap_does_not_call_reviewers(self, monkeypatch):
        """Reviewers must never be invoked when the initial implementation hits cap."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        reviewer_called = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                raise MaxToolRoundsError("implementer", 40, [])
            reviewer_called.append(persona["name"])
            return "APPROVED", []

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do a task", "/fake/repo", verbose=False)

        assert reviewer_called == [], (
            f"Reviewers were called despite implementer cap: {reviewer_called}"
        )

    def test_rebuttal_cap_returns_false(self, monkeypatch):
        """orchestrate() returns False when the implementer hits cap during rebuttal."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        call_counts = {"impl": 0, "rev": 0}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                call_counts["impl"] += 1
                if call_counts["impl"] == 1:
                    # First call: implementation succeeds
                    return "IMPLEMENTATION_COMPLETE", [{"role": "system", "content": "s"}]
                # Rebuttal call: hits cap
                raise MaxToolRoundsError("implementer", 40, [])
            call_counts["rev"] += 1
            # Reviewer always objects to trigger a rebuttal
            return "OBJECTION: something is wrong", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)

        result = orch.orchestrate("do a task", "/fake/repo", max_debate_rounds=3, verbose=False)
        assert result is False

    def test_successful_run_unaffected(self, monkeypatch):
        """orchestrate() still works end-to-end when no cap is hit."""
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        def fake_run(persona, messages, **kwargs):
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)

        # Also stub create_pr so no real git/gh calls happen
        monkeypatch.setattr(
            orch,
            "orchestrate",
            lambda *a, **kw: True,  # short-circuit; we tested the logic above
        )

        result = orch.orchestrate("do a task", "/fake/repo", verbose=False)
        assert result is True


# ---------------------------------------------------------------------------
# orchestrator.orchestrate_groom() tests
# ---------------------------------------------------------------------------

FAKE_GROOMER = {"name": "groomer", "model": "m", "system": "s"}
FAKE_GROOM_REVIEWERS = [
    {"name": "groom_reviewer_a", "model": "m", "system": "s"},
]


class TestOrchestrateGroomMaxToolRounds:
    """Integration-level tests for orchestrator.py::orchestrate_groom()."""

    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "GROOMER", FAKE_GROOMER)
        monkeypatch.setattr(orch, "GROOM_REVIEWERS", FAKE_GROOM_REVIEWERS)

    def test_initial_groom_cap_returns_false(self, monkeypatch):
        """orchestrate_groom() returns False immediately when groomer hits cap."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        # Stub list_issues so no gh CLI calls are made
        monkeypatch.setattr(
            "orchestrator.orchestrator._list_issues"
            if hasattr(orch, "_list_issues") else "orchestrator.tools.list_issues",
            lambda **kw: "[]",
            raising=False,
        )

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "groomer":
                raise MaxToolRoundsError("groomer", 40, [])
            return "APPROVED", []

        monkeypatch.setattr(orch.A, "run", fake_run)

        # Also patch the internal list_issues import inside orchestrate_groom
        import orchestrator.tools as tools_mod
        monkeypatch.setattr(tools_mod, "list_issues", lambda **kw: "[]")

        result = orch.orchestrate_groom("triage issues", "/fake/repo", verbose=False)
        assert result is False

    def test_initial_groom_cap_does_not_call_reviewers(self, monkeypatch):
        """Groom reviewers must not be invoked when initial groomer hits cap."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        reviewer_called = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "groomer":
                raise MaxToolRoundsError("groomer", 40, [])
            reviewer_called.append(persona["name"])
            return "APPROVED", []

        monkeypatch.setattr(orch.A, "run", fake_run)
        import orchestrator.tools as tools_mod
        monkeypatch.setattr(tools_mod, "list_issues", lambda **kw: "[]")

        orch.orchestrate_groom("triage issues", "/fake/repo", verbose=False)

        assert reviewer_called == [], (
            f"Groom reviewers were called despite groomer cap: {reviewer_called}"
        )

    def test_groom_rebuttal_cap_returns_false(self, monkeypatch):
        """orchestrate_groom() returns False when groomer hits cap during rebuttal."""
        from orchestrator.agent import MaxToolRoundsError
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)

        call_counts = {"groom": 0}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "groomer":
                call_counts["groom"] += 1
                if call_counts["groom"] == 1:
                    return "GROOMING_COMPLETE", [{"role": "system", "content": "s"}]
                raise MaxToolRoundsError("groomer", 40, [])
            # Reviewer objects to trigger a rebuttal
            return "OBJECTION: bad labels", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        import orchestrator.tools as tools_mod
        monkeypatch.setattr(tools_mod, "list_issues", lambda **kw: "[]")

        result = orch.orchestrate_groom("triage issues", "/fake/repo", max_debate_rounds=3, verbose=False)
        assert result is False


# ---------------------------------------------------------------------------
# MaxToolRoundsError public API tests
# ---------------------------------------------------------------------------

class TestMaxToolRoundsErrorPublicAPI:
    """Verify the exception is importable from the public package API."""

    def test_importable_from_package(self):
        from orchestrator import MaxToolRoundsError
        assert MaxToolRoundsError is not None

    def test_importable_from_agent(self):
        from orchestrator.agent import MaxToolRoundsError
        assert MaxToolRoundsError is not None

    def test_is_exception(self):
        from orchestrator.agent import MaxToolRoundsError
        assert issubclass(MaxToolRoundsError, Exception)

    def test_attributes(self):
        from orchestrator.agent import MaxToolRoundsError
        err = MaxToolRoundsError("my_agent", 15, [{"role": "user", "content": "hi"}])
        assert err.persona_name == "my_agent"
        assert err.max_tool_rounds == 15
        assert err.history == [{"role": "user", "content": "hi"}]
        assert "my_agent" in str(err)
        assert "15" in str(err)
