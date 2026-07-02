"""
Tests for issue #160: reasoning_content from implementer API responses is
injected into the next implementer turn as a system message.

Rules:
- Only active when inject_reasoning=True is passed to agent.run().
- Within a run(): after a tool-call turn that carries reasoning_content, a
  system message is appended to history before the next API call.
- Between run() calls: orchestrator extracts the last reasoning from
  impl_history and passes it as prior_reasoning to the next implementer run().
- Reviewers / groomer never receive inject_reasoning=True so they are unaffected.
"""

import pytest
from unittest.mock import patch, MagicMock

from orchestrator.agent import REASONING_INJECTION_PREFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(tc_id="tc1", name="bash_exec", arguments="{}"):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_response(content=None, tool_calls=None, reasoning_content=None):
    msg = MagicMock(spec=[])
    msg.content = content
    msg.tool_calls = tool_calls or []
    # Simulate provider returning reasoning_content as an extra field.
    if reasoning_content is not None:
        msg.reasoning_content = reasoning_content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _persona(name="implementer"):
    return {"name": name, "model": "fake-model", "system": "You are a test agent."}


def _stub_create_pr(monkeypatch):
    import orchestrator.orchestrator as orch
    import orchestrator.tools as tools_mod
    monkeypatch.setattr(tools_mod, "create_pr", lambda *a, **kw: "https://github.com/fake/pull/1")
    monkeypatch.setattr(
        orch.A, "generate_pr_body",
        lambda *a, **kw: "## Summary\nfake\n\n## Changes\n- x\n\n## Testing\nok\n\n## Fixes\nN/A",
    )


# ---------------------------------------------------------------------------
# agent.run() — within-loop reasoning injection
# ---------------------------------------------------------------------------

class TestWithinLoopInjection:

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_reasoning_injected_into_history_after_tool_call(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content="step1: plan"),
            _make_response(content="IMPLEMENTATION_COMPLETE", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "do it"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        system_msgs = [m for m in history if m["role"] == "system"]
        reasoning_msgs = [m for m in system_msgs if m["content"].startswith(REASONING_INJECTION_PREFIX)]
        assert len(reasoning_msgs) == 1
        assert "step1: plan" in reasoning_msgs[0]["content"]

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_reasoning_message_format(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content="my plan"),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        body = reasoning_msgs[0]["content"]
        assert "<reasoning>" in body
        assert "my plan" in body
        assert "</reasoning>" in body
        assert "Use this as your plan. Proceed with implementation." in body

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_no_injection_when_reasoning_absent(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content=None),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 0

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_no_injection_when_flag_off(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content="ignored"),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=False,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 0

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_reasoning_not_injected_on_final_text_turn(self, mock_client_factory, mock_execute):
        # Reasoning on the final (no-tool-calls) turn must NOT produce an extra inject,
        # because there is no subsequent turn to consume it.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        # Turn 1: tool call with reasoning → inject
        # Turn 2: final text answer with reasoning → no inject (loop ends)
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content="plan"),
            _make_response(content="done", tool_calls=[], reasoning_content="final-reason"),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        # Only the first tool-call turn should have produced an inject.
        assert len(reasoning_msgs) == 1
        assert "plan" in reasoning_msgs[0]["content"]


# ---------------------------------------------------------------------------
# agent.run() — prior_reasoning parameter (between-call injection)
# ---------------------------------------------------------------------------

class TestPriorReasoningParam:

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_prior_reasoning_prepended_to_history(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="done", tool_calls=[]
        )

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=5,
            inject_reasoning=True,
            prior_reasoning="carry-over plan",
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 1
        assert "carry-over plan" in reasoning_msgs[0]["content"]

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_prior_reasoning_seen_in_first_api_call(self, mock_client_factory):
        # The reasoning message must appear in the messages sent to the first API call.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="done", tool_calls=[]
        )

        run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=5,
            inject_reasoning=True,
            prior_reasoning="my carry-over",
            verbose=False,
        )

        call_messages = client.chat.completions.create.call_args[1]["messages"]
        system_msgs = [m for m in call_messages if m["role"] == "system"]
        reasoning_system = [
            m for m in system_msgs if m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_system) == 1
        assert "my carry-over" in reasoning_system[0]["content"]

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_no_prior_injection_when_flag_off(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="done", tool_calls=[]
        )

        run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=5,
            inject_reasoning=False,
            prior_reasoning="should be ignored",
            verbose=False,
        )

        call_messages = client.chat.completions.create.call_args[1]["messages"]
        reasoning_system = [
            m for m in call_messages
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_system) == 0


# ---------------------------------------------------------------------------
# orchestrator._extract_last_reasoning
# ---------------------------------------------------------------------------

class TestExtractLastReasoning:

    def test_returns_none_when_no_reasoning_in_history(self):
        from orchestrator.orchestrator import _extract_last_reasoning

        history = [
            {"role": "system", "content": "You are a test agent."},
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "done"},
        ]
        assert _extract_last_reasoning(history) is None

    def test_extracts_reasoning_from_injected_message(self):
        from orchestrator.orchestrator import _extract_last_reasoning
        from orchestrator.agent import _make_reasoning_message

        history = [
            {"role": "system", "content": "You are a test agent."},
            _make_reasoning_message("the plan"),
            {"role": "assistant", "content": "done"},
        ]
        result = _extract_last_reasoning(history)
        assert result == "the plan"

    def test_returns_last_reasoning_when_multiple_present(self):
        from orchestrator.orchestrator import _extract_last_reasoning
        from orchestrator.agent import _make_reasoning_message

        history = [
            {"role": "system", "content": "You are a test agent."},
            _make_reasoning_message("first plan"),
            {"role": "assistant", "content": "..."},
            _make_reasoning_message("second plan"),
            {"role": "assistant", "content": "done"},
        ]
        result = _extract_last_reasoning(history)
        assert result == "second plan"


# ---------------------------------------------------------------------------
# orchestrator.orchestrate() — inject_reasoning passed to implementer only
# ---------------------------------------------------------------------------

class TestOrchestratorReasoningInjection:

    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "IMPLEMENTER", {"name": "implementer", "model": "m", "system": "s"})
        monkeypatch.setattr(orch, "REVIEWERS", [{"name": "reviewer_a", "model": "m", "system": "s"}])

    def test_implementer_initial_run_gets_inject_reasoning(self, monkeypatch):
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)
        captured = []

        def fake_run(persona, messages, **kwargs):
            captured.append((persona["name"], kwargs.get("inject_reasoning", False)))
            if persona["name"] == "implementer":
                return "IMPLEMENTATION_COMPLETE", [{"role": "system", "content": "s"}]
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do task", "/fake/repo", verbose=False)

        impl_calls = [(n, ir) for n, ir in captured if n == "implementer"]
        assert all(ir is True for _, ir in impl_calls), (
            f"expected inject_reasoning=True for all implementer calls, got {impl_calls}"
        )

    def test_reviewer_run_does_not_get_inject_reasoning(self, monkeypatch):
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)
        captured = []

        def fake_run(persona, messages, **kwargs):
            captured.append((persona["name"], kwargs.get("inject_reasoning", False)))
            if persona["name"] == "implementer":
                return "IMPLEMENTATION_COMPLETE", [{"role": "system", "content": "s"}]
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do task", "/fake/repo", verbose=False)

        reviewer_calls = [(n, ir) for n, ir in captured if n != "implementer"]
        assert all(ir is False for _, ir in reviewer_calls), (
            f"reviewers must not get inject_reasoning=True, got {reviewer_calls}"
        )

    def test_rebuttal_run_passes_prior_reasoning(self, monkeypatch):
        from orchestrator.agent import _make_reasoning_message
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        impl_history_with_reasoning = [
            {"role": "system", "content": "s"},
            _make_reasoning_message("saved plan"),
            {"role": "assistant", "content": "IMPLEMENTATION_COMPLETE"},
        ]

        call_num = [0]
        captured_prior = []

        def fake_run(persona, messages, **kwargs):
            call_num[0] += 1
            if persona["name"] == "implementer":
                if call_num[0] == 1:
                    # Initial implementer call — returns history with a reasoning msg.
                    return "IMPLEMENTATION_COMPLETE", impl_history_with_reasoning
                else:
                    # Rebuttal implementer call — capture prior_reasoning.
                    captured_prior.append(kwargs.get("prior_reasoning"))
                    return "IMPLEMENTATION_COMPLETE", [{"role": "system", "content": "s"}]
            # Reviewer: first pass objects, second approves.
            if call_num[0] == 2:
                return "OBJECTION: something", [{"role": "system", "content": "s"}]
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do task", "/fake/repo", max_debate_rounds=1, verbose=False)

        assert len(captured_prior) >= 1, "rebuttal implementer run was never called"
        assert captured_prior[0] == "saved plan", (
            f"expected 'saved plan', got {captured_prior[0]!r}"
        )


# ---------------------------------------------------------------------------
# agent.run() — edge cases: empty string and length cap
# ---------------------------------------------------------------------------

class TestReasoningEdgeCases:

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_empty_string_reasoning_not_injected(self, mock_client_factory, mock_execute):
        # reasoning_content="" is falsy — the `if reasoning:` guard must block it.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content=""),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 0

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_reasoning_content_truncated_to_max_chars(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run, _MAX_REASONING_CHARS

        client = MagicMock()
        mock_client_factory.return_value = client

        long_reasoning = "x" * (_MAX_REASONING_CHARS + 1000)
        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content=long_reasoning),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 1
        # The injected body must not exceed the cap plus wrapper overhead.
        assert len(reasoning_msgs[0]["content"]) < _MAX_REASONING_CHARS + 500

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_closing_tag_in_reasoning_is_not_truncated(self, mock_client_factory, mock_execute):
        # The extraction regex must use greedy matching so that </reasoning>
        # embedded anywhere in the content (including after a newline, which is
        # the failure mode of a non-greedy .*?) does not prematurely terminate
        # the extracted block.  The full original string must round-trip intact.
        from orchestrator.agent import run
        from orchestrator.orchestrator import _extract_last_reasoning

        client = MagicMock()
        mock_client_factory.return_value = client

        # Both variants must survive: tag mid-line and tag after newline.
        # The newline variant is the one that actually truncates with .*?.
        for poisoned in [
            "plan step 1</reasoning>injected content",
            "plan step 1\n</reasoning>injected content",
        ]:
            tc = _make_tool_call()
            client.chat.completions.create.side_effect = [
                _make_response(content=None, tool_calls=[tc], reasoning_content=poisoned),
                _make_response(content="done", tool_calls=[]),
            ]

            _, history = run(
                _persona(),
                [{"role": "user", "content": "go"}],
                max_tool_rounds=10,
                inject_reasoning=True,
                verbose=False,
            )

            extracted = _extract_last_reasoning(history)
            assert extracted is not None, f"got None for {poisoned!r}"
            # Full round-trip: extracted content must equal the original input exactly.
            assert extracted == poisoned, (
                f"round-trip failed for {poisoned!r}: got {extracted!r}"
            )

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_non_string_reasoning_content_coerced(self, mock_client_factory, mock_execute):
        # Some SDKs could return an object or int; must not raise, must inject safely.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client

        tc = _make_tool_call()
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[tc], reasoning_content=42),
            _make_response(content="done", tool_calls=[]),
        ]

        _, history = run(
            _persona(),
            [{"role": "user", "content": "go"}],
            max_tool_rounds=10,
            inject_reasoning=True,
            verbose=False,
        )

        reasoning_msgs = [
            m for m in history
            if m["role"] == "system" and m["content"].startswith(REASONING_INJECTION_PREFIX)
        ]
        assert len(reasoning_msgs) == 1
        assert "42" in reasoning_msgs[0]["content"]


# ---------------------------------------------------------------------------
# orchestrator — reasoning messages stripped from shared_history before review
# ---------------------------------------------------------------------------

class TestReasoningStrippedFromSharedHistory:

    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "IMPLEMENTER", {"name": "implementer", "model": "m", "system": "s"})
        monkeypatch.setattr(orch, "REVIEWERS", [{"name": "reviewer_a", "model": "m", "system": "s"}])

    def test_reviewer_messages_contain_no_reasoning_system_msgs(self, monkeypatch):
        from orchestrator.agent import _make_reasoning_message
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        # Implementer history contains a reasoning injection system message.
        impl_history = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "do task"},
            _make_reasoning_message("secret plan"),
            {"role": "assistant", "content": "IMPLEMENTATION_COMPLETE"},
        ]

        reviewer_messages_seen = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                return "IMPLEMENTATION_COMPLETE", impl_history
            # Capture what the reviewer actually receives.
            reviewer_messages_seen.extend(messages)
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do task", "/fake/repo", verbose=False)

        reasoning_in_reviewer = [
            m for m in reviewer_messages_seen
            if m.get("role") == "system" and m.get("content", "").startswith(REASONING_INJECTION_PREFIX)
        ]
        assert reasoning_in_reviewer == [], (
            f"Reviewer received reasoning system messages it should not see: {reasoning_in_reviewer}"
        )

    def test_rebuttal_shared_history_also_stripped(self, monkeypatch):
        from orchestrator.agent import _make_reasoning_message
        import orchestrator.orchestrator as orch

        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        impl_history_with_reasoning = [
            {"role": "system", "content": "s"},
            _make_reasoning_message("rebuttal plan"),
            {"role": "assistant", "content": "IMPLEMENTATION_COMPLETE"},
        ]

        call_num = [0]
        reviewer_messages_seen = []

        def fake_run(persona, messages, **kwargs):
            call_num[0] += 1
            if persona["name"] == "implementer":
                return "IMPLEMENTATION_COMPLETE", impl_history_with_reasoning
            reviewer_messages_seen.extend(messages)
            # First review: object to force a rebuttal; second: approve.
            if call_num[0] == 2:
                return "OBJECTION: fix it", [{"role": "system", "content": "s"}]
            return "APPROVED", [{"role": "system", "content": "s"}]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("do task", "/fake/repo", max_debate_rounds=1, verbose=False)

        reasoning_in_reviewer = [
            m for m in reviewer_messages_seen
            if m.get("role") == "system" and m.get("content", "").startswith(REASONING_INJECTION_PREFIX)
        ]
        assert reasoning_in_reviewer == [], (
            f"Reviewer received reasoning system messages after rebuttal: {reasoning_in_reviewer}"
        )
