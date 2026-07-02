"""
Tests for issue #134: <think>...</think> blocks emitted by reasoning models
must be stripped before the harness decides whether the turn is complete.
"""

from unittest.mock import patch, MagicMock


def _make_response(content=None, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _persona(name="test_agent"):
    return {"name": name, "model": "fake-model", "system": "You are a test agent."}


class TestStripThink:

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_think_block_stripped_from_returned_content(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="<think>internal reasoning</think>Here is my answer.",
            tool_calls=[],
        )

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}])
        assert text == "Here is my answer."

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_bare_close_tag_stripped(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="</think>Actual response.",
            tool_calls=[],
        )

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}])
        assert text == "Actual response."

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_only_think_content_causes_reinvocation(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.side_effect = [
            _make_response(content="<think>still thinking…</think>", tool_calls=[]),
            _make_response(content="Done.", tool_calls=[]),
        ]

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}], max_tool_rounds=5)
        assert text == "Done."
        assert client.chat.completions.create.call_count == 2

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_bare_close_tag_only_causes_reinvocation(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.side_effect = [
            _make_response(content="</think>", tool_calls=[]),
            _make_response(content="IMPLEMENTATION_COMPLETE", tool_calls=[]),
        ]

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}], max_tool_rounds=5)
        assert text == "IMPLEMENTATION_COMPLETE"
        assert client.chat.completions.create.call_count == 2

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_multiline_think_block_stripped(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="<think>\nStep 1: analyse\nStep 2: plan\n</think>\nFinal answer.",
            tool_calls=[],
        )

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}])
        assert text == "Final answer."

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_stripped_content_stored_in_history(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="<think>secret</think>Public answer.",
            tool_calls=[],
        )

        _, history = run(_persona(), [{"role": "user", "content": "hi"}])
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Public answer."
        assert "<think>" not in assistant_msgs[0]["content"]

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_reinvocation_leaves_no_empty_assistant_entry(self, mock_client_factory):
        # An empty assistant entry in history causes API 400s on strict providers.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.side_effect = [
            _make_response(content="<think>reasoning</think>", tool_calls=[]),
            _make_response(content="Real answer.", tool_calls=[]),
        ]

        _, history = run(_persona(), [{"role": "user", "content": "hi"}], max_tool_rounds=5)
        assistant_msgs = [m for m in history if m["role"] == "assistant"]
        assert all(m["content"] for m in assistant_msgs), (
            "empty assistant entry found in history"
        )

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash"}}])
    @patch("orchestrator.agent._client")
    def test_mid_sentence_think_block_preserves_word_boundary(self, mock_client_factory):
        # Replacing with "" would concatenate adjacent words; replace with " " instead.
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        client.chat.completions.create.return_value = _make_response(
            content="word<think>x</think>word",
            tool_calls=[],
        )

        text, _ = run(_persona(), [{"role": "user", "content": "hi"}])
        assert text == "word word"
