"""
Tests for issue #133: malformed tool-call JSON from a model must not crash
agent.run().  Instead the error is fed back to the model as a tool result so
it can recover, and the loop continues normally.
"""

from unittest.mock import patch, MagicMock


def _make_tool_call(tc_id: str, name: str, arguments: str):
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


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


class TestMalformedToolJson:

    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_json_decode_error_does_not_crash(self, mock_client_factory):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        bad_tc = _make_tool_call("tc1", "bash_exec", '{"command": "echo hi"')  # missing closing brace
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[bad_tc]),
            _make_response(content="All done.", tool_calls=[]),
        ]

        text, _ = run(_persona(), [{"role": "user", "content": "go"}], max_tool_rounds=5)
        assert text == "All done."

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_error_result_appended_to_history(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        bad_tc = _make_tool_call("tc1", "bash_exec", "not json at all,,,")
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[bad_tc]),
            _make_response(content="Done.", tool_calls=[]),
        ]

        _, history = run(_persona(), [{"role": "user", "content": "go"}], max_tool_rounds=5)

        tool_results = [m for m in history if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert "ERROR" in tool_results[0]["content"]
        assert tool_results[0]["tool_call_id"] == "tc1"

    @patch("orchestrator.agent.T.execute", return_value="ok")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_execute_not_called_on_bad_json(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        # Kimi-style truncated string: missing comma delimiter mid-object
        bad_args = '{"command": "echo \\"hello world\\"  "content": "x"}'
        bad_tc = _make_tool_call("tc1", "bash_exec", bad_args)
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[bad_tc]),
            _make_response(content="Done.", tool_calls=[]),
        ]

        run(_persona(), [{"role": "user", "content": "go"}], max_tool_rounds=5)
        mock_execute.assert_not_called()

    @patch("orchestrator.agent.T.execute", return_value="tool output")
    @patch("orchestrator.agent.T.DEFS", [{"function": {"name": "bash_exec"}}])
    @patch("orchestrator.agent._client")
    def test_good_call_after_bad_call_executes(self, mock_client_factory, mock_execute):
        from orchestrator.agent import run

        client = MagicMock()
        mock_client_factory.return_value = client
        bad_tc = _make_tool_call("tc1", "bash_exec", "{broken")
        good_tc = _make_tool_call("tc2", "bash_exec", '{"command": "echo ok"}')
        client.chat.completions.create.side_effect = [
            _make_response(content=None, tool_calls=[bad_tc]),
            _make_response(content=None, tool_calls=[good_tc]),
            _make_response(content="Finished.", tool_calls=[]),
        ]

        text, history = run(_persona(), [{"role": "user", "content": "go"}], max_tool_rounds=5)

        assert text == "Finished."
        mock_execute.assert_called_once()
        tool_results = [m for m in history if m.get("role") == "tool"]
        assert len(tool_results) == 2
        assert "ERROR" in tool_results[0]["content"]
        assert tool_results[1]["content"] == "tool output"
