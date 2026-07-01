"""
Unit tests for orchestrator/test_model_compat.py.

All OpenAI API calls are mocked; no network access is required.
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on the path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import orchestrator.test_model_compat as compat


# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI response objects
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: str, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_response(tool_calls=None, content: str = "") -> MagicMock:
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_client(responses: list) -> MagicMock:
    """Return a mock OpenAI client whose create() yields responses in order."""
    client = MagicMock()
    client.chat.completions.create.side_effect = responses
    return client


# ---------------------------------------------------------------------------
# _is_markdown_wrapped
# ---------------------------------------------------------------------------

class TestIsMarkdownWrapped:
    def test_bare_json_is_not_wrapped(self):
        assert not compat._is_markdown_wrapped('{"command": "echo hi"}')

    def test_backtick_block_is_wrapped(self):
        assert compat._is_markdown_wrapped('```json\n{"command": "echo hi"}\n```')

    def test_plain_backtick_block_is_wrapped(self):
        assert compat._is_markdown_wrapped('```\n{"command": "echo hi"}\n```')

    def test_empty_string_is_not_wrapped(self):
        assert not compat._is_markdown_wrapped("")


# ---------------------------------------------------------------------------
# check_tool_call
# ---------------------------------------------------------------------------

class TestCheckToolCall:
    def test_correct_tool_call_passes(self, capsys):
        tc = _make_tool_call("bash_exec", '{"command": "echo hello"}')
        client = _make_client([_make_response(tool_calls=[tc])])
        result = compat.check_tool_call(client, "test-model")
        assert result is True
        out = capsys.readouterr().out
        assert "[PASS]" in out
        assert "[FAIL]" not in out

    def test_no_tool_call_fails(self, capsys):
        client = _make_client([_make_response(content="I can help with that!")])
        result = compat.check_tool_call(client, "test-model")
        assert result is False
        out = capsys.readouterr().out
        assert "[FAIL] tool call issued" in out

    def test_wrong_tool_name_fails(self, capsys):
        tc = _make_tool_call("bash", '{"command": "echo hello"}')
        client = _make_client([_make_response(tool_calls=[tc])])
        result = compat.check_tool_call(client, "test-model")
        assert result is False
        out = capsys.readouterr().out
        assert "[FAIL] tool name is exact registered name" in out
        assert "got 'bash'" in out

    def test_markdown_wrapped_arguments_fails(self, capsys):
        args = '```json\n{"command": "echo hello"}\n```'
        tc = _make_tool_call("bash_exec", args)
        client = _make_client([_make_response(tool_calls=[tc])])
        result = compat.check_tool_call(client, "test-model")
        assert result is False
        out = capsys.readouterr().out
        assert "[FAIL] arguments not markdown-wrapped" in out

    def test_invalid_json_arguments_fails(self, capsys):
        tc = _make_tool_call("bash_exec", "not valid json")
        client = _make_client([_make_response(tool_calls=[tc])])
        result = compat.check_tool_call(client, "test-model")
        assert result is False
        out = capsys.readouterr().out
        assert "[FAIL] arguments are valid JSON" in out


# ---------------------------------------------------------------------------
# check_reviewer_verdict
# ---------------------------------------------------------------------------

class TestCheckReviewerVerdict:
    def test_approved_verdict_passes(self, capsys):
        client = _make_client([_make_response(content="Looks good.\n\nAPPROVED")])
        result = compat.check_reviewer_verdict(client, "test-model")
        assert result is True
        assert "[PASS]" in capsys.readouterr().out

    def test_objection_verdict_passes(self, capsys):
        client = _make_client([
            _make_response(content="Missing error handling.\n\nOBJECTION: line 5 has no try/except")
        ])
        result = compat.check_reviewer_verdict(client, "test-model")
        assert result is True
        assert "[PASS]" in capsys.readouterr().out

    def test_no_verdict_fails(self, capsys):
        client = _make_client([_make_response(content="Hmm, looks interesting but I am not sure.")])
        result = compat.check_reviewer_verdict(client, "test-model")
        assert result is False
        out = capsys.readouterr().out
        assert "[FAIL] response contains APPROVED or OBJECTION" in out

    def test_verdict_case_insensitive(self, capsys):
        # The regex uses IGNORECASE so lowercase variants still count.
        client = _make_client([_make_response(content="objection: missing test")])
        result = compat.check_reviewer_verdict(client, "test-model")
        assert result is True


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_prints_default_and_fast_model(self, capsys):
        compat.list_models()
        out = capsys.readouterr().out
        assert "DEFAULT_MODEL" in out
        assert "FAST_MODEL" in out
        assert "anthropic/claude-sonnet-4-6" in out
        assert "anthropic/claude-haiku-4-5" in out


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

class TestMain:
    def _run_main(self, argv: list[str]) -> int:
        with patch("sys.argv", ["test_model_compat.py"] + argv):
            return compat.main()

    def test_list_models_exits_zero(self, capsys):
        result = self._run_main(["--list-models"])
        assert result == 0
        assert "DEFAULT_MODEL" in capsys.readouterr().out

    def test_bad_provider_exits_one(self, capsys):
        result = self._run_main(["--model", "foo/bar", "--provider", "nonexistent"])
        assert result == 1

    def test_all_pass_exits_zero(self, capsys):
        tc = _make_tool_call("bash_exec", '{"command": "echo hello"}')
        tool_response = _make_response(tool_calls=[tc])
        verdict_response = _make_response(content="APPROVED")

        with (
            patch("orchestrator.test_model_compat._make_client") as mock_make,
        ):
            mock_client = _make_client([tool_response, verdict_response])
            mock_make.return_value = mock_client
            result = self._run_main(["--model", "test/model", "--provider", "litellm"])

        assert result == 0

    def test_any_fail_exits_one(self, capsys):
        # Tool call returns text, no tool call → check 1 fails.
        tool_response = _make_response(content="I can't do that.")
        verdict_response = _make_response(content="APPROVED")

        with patch("orchestrator.test_model_compat._make_client") as mock_make:
            mock_client = _make_client([tool_response, verdict_response])
            mock_make.return_value = mock_client
            result = self._run_main(["--model", "test/model", "--provider", "litellm"])

        assert result == 1
